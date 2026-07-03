# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.elastic — mmap-elastic weight loading.

Covers the pure-Python pieces unconditionally (state registry, realign,
profile/settings flag plumbing) and the native zero-copy path when the
_elastic_mmap extension is built (scripts/build-elastic-ext.sh); native
tests skip cleanly otherwise.
"""

import json

import mlx.core as mx
import numpy as np
import pytest

from omlx.elastic import loader as elastic_loader
from omlx.elastic import realign, state

requires_native = pytest.mark.skipif(
    not elastic_loader.is_available(),
    reason="_elastic_mmap extension not built (scripts/build-elastic-ext.sh)",
)


@pytest.fixture(autouse=True)
def _reset_elastic_state():
    """Keep the sticky process-wide elastic state from leaking across tests."""
    yield
    state._wiring_blocked = False
    state._wiring_blocked_reason = ""
    state._elastic_model_dirs.clear()
    mx.load = elastic_loader._REAL_MX_LOAD


def _write_misaligned_shard(path):
    """Hand-write a safetensors shard whose f32/u32 tensors sit at offsets
    that are NOT multiples of their itemsize (the stock-checkpoint trap)."""
    rng = np.random.default_rng(7)
    tensors = {
        "oddball": rng.integers(0, 255, size=(3,), dtype=np.uint8),  # 3 bytes
        "weights": rng.standard_normal((4, 8)).astype(np.float32),
        "quants": rng.integers(0, 2**31, size=(16,), dtype=np.uint32),
    }
    header = {}
    pos = 0
    for name, arr in tensors.items():  # insertion order => u8 first
        n = arr.nbytes
        header[name] = {
            "dtype": {"uint8": "U8", "float32": "F32", "uint32": "U32"}[str(arr.dtype)],
            "shape": list(arr.shape),
            "data_offsets": [pos, pos + n],
        }
        pos += n
    hjson = json.dumps(header, separators=(",", ":")).encode()
    pad = (-(8 + len(hjson))) % 8  # data base 8-aligned => oddball breaks f32
    hjson += b" " * pad
    with open(path, "wb") as f:
        f.write(len(hjson).to_bytes(8, "little"))
        f.write(hjson)
        for arr in tensors.values():
            f.write(arr.tobytes())
    return tensors


class TestRealign:
    def test_detects_misalignment(self, tmp_path):
        shard = tmp_path / "model.safetensors"
        _write_misaligned_shard(shard)
        rep = realign.shard_alignment(shard)
        assert not rep.aligned
        assert rep.misaligned_tensors >= 1  # f32/u32 after the 3-byte u8

    def test_realign_in_place_is_idempotent_and_preserves_bytes(self, tmp_path):
        shard = tmp_path / "model.safetensors"
        ref = _write_misaligned_shard(shard)

        summary = realign.ensure_realigned(tmp_path)
        assert summary["realigned"] == ["model.safetensors"]
        assert realign.shard_alignment(shard).aligned

        # Tensor payloads byte-identical through any stock reader.
        loaded = mx.load(str(shard))
        for name, arr in ref.items():
            assert np.array_equal(np.array(loaded[name]), arr), name

        # Second call: header-only scan, no rewrite.
        summary2 = realign.ensure_realigned(tmp_path)
        assert summary2["realigned"] == []
        assert summary2["already_aligned"] == ["model.safetensors"]

    def test_realign_to_out_dir(self, tmp_path):
        src = tmp_path / "m"
        src.mkdir()
        _write_misaligned_shard(src / "model.safetensors")
        (src / "config.json").write_text("{}")
        out = tmp_path / "m-elastic"
        summary = realign.ensure_realigned(src, in_place=False, out_dir=out)
        assert summary["dir"] == str(out)
        assert realign.shard_alignment(out / "model.safetensors").aligned
        assert (out / "config.json").exists()
        # Source untouched.
        assert not realign.shard_alignment(src / "model.safetensors").aligned


class TestState:
    def test_registry_and_assert_mutable(self, tmp_path):
        state.register_elastic_model(str(tmp_path))
        assert state.is_elastic_model_dir(str(tmp_path))
        with pytest.raises(RuntimeError, match="read-only"):
            state.assert_mutable(str(tmp_path), "fuse_lora")
        # Non-elastic dirs pass.
        state.assert_mutable("/nonexistent", "fuse_lora")

    def test_block_wiring_sticky(self):
        assert not state.wiring_blocked()
        state.block_wiring("test reason")
        assert state.wiring_blocked()
        assert state.wiring_blocked_reason() == "test reason"

    def test_enforcer_skips_raise_when_blocked(self):
        from omlx.process_memory_enforcer import _apply_metal_wired_limit

        state.block_wiring("elastic test")
        applied, previous = _apply_metal_wired_limit(42 * 1024**3)
        assert applied == 0
        assert previous is None


@requires_native
class TestNativeLoader:
    def test_elastic_load_matches_stock(self, tmp_path):
        shard = tmp_path / "model.safetensors"
        _write_misaligned_shard(shard)
        realign.ensure_realigned(tmp_path)

        elastic = elastic_loader.load_shard_elastic(str(shard))
        stock = mx.load(str(shard))
        assert set(elastic) == set(stock)
        for name in stock:
            assert bool(mx.array_equal(elastic[name], stock[name]).item()), name

    def test_misaligned_falls_back_to_copy(self, tmp_path):
        shard = tmp_path / "model.safetensors"
        ref = _write_misaligned_shard(shard)  # NOT realigned

        elastic = elastic_loader.load_shard_elastic(str(shard))
        for name, arr in ref.items():
            assert np.array_equal(np.array(elastic[name]), arr), name

    def test_scope_patches_and_restores(self, tmp_path):
        model_dir = tmp_path / "model"
        other_dir = tmp_path / "other"
        model_dir.mkdir()
        other_dir.mkdir()
        _write_misaligned_shard(model_dir / "model.safetensors")
        ref_other = _write_misaligned_shard(other_dir / "model.safetensors")
        realign.ensure_realigned(model_dir)

        original = mx.load
        with elastic_loader.elastic_load_scope(str(model_dir)):
            assert mx.load is not original
            got = mx.load(str(model_dir / "model.safetensors"))
            assert got  # elastic views
            # Out-of-scope path uses the real loader.
            other = mx.load(str(other_dir / "model.safetensors"))
            for name, arr in ref_other.items():
                assert np.array_equal(np.array(other[name]), arr)
        assert mx.load is original
        assert state.is_elastic_model_dir(str(model_dir))
        assert state.wiring_blocked()

    def test_scope_not_reentrant(self, tmp_path):
        (tmp_path / "model.safetensors").touch()
        with elastic_loader.elastic_load_scope(str(tmp_path)):
            with pytest.raises(RuntimeError, match="not reentrant"):
                with elastic_loader.elastic_load_scope(str(tmp_path)):
                    pass
