# SPDX-License-Identifier: Apache-2.0
"""Elastic (mmap) safetensors loading: file-backed, OS-reclaimable weights.

The mechanism (validated 2026-07-02 on Qwen3-Coder-30B-8bit: decode parity
with stock at ~112 tok/s, physical footprint 0.49GB vs 30.7GB, survives
memory pressure that craters the stock loader):

- ``_elastic_mmap.wrap_file``: one MAP_SHARED|PROT_READ mmap per shard,
  wrapped zero-copy into a Metal buffer (newBufferWithBytesNoCopy).
- ``_elastic_mmap.make_view``: each tensor is a typed byte-offset view
  sharing that buffer (public ``array::copy_shared_buffer``). No copies;
  pages stay clean and file-backed, so the OS can evict and refault them
  instead of swapping.

Use :func:`elastic_load_scope` around an ``mlx_lm.load`` call. The scope
temporarily patches ``mx.load`` so that safetensors shards *inside the
scoped model directory* load elastically; everything else falls through to
the real ``mx.load``. Loads in oMLX are serialized on the global MLX
executor thread (engine_core.get_mlx_executor), which makes the temporary
patch race-free in practice; a module lock additionally rejects nested or
concurrent scopes outright.

Requirements:
- The native extension must be built: ``scripts/build-elastic-ext.sh``.
- Shards should be itemsize-aligned (``omlx.elastic.realign``). Misaligned
  tensors fall back to a stock copy (counted + logged) — correct but not
  elastic for those bytes.

Constraint: weights are READ-ONLY (PROT_READ). See state.assert_mutable.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager

import mlx.core as mx

from . import state

logger = logging.getLogger(__name__)

_scope_lock = threading.Lock()

# Import-time probe result cache: None = untried, module or Exception after.
_native = None
_native_error: Exception | None = None


class ElasticUnavailableError(RuntimeError):
    pass


def _get_native():
    """Import the _elastic_mmap extension, with a clear error if unbuilt."""
    global _native, _native_error
    if _native is not None:
        return _native
    if _native_error is not None:
        raise ElasticUnavailableError(str(_native_error)) from _native_error
    try:
        from . import _elastic_mmap  # type: ignore[attr-defined]

        _native = _elastic_mmap
        return _native
    except ImportError as e:
        _native_error = ElasticUnavailableError(
            "elastic_load requires the _elastic_mmap native extension. "
            "Build it with: scripts/build-elastic-ext.sh "
            f"(import error: {e})"
        )
        raise _native_error from e


def is_available() -> bool:
    try:
        _get_native()
        return True
    except ElasticUnavailableError:
        return False


def load_shard_elastic(path: str) -> dict[str, mx.array]:
    """Load one safetensors shard as zero-copy mmap views.

    Misaligned tensors (shard not realigned) fall back to a one-shot stock
    ``mx.load`` of the shard and are materialized copies; a summary WARNING
    is emitted because those bytes are dirty anonymous memory, not elastic.
    """
    native = _get_native()

    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    base = 8 + header_len

    parent = native.wrap_file(path, True)
    out: dict[str, mx.array] = {}
    fallback_shard: dict[str, mx.array] | None = None
    fallbacks = 0

    for name, spec in header.items():
        if name == "__metadata__":
            continue
        byte_off = base + spec["data_offsets"][0]
        shape = spec["shape"] or [1]
        try:
            out[name] = native.make_view(parent, byte_off, shape, spec["dtype"])
        except RuntimeError:
            if fallback_shard is None:
                fallback_shard = _REAL_MX_LOAD(path)
            out[name] = fallback_shard[name]
            fallbacks += 1

    if fallbacks:
        logger.warning(
            "elastic: %d tensor(s) in %s misaligned -> stock copies (run "
            "omlx.elastic.realign on this model for 100%% elastic loading)",
            fallbacks,
            os.path.basename(path),
        )
    # Parent custody is shared through each view's Data; no need to keep
    # a reference to `parent` here.
    return out


# The genuine mx.load, captured at import so scopes can nest with fallbacks
# and load_shard_elastic can do stock fallback loads while the patch is live.
_REAL_MX_LOAD = mx.load


def _clamp_wired_limit(model_dir: str) -> None:
    """Block + undo Metal wiring before any elastic mapping exists.

    MLX's residency set wires buffers up to the process wired limit. Our
    mappings enter that set at load; if the limit is already raised (oMLX's
    process_memory_enforcer mirrors iogpu.wired_limit_mb at startup), the
    checkpoint pages would be wired resident and the OS could no longer
    evict them — the entire point of elastic loading. Clamp to 0 (MLX
    default = nothing wired) and mark the process so the enforcer does not
    raise it again. Process-wide trade-off, logged: stock co-tenant models
    in this process also lose wiring.
    """
    state.block_wiring(
        f"elastic model loaded from {model_dir} (file-backed weights must "
        "stay evictable)"
    )
    try:
        previous = mx.set_wired_limit(0)
        if previous:
            logger.info(
                "elastic: Metal wired limit lowered %.1fGB -> 0 for this "
                "process (elastic weights must stay evictable; stock "
                "co-tenant models are also unwired)",
                previous / 1e9,
            )
    except Exception as e:  # noqa: BLE001 — older macOS / no Metal
        logger.debug("elastic: mx.set_wired_limit(0) unavailable: %s", e)


@contextmanager
def elastic_load_scope(model_dir: str):
    """Patch ``mx.load`` so this model's shards load as elastic mmap views.

    Scoped: only ``*.safetensors`` files under ``model_dir`` (realpath
    prefix match) are affected; all other loads use the real ``mx.load``.
    Restores ``mx.load`` on exit. Not reentrant.
    """
    _get_native()  # fail fast (and loudly) before touching anything

    root = os.path.realpath(str(model_dir))

    if not _scope_lock.acquire(blocking=False):
        raise RuntimeError(
            "elastic_load_scope is not reentrant; model loads must be "
            "serialized (oMLX serializes them on the global MLX executor)"
        )
    try:
        _clamp_wired_limit(str(model_dir))

        def _in_scope(p: str) -> bool:
            if not p.endswith(".safetensors"):
                return False
            # Compare the shard's *containing directory* (resolved), so both
            # regular files and HF-cache symlinked shards (whose own realpath
            # points into blobs/) match the scoped model dir.
            d = os.path.realpath(os.path.dirname(p))
            return d == root or d.startswith(root + os.sep)

        def _patched_load(path, *args, **kwargs):
            p = str(path)
            if _in_scope(p):
                return load_shard_elastic(p)
            return _REAL_MX_LOAD(path, *args, **kwargs)

        mx.load = _patched_load
        try:
            yield
        finally:
            mx.load = _REAL_MX_LOAD
        state.register_elastic_model(str(model_dir))
    finally:
        _scope_lock.release()
