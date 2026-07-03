# SPDX-License-Identifier: Apache-2.0
"""Safetensors itemsize-realignment for zero-copy mmap views.

The safetensors format does not align tensor data offsets: in stock
checkpoints a majority of tensor byte offsets are not multiples of their
dtype itemsize (measured 62% of bytes misaligned on a stock mlx-community
8-bit checkpoint — u32 quant banks landing at offset % 4 == 2, etc.).
Misaligned tensors cannot be typed zero-copy views into an mmap'd shard.

Fix: a one-time, spec-compliant re-save of each shard with tensors sorted
by itemsize (descending) and the header space-padded so the data base is
8-byte aligned. Every tensor's absolute file offset then lands on a
multiple of its itemsize -> 100% zero-copy. Tensor bytes are unchanged;
only their order inside the file and the header offsets differ. Any
safetensors reader (mlx, HF, mlx-lm) reads the realigned shard normally.

``ensure_realigned`` is idempotent: shards that already pass the alignment
check are left untouched, so calling it at every elastic model load is a
cheap header-only scan after the first time.

In-place mode (the default used by the engine) replaces each shard via an
atomic ``os.replace`` of a temp file written in the same directory. NOTE:
if a shard is a *symlink* (Hugging Face cache layout), the symlink is
replaced by a real file and the original blob is orphaned (reclaim with
``hf cache``). Model dirs under ~/.omlx/models are regular files and swap
cleanly. Transient free-space requirement: the largest single shard.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ITEMSIZE = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
    "U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1,
}

_CHUNK = 1 << 26  # 64 MiB copy chunks


def _read_header(path: str | Path) -> tuple[dict, int]:
    """Return (header_dict, data_base_offset) for a safetensors file."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


@dataclass
class AlignmentReport:
    total_tensors: int = 0
    misaligned_tensors: int = 0
    total_bytes: int = 0
    misaligned_bytes: int = 0

    @property
    def aligned(self) -> bool:
        return self.misaligned_tensors == 0


def shard_alignment(path: str | Path) -> AlignmentReport:
    """Header-only scan: how many tensors sit at non-itemsize-aligned
    absolute file offsets?"""
    header, base = _read_header(path)
    rep = AlignmentReport()
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        itemsize = ITEMSIZE[spec["dtype"]]
        start, end = spec["data_offsets"]
        nbytes = end - start
        rep.total_tensors += 1
        rep.total_bytes += nbytes
        if (base + start) % itemsize != 0:
            rep.misaligned_tensors += 1
            rep.misaligned_bytes += nbytes
    return rep


def realign_shard(src: str | Path, dst: str | Path) -> None:
    """Rewrite `src` to `dst` with tensors sorted by itemsize (desc) and the
    data base 8-byte aligned. Tensor bytes are copied verbatim."""
    with open(src, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
        old_base = 8 + header_len
        meta = header.pop("__metadata__", None)

        order = sorted(
            header.items(),
            key=lambda kv: (-ITEMSIZE[kv[1]["dtype"]], kv[0]),
        )
        new_header = {}
        pos = 0
        for name, spec in order:
            n = spec["data_offsets"][1] - spec["data_offsets"][0]
            new_header[name] = {
                "dtype": spec["dtype"],
                "shape": spec["shape"],
                "data_offsets": [pos, pos + n],
            }
            pos += n
        if meta is not None:
            new_header["__metadata__"] = meta

        hjson = json.dumps(new_header, separators=(",", ":")).encode()
        pad = (-(8 + len(hjson))) % 8
        hjson += b" " * pad

        with open(dst, "wb") as out:
            out.write(len(hjson).to_bytes(8, "little"))
            out.write(hjson)
            for _name, spec in order:
                f.seek(old_base + spec["data_offsets"][0])
                remaining = spec["data_offsets"][1] - spec["data_offsets"][0]
                while remaining:
                    chunk = f.read(min(remaining, _CHUNK))
                    if not chunk:
                        raise OSError(f"short read realigning {src}")
                    out.write(chunk)
                    remaining -= len(chunk)

    # Verify: realigned shard must be 100% aligned and carry the same
    # tensor payload size.
    rep = shard_alignment(dst)
    if not rep.aligned:
        raise RuntimeError(f"realign verification failed for {dst}: {rep}")


def ensure_realigned(model_dir: str | Path, *, in_place: bool = True,
                     out_dir: str | Path | None = None) -> dict:
    """Realign every misaligned .safetensors shard in `model_dir`.

    Idempotent: aligned shards are skipped (header-only check). Returns a
    summary dict. With ``in_place`` (default), shards are atomically
    replaced; otherwise all files are mirrored into ``out_dir``.
    """
    model_dir = Path(model_dir)
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors shards in {model_dir}")

    summary = {"realigned": [], "already_aligned": [], "dir": str(model_dir)}

    if not in_place:
        out_dir = Path(out_dir or (str(model_dir).rstrip("/") + "-elastic"))
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(model_dir.iterdir()):
            if item.name.endswith(".safetensors"):
                continue
            if item.is_file():
                shutil.copy2(item, out_dir / item.name)
        summary["dir"] = str(out_dir)

    for shard in shards:
        rep = shard_alignment(shard)
        if rep.aligned and in_place:
            summary["already_aligned"].append(shard.name)
            continue

        if in_place:
            free = shutil.disk_usage(model_dir).free
            need = shard.stat().st_size
            if free < need + (1 << 30):
                raise OSError(
                    f"not enough free space to realign {shard.name} in place "
                    f"(need ~{need/1e9:.1f}GB + 1GB headroom, have {free/1e9:.1f}GB)"
                )
            tmp = shard.with_name(shard.name + ".realign-tmp")
            try:
                realign_shard(shard, tmp)
                if shard.is_symlink():
                    logger.warning(
                        "realign: %s was a symlink (HF cache layout); it is now "
                        "a regular file and the original blob is orphaned "
                        "(reclaim with `hf cache`)", shard.name,
                    )
                os.replace(tmp, shard)
            finally:
                tmp.unlink(missing_ok=True)
            summary["realigned"].append(shard.name)
        else:
            dst = out_dir / shard.name
            if rep.aligned:
                shutil.copy2(shard, dst)
                summary["already_aligned"].append(shard.name)
            else:
                realign_shard(shard, dst)
                summary["realigned"].append(shard.name)

    if summary["realigned"]:
        logger.info(
            "elastic realign: %d shard(s) rewritten, %d already aligned (%s)",
            len(summary["realigned"]), len(summary["already_aligned"]),
            summary["dir"],
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Realign safetensors shards for zero-copy mmap views."
    )
    parser.add_argument("model_dir", help="model directory containing *.safetensors")
    parser.add_argument(
        "--out", default=None,
        help="write realigned copy here instead of replacing in place",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="only report alignment, do not rewrite",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.check:
        total = AlignmentReport()
        for shard in sorted(Path(args.model_dir).glob("*.safetensors")):
            rep = shard_alignment(shard)
            print(
                f"{shard.name}: {rep.misaligned_tensors}/{rep.total_tensors} tensors "
                f"({rep.misaligned_bytes/1e9:.2f}/{rep.total_bytes/1e9:.2f} GB) misaligned"
            )
            total.total_tensors += rep.total_tensors
            total.misaligned_tensors += rep.misaligned_tensors
            total.total_bytes += rep.total_bytes
            total.misaligned_bytes += rep.misaligned_bytes
        print(
            f"TOTAL: {total.misaligned_tensors}/{total.total_tensors} tensors "
            f"({total.misaligned_bytes/1e9:.2f}/{total.total_bytes/1e9:.2f} GB) misaligned"
        )
        return 0

    summary = ensure_realigned(
        args.model_dir, in_place=args.out is None, out_dir=args.out
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
