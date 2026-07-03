# omlx.elastic — mmap-elastic weight loading

File-backed, OS-reclaimable model weights for oMLX. Enables models near or
beyond the comfortable RAM budget (e.g. an 85GB checkpoint on a 128GB Mac
with co-tenants) and graceful behavior under memory pressure.

## Mechanism

llama.cpp-style, applied to MLX + safetensors:

1. **`_elastic_mmap.wrap_file`** — one `MAP_SHARED|PROT_READ` mmap per
   safetensors shard, wrapped zero-copy into a Metal buffer via MLX's
   public `mlx::allocator::make_buffer` (`newBufferWithBytesNoCopy`).
2. **`_elastic_mmap.make_view`** — every tensor is a typed byte-offset view
   sharing that buffer (public `array::copy_shared_buffer`). No copies.
3. Pages are **clean + file-backed**: under memory pressure the OS evicts
   them at zero cost and refaults them from the checkpoint file, instead of
   swapping dirty anonymous pages. Physical footprint after load+generate is
   ~0.5GB for a 30GB model (page cache holds the hot weights, charged to the
   system, evictable — not to the process).

Measured (Qwen3-Coder-30B-8bit, this box, 2026-07-02, production path):

| | stock | elastic |
|---|---|---|
| decode | 100.1 tok/s | 110.1 tok/s |
| phys_footprint after load+generate | 32.9 GB | 0.51 GB |
| tensor bytes | — | byte-identical (spot-checked) |

## Usage

Per-model opt-in, **default OFF**. Set `elastic_load: true` in the model's
settings (admin UI / model settings JSON / profile). When off, the load
path is byte-identical to stock oMLX — this package is not even imported.

Build the native extension once per mlx-wheel bump:

```bash
scripts/build-elastic-ext.sh            # uses .venv/bin/python
scripts/build-elastic-ext.sh /path/python  # explicit env
```

Requirements: cmake >= 3.25, C++17 toolchain, network on first run
(FetchContent pins nanobind v2.12.0 — MUST match the mlx wheel's nanobind:
NB_STATIC, STABLE_ABI, NB_DOMAIN mlx, or `mx.array` objects cannot cross
the module boundary).

If the flag is on but the extension is missing, the engine logs an ERROR
and falls back to the stock loader (the model still serves, non-elastically).

## Realignment (one-time, idempotent)

safetensors does not align tensor offsets; in stock checkpoints most tensor
byte offsets are not multiples of their dtype itemsize (62% of bytes on a
stock mlx-community 8-bit checkpoint), and misaligned tensors cannot be
zero-copy typed views. `omlx.elastic.realign` rewrites each shard once —
tensors sorted by itemsize desc, header padded to an 8-byte data base —
after which 100% of tensors are aligned. Spec-compliant, byte-identical
payloads, readable by any safetensors tool.

The engine runs `ensure_realigned()` automatically at elastic load; after
the first time it is a header-only scan. Manual use:

```bash
python -m omlx.elastic.realign --check <model-dir>   # report only
python -m omlx.elastic.realign <model-dir>           # in-place, atomic per shard
python -m omlx.elastic.realign <model-dir> --out DIR # keep original
```

In-place realign of a Hugging Face *cache* snapshot replaces shard symlinks
with real files (original blobs orphaned; reclaim with `hf cache`). Dirs
under `~/.omlx/models/` are regular files and swap cleanly. Transient free
space needed: the largest single shard.

## Constraints

- **Weights are READ-ONLY** (`PROT_READ`). Any weight-mutating path —
  fused LoRA, in-place requantization, weight updates — must materialize a
  private copy first or it will SIGBUS/fault. Elastic-loaded model dirs are
  registered in `omlx.elastic.state`; mutating code must call
  `state.assert_mutable(model_dir, op)` before touching weights. Inference
  (including TurboQuant KV, which touches only activations/cache) is
  unaffected.
- **Metal wiring is disabled process-wide** while an elastic model is
  loaded. MLX inserts all buffers into its residency set; a raised wired
  limit would ask the OS to pin them, exactly what elastic weights must
  avoid. The loader clamps `mx.set_wired_limit(0)` before mapping and
  `process_memory_enforcer` skips subsequent raises
  (`omlx.elastic.state.wiring_blocked()`). Sticky until process restart.
  Trade-off: stock co-tenant models in the same process also lose wiring
  (measured on this box: residency-set membership never charged the
  file-backed pages to the process footprint even with the limit raised,
  so the clamp is defense-in-depth — but wiring semantics are Apple's to
  change, so we keep it).
- **Text models via BatchedEngine only** (the `mlx_lm.load` path). VLM /
  dflash / STT / TTS engines use separate loaders and ignore the flag.
- **Custom-quantization models** (e.g. ParoQuant) bypass elastic: their
  loaders requantize at load time (weight-mutating) and return before the
  elastic scope.
- **Decode speed depends on SSD refault throughput under pressure.** With
  free RAM, pages live in the unified page cache at full speed (parity+
  with stock). Under sustained incompressible pressure the model degrades
  gracefully (refaults from SSD) instead of being OOM-killed.

## Files

- `native/elastic_mmap.cpp` + `native/CMakeLists.txt` — the extension
  (~190 lines, public MLX C++ API only).
- `loader.py` — safetensors header parse, per-tensor views,
  `elastic_load_scope` (scoped `mx.load` patch), wired-limit clamp.
- `realign.py` — alignment check + one-time realigner + CLI.
- `state.py` — process-wide wiring block + read-only model registry
  (stdlib-only; importable from the enforcer at boot with zero cost).
- Engine hook: `omlx/engine/batched.py` (`_load_model_sync`), gated on
  `ModelSettings.elastic_load`.
- Enforcer guard: `omlx/process_memory_enforcer.py`
  (`_apply_metal_wired_limit`).

Validation receipts: `experiments/mmap-elastic-prototype/RESULTS-85gb.md`
(the 85GB Qwen3-Coder-Next case: loads + generates where the stock loader
OOMs) and `tests/test_elastic_loader.py`.
