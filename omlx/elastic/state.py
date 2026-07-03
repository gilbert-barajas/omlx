# SPDX-License-Identifier: Apache-2.0
"""Process-wide state for elastic (mmap) weight loading.

Pure-stdlib module (no mlx import) so it is safe to import from anywhere,
including process_memory_enforcer at module import time.

Two responsibilities:

1. **Wiring block.** Elastic weights are file-backed PROT_READ mappings
   whose pages must stay *clean and evictable*. MLX inserts every buffer —
   including our zero-copy wraps — into its Metal residency set; raising the
   process wired limit (``mx.set_wired_limit``) makes that set wire its
   buffers, which would pin the mapped checkpoint in physical memory and
   defeat elasticity. The wired limit is per-process, so once any elastic
   model is loaded, the whole process must stop raising it. The elastic
   loader calls :func:`block_wiring` before mapping anything;
   ``process_memory_enforcer._apply_metal_wired_limit`` checks
   :func:`wiring_blocked` and skips.

2. **Read-only weight registry.** Elastic weights live in PROT_READ
   mappings: any in-place weight mutation (fused LoRA, requantization,
   weight update) would SIGBUS on CPU or corrupt/fault on GPU. Model dirs
   loaded elastically are registered here; mutating paths must call
   :func:`assert_mutable` (or check :func:`is_elastic_model_dir`) and
   either refuse or materialize a private copy first.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_wiring_blocked = False
_wiring_blocked_reason = ""
_elastic_model_dirs: set[str] = set()


def block_wiring(reason: str) -> None:
    """Mark this process as no-Metal-wiring. Sticky until process exit.

    Elastic mappings can outlive an engine unload (custody is shared with
    any surviving array view), so the block is deliberately never lifted.
    """
    global _wiring_blocked, _wiring_blocked_reason
    with _lock:
        if not _wiring_blocked:
            _wiring_blocked = True
            _wiring_blocked_reason = reason
            logger.info(
                "Metal wired-limit raises are now blocked for this process: %s",
                reason,
            )


def wiring_blocked() -> bool:
    return _wiring_blocked


def wiring_blocked_reason() -> str:
    return _wiring_blocked_reason


def register_elastic_model(model_dir: str) -> None:
    """Record that `model_dir`'s weights are mapped read-only in-process."""
    with _lock:
        _elastic_model_dirs.add(os.path.realpath(model_dir))


def is_elastic_model_dir(model_dir: str) -> bool:
    return os.path.realpath(model_dir) in _elastic_model_dirs


def elastic_model_dirs() -> frozenset[str]:
    return frozenset(_elastic_model_dirs)


def assert_mutable(model_dir: str, operation: str) -> None:
    """Refuse weight-mutating operations on elastic-loaded models.

    Raises:
        RuntimeError: if `model_dir` was loaded through the elastic path.
    """
    if is_elastic_model_dir(model_dir):
        raise RuntimeError(
            f"{operation}: model at {model_dir!r} was loaded with "
            "elastic_load=true; its weights are read-only PROT_READ file "
            "mappings. Reload the model with elastic_load off (materialized "
            "private copy) before mutating weights."
        )
