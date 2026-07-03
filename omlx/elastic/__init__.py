# SPDX-License-Identifier: Apache-2.0
"""Elastic (mmap) weight loading for oMLX.

File-backed, OS-reclaimable model weights: safetensors shards are mmap'd
PROT_READ and wrapped zero-copy into Metal buffers, so the OS can evict
weight pages under memory pressure and refault them from disk instead of
swapping. Enables >RAM-budget models and graceful co-tenancy.

Opt-in per model via the ``elastic_load`` model setting (default OFF).

Lazy package: importing ``omlx.elastic`` (or ``omlx.elastic.state``, which
process_memory_enforcer does on every server boot) must never pull in the
loader/realign machinery or the native extension. Loader symbols resolve
on first use via PEP 562 ``__getattr__``, so the default (non-elastic)
path stays import-light and cannot be broken by a missing extension.

See omlx/elastic/README.md for the mechanism, constraints (read-only
weights, no Metal wiring), and build instructions.
"""

from . import state as state  # stdlib-only, safe to import eagerly
from .state import (  # noqa: F401
    assert_mutable,
    elastic_model_dirs,
    is_elastic_model_dir,
    wiring_blocked,
    wiring_blocked_reason,
)

_LOADER_ATTRS = {
    "ElasticUnavailableError",
    "elastic_load_scope",
    "is_available",
    "load_shard_elastic",
}
_REALIGN_ATTRS = {"ensure_realigned", "shard_alignment"}

__all__ = sorted(
    {
        "state",
        "assert_mutable",
        "elastic_model_dirs",
        "is_elastic_model_dir",
        "wiring_blocked",
        "wiring_blocked_reason",
    }
    | _LOADER_ATTRS
    | _REALIGN_ATTRS
)


def __getattr__(name: str):
    if name in _LOADER_ATTRS:
        from . import loader

        return getattr(loader, name)
    if name in _REALIGN_ATTRS:
        from . import realign

        return getattr(realign, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
