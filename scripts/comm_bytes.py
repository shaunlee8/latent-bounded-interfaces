"""Per-rank outbound byte counting at the torch.distributed call boundary.
dist.send, dist.broadcast, dist.all_reduce, and dist.batch_isend_irecv are
wrapped to accumulate the payload bytes each rank sends: point-to-point at the
sender, broadcast at the source rank, all_reduce as each rank's contribution
(logical payload, not NCCL wire traffic). isend/irecv stay unpatched so
torch's pipeline schedules keep their P2POp identity checks; batched sends are
counted inside batch_isend_irecv."""

from __future__ import annotations

import os
import sys

import torch.distributed as dist

_TOTAL = 0
_ORIG: dict[str, object] = {}


def _add(tensor) -> None:
    global _TOTAL
    _TOTAL += tensor.element_size() * tensor.numel()


def _wrap_tensor_first(name):
    orig = _ORIG[name]

    def wrapped(tensor, *args, **kwargs):
        _add(tensor)
        return orig(tensor, *args, **kwargs)

    return wrapped


def _wrap_broadcast():
    orig = _ORIG["broadcast"]

    def wrapped(tensor, src=None, *args, **kwargs):
        if src is not None and src == dist.get_rank():
            _add(tensor)
        return orig(tensor, src, *args, **kwargs)

    return wrapped


def _wrap_batch_isend_irecv():
    orig = _ORIG["batch_isend_irecv"]
    isend = dist.isend

    def wrapped(p2p_op_list):
        try:
            for op in p2p_op_list:
                if op.op is isend:
                    _add(op.tensor)
        except AttributeError:
            pass  # counting must never break the underlying call
        return orig(p2p_op_list)

    return wrapped


def install() -> None:
    """Patch dist.* once; idempotent."""
    if _ORIG:
        return
    for name in ("send", "all_reduce"):
        _ORIG[name] = getattr(dist, name)
        setattr(dist, name, _wrap_tensor_first(name))
    _ORIG["broadcast"] = dist.broadcast
    dist.broadcast = _wrap_broadcast()
    _ORIG["batch_isend_irecv"] = dist.batch_isend_irecv
    dist.batch_isend_irecv = _wrap_batch_isend_irecv()

    if float(os.environ.get("LBI_EMU_BW_MBPS", "0") or 0) > 0:
        from scripts import link_emu  # wraps the counters just installed
        link_emu.install()


def reset() -> None:
    global _TOTAL
    _TOTAL = 0
    emu = sys.modules.get("scripts.link_emu")
    if emu is not None:
        emu.reset_stats()


def total_bytes() -> int:
    return _TOTAL
