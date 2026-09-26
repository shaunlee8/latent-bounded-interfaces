"""Emulated slow interconnect at the torch.distributed call boundary. Each
operation is charged delay = RTT + outbound_bytes / bandwidth, with outbound
bytes as comm_bytes counts them and all_reduce charged its ring wire traffic
2(K-1)/K x payload. The two charging modes bracket a real link.

- LBI_EMU_MODE=sync (default): every operation is charged at its issue site
  by a host-side sleep owed by the sender, so no transfer overlaps compute.
- LBI_EMU_MODE=overlap: asynchronous operations are charged at their wait()
  and point-to-point transfers once on the receiving side, so transfers
  overlap the caller's compute.
- LBI_EMU_BW_MBPS (megabits/s; unset or 0 = off) and LBI_EMU_RTT_MS set the
  link; LBI_EMU_P2P_SCALE scales point-to-point bytes only (activation
  compression on the pipeline's boundary payloads); LBI_EMU_LOG=1 prints
  per-rank counts and bytes of the charged operations at exit.

comm_bytes.install() installs this module on top of its counters when
LBI_EMU_BW_MBPS is set, so only the timed region of a benchmark is charged."""

from __future__ import annotations

import atexit
import os
import time

import torch.distributed as dist

MBPS_TO_BYTES_PER_S = 125_000.0

_ORIG: dict[str, object] = {}
_STATS: dict[str, list[float]] = {}  # op kind -> [count, bytes]
_RANK = [-1]


def _cfg():
    bw = float(os.environ.get("LBI_EMU_BW_MBPS", "0") or 0)
    rtt = float(os.environ.get("LBI_EMU_RTT_MS", "0") or 0) / 1e3
    return bw * MBPS_TO_BYTES_PER_S, rtt


def _mode() -> str:
    return os.environ.get("LBI_EMU_MODE", "sync")


def _p2p_scale() -> float:
    return float(os.environ.get("LBI_EMU_P2P_SCALE", "1") or 1)


def _delay(nbytes: float) -> float:
    bps, rtt = _cfg()
    if bps <= 0:
        return 0.0
    return rtt + nbytes / bps


def _charge(nbytes: float) -> None:
    time.sleep(_delay(nbytes))


def _nbytes(tensor) -> int:
    return tensor.element_size() * tensor.numel()


def _note(kind: str, nbytes: float) -> None:
    s = _STATS.setdefault(kind, [0, 0.0])
    s[0] += 1
    s[1] += nbytes
    if _RANK[0] < 0 and dist.is_initialized():
        _RANK[0] = dist.get_rank()


def reset_stats() -> None:
    _STATS.clear()


def _print_stats() -> None:
    if os.environ.get("LBI_EMU_LOG", "0") != "1" or not _STATS:
        return
    parts = [f"{k}:n={int(v[0])},bytes={v[1]:.0f}" for k, v in sorted(_STATS.items())]
    print(f"emu-ops rank={_RANK[0]} mode={_mode()} " + " ".join(parts), flush=True)


class _DeferredWork:
    """Work handle whose wait() blocks until the emulated transfer would
    have completed; everything else delegates to the real handle."""

    def __init__(self, work, deadline: float):
        self._work = work
        self._deadline = deadline

    def wait(self, *args, **kwargs):
        rem = self._deadline - time.perf_counter()
        if rem > 0:
            time.sleep(rem)
        return self._work.wait(*args, **kwargs) if self._work is not None else True

    def is_completed(self):
        if time.perf_counter() < self._deadline:
            return False
        return self._work.is_completed() if self._work is not None else True

    def __getattr__(self, name):
        return getattr(self._work, name)


def _deferred(work, nbytes: float):
    return _DeferredWork(work, time.perf_counter() + _delay(nbytes))


def install() -> None:
    """Patch dist.* once, wrapping whatever is currently installed
    (i.e. on top of comm_bytes' counters); idempotent."""
    if _ORIG:
        return

    _ORIG["send"] = dist.send
    _ORIG["all_reduce"] = dist.all_reduce
    _ORIG["broadcast"] = dist.broadcast
    _ORIG["batch_isend_irecv"] = dist.batch_isend_irecv
    atexit.register(_print_stats)

    def send(tensor, *args, **kwargs):
        nb = _nbytes(tensor) * _p2p_scale()
        _note("send", nb)
        _charge(nb)  # a blocking send is owed its transfer in either mode
        return _ORIG["send"](tensor, *args, **kwargs)

    def all_reduce(tensor, *args, **kwargs):
        k = dist.get_world_size()
        nb = _nbytes(tensor) * 2 * (k - 1) / k
        _note("all_reduce", nb)
        if _mode() == "overlap" and kwargs.get("async_op", False):
            return _deferred(_ORIG["all_reduce"](tensor, *args, **kwargs), nb)
        _charge(nb)
        return _ORIG["all_reduce"](tensor, *args, **kwargs)

    def broadcast(tensor, src=None, *args, **kwargs):
        nb = _nbytes(tensor) if (src is not None and src == dist.get_rank()) else 0
        _note("broadcast", nb)
        if _mode() == "overlap" and kwargs.get("async_op", False):
            return _deferred(_ORIG["broadcast"](tensor, src, *args, **kwargs), nb)
        if nb > 0:
            _charge(nb)  # receivers owe nothing: the source's sleep delays arrival
        return _ORIG["broadcast"](tensor, src, *args, **kwargs)

    def batch_isend_irecv(p2p_op_list):
        out = 0.0
        arrivals = []
        try:
            for op in p2p_op_list:
                nb = _nbytes(op.tensor) * _p2p_scale()
                if op.op is dist.isend:
                    out += nb
                else:
                    arrivals.append(nb)
        except AttributeError:
            pass  # emulation must never break the underlying call
        _note("p2p_send", out)
        _note("p2p_recv", sum(arrivals))
        if _mode() == "overlap":
            works = _ORIG["batch_isend_irecv"](p2p_op_list)
            # the transfer is paid once, on the receiving side (arrival =
            # issue + delay from the receiver's clock); a send handle
            # completes at once, as if its buffer were copied out. Charging
            # both sides serialized the same interval twice (torch's
            # pipeline schedules wait on every send batch immediately).
            nb_iter = iter(arrivals)
            deferred = []
            for w, op in zip(works, p2p_op_list):
                nb = 0.0 if op.op is dist.isend else next(nb_iter, 0.0)
                deferred.append(_deferred(w, nb) if nb > 0 else w)
            return deferred
        if out > 0:
            _charge(out)  # receive-only batches owe nothing (see broadcast)
        return _ORIG["batch_isend_irecv"](p2p_op_list)

    dist.send = send
    dist.all_reduce = all_reduce
    dist.broadcast = broadcast
    dist.batch_isend_irecv = batch_isend_irecv
