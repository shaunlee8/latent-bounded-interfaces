"""Hand-CUDA kernels for the Mamba-3 backward and forward-mode walks.

`chunkparallel_pass_c_simple` is the correctness scaffold, parity-gated
against the tilelang pass C; `_mma` is its tensor-core rewrite. Optional
import: the tilelang path is the fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
try:
    import mamba3_lbi_cuda  # type: ignore
except Exception:  # pragma: no cover - optional extension
    mamba3_lbi_cuda = None  # type: ignore[assignment]


def has_cuda_pass_c() -> bool:
    return mamba3_lbi_cuda is not None


def chunkparallel_pass_c(*args, use_mma: bool = False) -> None:
    """Same argument order as the tilelang `mamba_chunkparallel_bwd_bwd` call.
    `use_mma=True` runs the tensor-core (wmma) kernel; False the scalar scaffold."""
    if mamba3_lbi_cuda is None:
        raise RuntimeError("mamba3_lbi_cuda extension not built (run cuda/mamba3/build.sh)")
    mamba3_lbi_cuda.chunkparallel_pass_c(*args, use_mma=use_mma)
