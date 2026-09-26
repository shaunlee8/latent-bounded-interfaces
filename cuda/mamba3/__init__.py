"""The Mamba-3 recurrence kernel of the forward-mode construction, as an
optional torch extension; the tilelang recurrence is the fallback."""

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
