from __future__ import annotations

import importlib.util

import pytest
import torch

import cuda.interface as interface_scan
from cuda.interface import suffix_scan_pullbacks


def _manual_suffix(mats: torch.Tensor) -> torch.Tensor:
    bsz, n_regions, rank, _ = mats.shape
    out = torch.empty((bsz, n_regions + 1, rank, rank), device=mats.device, dtype=mats.dtype)
    eye = torch.eye(rank, device=mats.device, dtype=mats.dtype).view(1, rank, rank).expand(bsz, rank, rank)
    out[:, n_regions] = eye
    running = eye.clone()
    for ridx in reversed(range(n_regions)):
        running = mats[:, ridx] @ running
        out[:, ridx] = running
    return out


def _cuda_extension_available() -> bool:
    return torch.cuda.is_available() and importlib.util.find_spec("interface_scan_cuda") is not None


def _random_mats(*, bsz: int, n_regions: int, rank: int, device: str | torch.device, dtype: torch.dtype) -> torch.Tensor:
    scale = 1.0 / max(rank, 1) ** 0.5
    return torch.randn(bsz, n_regions, rank, rank, device=device, dtype=dtype) * scale


def _float32_tolerances(n_regions: int, rank: int) -> tuple[float, float]:
    if rank >= 256:
        return 7e-1, 1e-5
    if rank >= 128:
        return 2.5e-1, 1e-5
    if n_regions >= 8 or rank >= 16:
        return 3e-2, 1e-5
    return 2e-5, 1e-5


@pytest.mark.parametrize(
    "shape",
    [
        (1, 0, 1),
        (1, 1, 1),
        (1, 1, 4),
        (2, 3, 5),
        (2, 4, 8),
        (4, 8, 16),
        (1, 16, 64),
    ],
)
def test_suffix_scan_pullbacks_cpu_fallback_matches_manual(shape: tuple[int, int, int]) -> None:
    bsz, n_regions, rank = shape
    torch.manual_seed(0)
    mats = _random_mats(bsz=bsz, n_regions=n_regions, rank=rank, device="cpu", dtype=torch.float32)
    original = mats.clone()

    saved_extension = interface_scan.interface_scan_cuda
    interface_scan.interface_scan_cuda = None
    try:
        actual = suffix_scan_pullbacks(mats)
    finally:
        interface_scan.interface_scan_cuda = saved_extension

    expected = _manual_suffix(mats)
    assert actual.shape == expected.shape
    assert actual.dtype == mats.dtype
    atol, rtol = _float32_tolerances(n_regions, rank)
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol)
    assert torch.equal(mats, original)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 16),
        (2, 4, 32),
        (2, 8, 64),
        (1, 16, 128),
        (1, 8, 256),
    ],
)
def test_suffix_scan_pullbacks_cuda_matches_manual_if_extension_available(shape: tuple[int, int, int]) -> None:
    if not _cuda_extension_available():
        pytest.skip("CUDA extension unavailable")
    bsz, n_regions, rank = shape
    torch.manual_seed(0)
    mats = _random_mats(bsz=bsz, n_regions=n_regions, rank=rank, device="cuda", dtype=torch.float32)
    original = mats.clone()
    actual = suffix_scan_pullbacks(mats)
    expected = _manual_suffix(mats).to(dtype=actual.dtype)
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    atol, rtol = _float32_tolerances(n_regions, rank)
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol)
    assert torch.equal(mats, original)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_suffix_scan_pullbacks_cuda_promotes_low_precision_inputs(dtype: torch.dtype) -> None:
    if not _cuda_extension_available():
        pytest.skip("CUDA extension unavailable")
    torch.manual_seed(0)
    mats = _random_mats(bsz=2, n_regions=8, rank=64, device="cuda", dtype=torch.float32).to(dtype)
    original = mats.clone()
    actual = suffix_scan_pullbacks(mats)
    expected = _manual_suffix(mats.to(torch.float32))
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=2e-1, rtol=5e-3)
    assert torch.equal(mats, original)


def test_suffix_scan_pullbacks_structured_order_case_cpu() -> None:
    mats = torch.tensor(
        [
            [
                [[1.0, 2.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 3.0]],
                [[1.0, 0.0], [4.0, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    saved_extension = interface_scan.interface_scan_cuda
    interface_scan.interface_scan_cuda = None
    try:
        actual = suffix_scan_pullbacks(mats)
    finally:
        interface_scan.interface_scan_cuda = saved_extension
    expected = _manual_suffix(mats)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual[:, -1], torch.eye(2).view(1, 2, 2))


def test_suffix_scan_pullbacks_structured_order_case_cuda_if_extension_available() -> None:
    if not _cuda_extension_available():
        pytest.skip("CUDA extension unavailable")
    mats = torch.tensor(
        [
            [
                [[1.0, 2.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 3.0]],
                [[1.0, 0.0], [4.0, 1.0]],
            ]
        ],
        device="cuda",
        dtype=torch.float32,
    )
    original = mats.clone()
    actual = suffix_scan_pullbacks(mats)
    expected = _manual_suffix(mats)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(mats, original)
