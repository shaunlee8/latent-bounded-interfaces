"""Layout invariance of the registered Mamba-3 SISO backward. The mixer feeds
`mamba3_siso_combined` transposed, expanded, and sliced views, and the backward
Triton kernels assume packed layouts on the tensors saved for backward, which
`_Mamba3Function.forward` makes contiguous. Feeding views and feeding
contiguous copies of the same values must give the same gradients."""

from __future__ import annotations

import pytest
import torch

from tests.helpers import cos_rel

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@requires_cuda
def test_siso_combined_backward_is_layout_invariant() -> None:
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    bsz, seqlen, heads, hd, n_state = 2, 128, 4, 64, 64
    chunk = 64

    # Leaves in pre-view layouts mirroring the mixer graph: wide-buffer slices,
    # transposes, and a head-expand.
    q_base = torch.randn(bsz, seqlen, 1, n_state + 8, device=dev, dtype=dt, requires_grad=True)
    k_base = torch.randn(bsz, seqlen, 1, n_state + 8, device=dev, dtype=dt, requires_grad=True)
    # V/Z replicate the in_proj packing: a wider-buffer slice whose seq stride
    # W != H*hd is the proven corruption trigger.
    vz_width = 2 * heads * hd + 12
    vz_base = torch.randn(bsz, seqlen, vz_width, device=dev, dtype=dt, requires_grad=True)
    adt_base = -torch.rand(bsz, seqlen, heads, device=dev, dtype=torch.float32, requires_grad=True)
    dt_base = torch.rand(bsz, seqlen, heads, device=dev, dtype=torch.float32, requires_grad=True)
    trap_base = torch.randn(bsz, seqlen, heads, device=dev, dtype=dt, requires_grad=True)
    ang_base = torch.randn(bsz, seqlen, 1, n_state // 2, device=dev, dtype=dt, requires_grad=True)
    q_bias = torch.randn(heads, n_state, device=dev, dtype=torch.float32)
    k_bias = torch.randn(heads, n_state, device=dev, dtype=torch.float32)
    d_skip = torch.randn(heads, device=dev, dtype=torch.float32)
    leaves = [q_base, k_base, vz_base, adt_base, dt_base, trap_base, ang_base]

    def run(make_contiguous: bool):
        def m(x):
            return x.contiguous() if make_contiguous else x

        v_view = vz_base[..., : heads * hd].reshape(bsz, seqlen, heads, hd)
        z_view = vz_base[..., heads * hd : 2 * heads * hd].reshape(bsz, seqlen, heads, hd)
        assert not v_view.is_contiguous(), "test setup: V must be a strided view"
        out = mamba3_siso_combined(
            Q=m(q_base[..., :n_state]),
            K=m(k_base[..., :n_state]),
            V=m(v_view),
            ADT=m(adt_base.transpose(1, 2)),
            DT=m(dt_base.transpose(1, 2)),
            Trap=m(trap_base.transpose(1, 2)),
            Q_bias=q_bias,
            K_bias=k_bias,
            Angles=m(ang_base.expand(-1, -1, heads, -1)),
            D=d_skip,
            Z=m(z_view),
            chunk_size=chunk,
        )
        torch.manual_seed(7)
        g_out = torch.randn_like(out)
        return torch.autograd.grad(out, leaves, grad_outputs=g_out)

    grads_views = run(False)
    grads_cont = run(True)
    names = ("Q", "K", "VZ", "ADT", "DT", "Trap", "Angles")
    for name, gv, gc in zip(names, grads_views, grads_cont):
        cos, rel = cos_rel(gv, gc)
        assert cos > 0.9999 and rel < 1e-2, (
            f"{name}: view-fed backward diverges from contiguous-fed "
            f"(cos {cos:.4f}, rel {rel:.3e}): stride regression in the saved-tensor path"
        )
