"""Checks for the forward-mode interface JVP methods and the A_k assembly.
The decode and update JVPs must be exact adjoints of the transpose methods (an
inner-product identity at machine precision), and the assembled
A_k = skip + encode . J_region . decode must match autograd on a
self-contained torch region. The region JVP here is torch.func.jvp through a
torch region, the reference for the kernel providers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from interfaces.vector_mlp import VectorMLPInterface
from backward.pullbacks import interface_state_jacobian_for_region_forward


def _build(hidden_dim: int, *, B=3, L=7, D=12, r=5, seed=0):
    torch.manual_seed(seed)
    interface = VectorMLPInterface(
        feature_dim=D, num_regions=2, interface_width=r,
        interface_map_hidden_dim=hidden_dim,
    ).double()
    # Perturb params off their init so Jacobians are generic (norm weights != 1).
    with torch.no_grad():
        for p in interface.parameters():
            p.add_(0.1 * torch.randn_like(p))
    k = 1
    u_base = torch.randn(B, L, D, dtype=torch.float64)
    g = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, D)).double()
    with torch.no_grad():
        for p in g.parameters():
            p.mul_(0.7)
    s0 = torch.randn(B, r, dtype=torch.float64)

    cond = interface.decode(s0, k)
    u0 = u_base + cond.unsqueeze(1)
    y0 = g(u0)
    step = interface.update(s0, y0, k)
    region_cache = SimpleNamespace(state_in=s0, region_output=y0, interface_step=step, region_index=k)
    model = SimpleNamespace(interface=interface)
    return interface, model, region_cache, g, u_base, s0, k, (B, L, D, r)


@pytest.mark.parametrize("hidden_dim", [0, 16])
def test_decode_jvp_is_adjoint_of_decode_t(hidden_dim: int) -> None:
    interface, _, region_cache, *_ , dims = _build(hidden_dim)
    B, L, D, r = dims
    v = torch.randn(B, 1, r, dtype=torch.float64)          # state tangent
    w = torch.randn(B, 1, L, D, dtype=torch.float64)       # region-input cotangent
    jvp = interface.apply_decode_jacobian_to_state_input(
        region_cache=region_cache, state_input_tangent_basis=v)["d_region_input"]
    vjp = interface.apply_decode_jacobian_t_to_state_input(
        region_cache=region_cache, region_input_cotangent_basis=w)["g_state_input"]
    lhs = (jvp * w).sum().item()
    rhs = (v * vjp).sum().item()
    assert abs(lhs - rhs) <= 1e-9 * (abs(lhs) + abs(rhs) + 1e-12), f"decode adjoint mismatch {lhs} vs {rhs}"


@pytest.mark.parametrize("hidden_dim", [0, 16])
def test_update_jvp_is_adjoint_of_update_t(hidden_dim: int) -> None:
    interface, _, region_cache, *_, dims = _build(hidden_dim)
    B, L, D, r = dims
    t = torch.randn(B, 1, L, D, dtype=torch.float64)       # region-output tangent
    v = torch.randn(B, 1, r, dtype=torch.float64)          # skip state tangent
    g = torch.randn(B, 1, r, dtype=torch.float64)          # state-out cotangent
    jvp_out = interface.apply_update_jacobian_to_region_output(
        region_cache=region_cache, region_output_tangent_basis=t, state_input_tangent_basis=v)["d_state_out"]
    rev = interface.apply_update_jacobian_t_to_region_output(
        region_cache=region_cache, state_out_cotangent_basis=g)
    lhs = (jvp_out * g).sum().item()
    rhs = (t * rev["g_region_output"]).sum().item() + (v * rev["g_state_skip"]).sum().item()
    assert abs(lhs - rhs) <= 1e-9 * (abs(lhs) + abs(rhs) + 1e-12), f"update adjoint mismatch {lhs} vs {rhs}"


@pytest.mark.parametrize("hidden_dim", [0, 16])
def test_forward_mode_Ak_matches_autograd(hidden_dim: int) -> None:
    interface, model, region_cache, g, u_base, s0, k, dims = _build(hidden_dim)
    B, L, D, r = dims

    def region_output_jvp(basis: torch.Tensor) -> torch.Tensor:  # [B,P,L,D] -> [B,P,L,D]
        u0 = u_base + interface.decode(s0, k).unsqueeze(1)
        outs = []
        for p in range(basis.shape[1]):
            _, d = torch.func.jvp(g, (u0,), (basis[:, p],))
            outs.append(d)
        return torch.stack(outs, dim=1)

    # Forward-assembled A_k^T [B, R_in, R_out].
    Ak_fwd = interface_state_jacobian_for_region_forward(
        model=model, region_cache=region_cache, region_output_jvp=region_output_jvp)

    # Autograd ground truth of the true state-out map.
    def state_out(s: torch.Tensor) -> torch.Tensor:
        cond = interface.decode(s, k)
        u = u_base + cond.unsqueeze(1)
        y = g(u)
        return interface.update(s, y, k).state

    jac = torch.autograd.functional.jacobian(lambda s: state_out(s).sum(0), s0)  # [R_out, B, R_in]
    Ak_true_t = jac.permute(1, 2, 0).contiguous()  # [B, R_in, R_out] = A_k^T

    rel = ((Ak_fwd - Ak_true_t).abs().max() / (Ak_true_t.abs().max() + 1e-30)).item()
    assert rel <= 1e-9, f"forward-mode A_k disagrees with autograd (rel {rel:.2e})"
