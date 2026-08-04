"""Cache-native reverse VJP for the transformer region backend.

Walks the pre-norm residual blocks backward with a single cotangent stream
(residual threading makes the hidden and residual cotangents equal): param
grads are GEMMs against cached activations, the attention backward runs the
flash backward op on cached softmax stats, and the rope and causal-conv
transposes are closed form. The region forward is not replayed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from backbones.transformer.rope import _rope_tables
from backends.transformer import softmax_scale


def _rmsnorm_vjp(x, g_y, weight, eps):
    """y = (x_f*s)*w with s = rsqrt(mean x^2 + eps). The input Jacobian is
    symmetric, so the pullback reuses the tangent formula on u = g_y*w."""
    xf = x.float()
    s = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    u = g_y.float() * weight.float()
    g_x = u * s - xf * s.pow(3) * (xf * u).mean(dim=-1, keepdim=True)
    g_w = (g_y.float() * (xf * s)).sum(dim=(0, 1))
    return g_x, g_w.to(weight.dtype)


def _rope_transpose(x, base):
    """The rotation's transpose is the rotation by -theta (half-split)."""
    hd = x.shape[-1]
    half = hd // 2
    cos, sin = _rope_tables(x.shape[-2], half, base, x.device)
    xe = x[..., :half].float()
    xo = x[..., half:].float()
    ge = xe * cos + xo * sin
    go = -xe * sin + xo * cos
    return torch.cat((ge, go), dim=-1).to(x.dtype)


def _linear_wgrad(g, x):
    return g.reshape(-1, g.shape[-1]).transpose(0, 1) @ x.reshape(-1, x.shape[-1])


def _causal_conv_vjp(g_y, x_pre, weight):
    """y[l] = sum_k w[:,k]*x[l-(K-1)+k] (zero pad below); the input pullback
    is the anticausal correlation, the weight grad reads the cached pre-conv
    projection."""
    B, L, C = g_y.shape
    K = weight.shape[-1]
    w = weight.view(C, K).float()
    gp = F.pad(g_y.float(), (0, 0, 0, K - 1))
    g_x = None
    for k in range(K):
        sh = (K - 1) - k
        term = w[:, k] * gp[:, sh:sh + L]
        g_x = term if g_x is None else g_x + term
    xp = F.pad(x_pre.float(), (0, 0, K - 1, 0))
    g_w = torch.stack(
        [(g_y.float() * xp[:, k:k + L]).sum(dim=(0, 1)) for k in range(K)], dim=1)
    return g_x.to(g_y.dtype), g_w.to(weight.dtype)


def native_vjp_supported(backend, cache):
    for lc in cache.layer_caches:
        attn = backend.backbone.blocks[lc.layer_index].attn
        if attn.rope_interleaved:
            return False
        if attn.d_conv > 0 and lc.attention.qkv_preconv is None:
            return False
    return True


def transformer_region_parameter_vjp(backend, cache, output_cotangent):
    """Returns (named param grads, region-input cotangent [B, L, D])."""
    names = {id(p): n for n, p in backend.named_parameters()}
    grads: dict[str, torch.Tensor] = {}

    def put(param, g):
        n = names[id(param)]
        grads[n] = g.to(param.dtype) if n not in grads else grads[n] + g.to(param.dtype)

    g = output_cotangent.float()
    for lc in reversed(cache.layer_caches):
        blk = backend.backbone.blocks[lc.layer_index]
        attn, mlp = blk.attn, blk.mlp
        ac, mc = lc.attention, lc.mlp

        # ---- MLP: fc2 <- gate <- fc1 <- norm2 ----
        g16 = g.to(mc.down_input.dtype)
        put(mlp.fc2.weight, _linear_wgrad(g16, mc.down_input))
        if mlp.fc2.bias is not None:
            put(mlp.fc2.bias, g16.sum(dim=(0, 1)))
        g_di = g16 @ mlp.fc2.weight
        gate_f = mc.gate.float()
        sg = torch.sigmoid(gate_f)
        g_up = g_di * mc.activation.to(g_di.dtype)
        g_gate = (g_di.float() * mc.up.float()
                  * (sg * (1 + gate_f * (1 - sg)))).to(g_di.dtype)
        g_fc1 = torch.cat((g_up, g_gate), dim=-1)
        put(mlp.fc1.weight, _linear_wgrad(g_fc1, mc.norm_output))
        if mlp.fc1.bias is not None:
            put(mlp.fc1.bias, g_fc1.sum(dim=(0, 1)))
        g_nh2 = g_fc1 @ mlp.fc1.weight
        g_x, g_w = _rmsnorm_vjp(mc.norm_input, g_nh2, blk.norm2.weight, blk.norm2.eps)
        put(blk.norm2.weight, g_w)
        g = g + g_x

        # ---- Attention: out_proj <- flash island <- rope^T <- conv^T <- in_proj <- norm ----
        g16 = g.to(ac.out_proj_input.dtype)
        put(attn.out_proj.weight, _linear_wgrad(g16, ac.out_proj_input))
        if attn.out_proj.bias is not None:
            put(attn.out_proj.bias, g16.sum(dim=(0, 1)))
        g_o = g16 @ attn.out_proj.weight
        B, L, _ = g_o.shape
        H, hd = attn.n_heads, attn.head_dim
        g_heads = g_o.view(B, L, H, hd).transpose(1, 2)
        scale = softmax_scale(attn)
        if ac.attn_logsumexp is not None:
            # Cached softmax stats: the flash backward op runs directly.
            qc = ac.q_rope.contiguous()
            kc = ac.k_expanded.contiguous()
            vc = ac.v_expanded.contiguous()
            gq, gk, gv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                g_heads.to(qc.dtype).contiguous(), qc, kc, vc,
                ac.attn_out_bhld, ac.attn_logsumexp, None, None, L, L,
                0.0, True, ac.attn_philox_seed, ac.attn_philox_offset,
                scale=scale)
        else:
            q = ac.q_rope.detach().requires_grad_(True)
            k = ac.k_expanded.detach().requires_grad_(True)
            v = ac.v_expanded.detach().requires_grad_(True)
            with torch.enable_grad():
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
                gq, gk, gv = torch.autograd.grad(
                    out, (q, k, v), grad_outputs=g_heads.to(out.dtype))
        rep = H // attn.n_kv_heads
        if rep > 1:
            gk = gk.view(B, attn.n_kv_heads, rep, L, hd).sum(dim=2)
            gv = gv.view(B, attn.n_kv_heads, rep, L, hd).sum(dim=2)
        gq = _rope_transpose(gq, attn.rope_base)
        gk = _rope_transpose(gk, attn.rope_base)
        g_qkv = torch.cat((
            gq.transpose(1, 2).reshape(B, L, H * hd),
            gk.transpose(1, 2).reshape(B, L, attn.n_kv_heads * hd),
            gv.transpose(1, 2).reshape(B, L, attn.n_kv_heads * hd),
        ), dim=-1)
        if attn.d_conv > 0:
            if attn.conv1d.bias is not None:
                put(attn.conv1d.bias, g_qkv.sum(dim=(0, 1)))
            g_qkv, g_cw = _causal_conv_vjp(g_qkv, ac.qkv_preconv, attn.conv1d.weight)
            put(attn.conv1d.weight, g_cw.view_as(attn.conv1d.weight))
        put(attn.in_proj.weight, _linear_wgrad(g_qkv, ac.norm_output))
        if attn.in_proj.bias is not None:
            put(attn.in_proj.bias, g_qkv.sum(dim=(0, 1)))
        g_nh1 = g_qkv @ attn.in_proj.weight
        g_x, g_w = _rmsnorm_vjp(ac.norm_input, g_nh1, blk.norm.weight, blk.norm.eps)
        put(blk.norm.weight, g_w)
        g = g + g_x

    return grads, g.to(cache.region_input.dtype)
