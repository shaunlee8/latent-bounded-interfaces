from __future__ import annotations

import torch

from cuda.transformer import attention_input_pullback_basis as cuda_attention_input_pullback_basis

from backends.transformer import (
    TorchAutogradTransformerLowering,
    TransformerRegionBackend,
    TransformerRegionCache,
)


class CudaTransformerLowering:
    """CUDA lowering for Transformer region derivative contracts."""

    name = "cuda"

    def __init__(self, *, allow_reference_fallback: bool = False) -> None:
        self.allow_reference_fallback = bool(allow_reference_fallback)
        self.reference_lowering = TorchAutogradTransformerLowering()

    def attention_input_pullback_basis(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output_cotangent_basis: torch.Tensor,
        softmax_scale: float,
        causal: bool = True,
        kernel_mode: str = "fa2_mma_pblock4",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return cuda_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
            use_cuda=True,
            kernel_mode=kernel_mode,
        )

    def input_pullback_basis(
        self,
        *,
        backend: TransformerRegionBackend,
        cache: TransformerRegionCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        if output_cotangent_basis.dim() != 4:
            raise ValueError("output_cotangent_basis must have shape [B, P, T, D]")
        if self.allow_reference_fallback:
            return self.reference_lowering.input_pullback_basis(
                backend=backend,
                cache=cache,
                output_cotangent_basis=output_cotangent_basis,
            )
        raise NotImplementedError(
            "CudaTransformerLowering.input_pullback_basis is the CUDA kernel target. "
            "Implement the region input pullback for output_cotangent_basis [B, P, T, D], "
            "or construct with allow_reference_fallback=True for reference execution."
        )

    def parameter_vjp(
        self,
        *,
        backend: TransformerRegionBackend,
        cache: TransformerRegionCache,
        output_cotangent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.allow_reference_fallback:
            return self.reference_lowering.parameter_vjp(
                backend=backend,
                cache=cache,
                output_cotangent=output_cotangent,
            )
        raise NotImplementedError(
            "CudaTransformerLowering.parameter_vjp is not implemented yet. "
            "Implement backend parameter VJPs, or construct with allow_reference_fallback=True "
            "while developing input_pullback_basis."
        )
