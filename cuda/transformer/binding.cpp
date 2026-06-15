#include <torch/extension.h>

#include "attention_pullback.hpp"
#include "attention_pullback_mma.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "attention_input_pullback_basis_simple",
        &attention_input_pullback_basis_simple,
        "Scalar CUDA reference implementation for attention input pullback basis [B, P, H, T, Dh]"
    );
    m.def(
        "attention_input_pullback_basis_mma_p1",
        &attention_input_pullback_basis_mma_p1,
        "CUTLASS/CuTe attention input pullback for one cotangent basis [B, H, T, Dh] (CUDA)"
    );
    m.def(
        "attention_input_pullback_basis_mma_pblock4",
        &attention_input_pullback_basis_mma_pblock4,
        "CUTLASS/CuTe attention input pullback for up to four cotangent bases [B, P, H, T, Dh] (CUDA)"
    );
}
