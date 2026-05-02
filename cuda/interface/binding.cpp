#include <torch/extension.h>

#include "suffix_scan.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "suffix_scan_mats",
        &suffix_scan_mats_launcher,
        "Suffix scan over [B, K, R, R] pullback matrices (CUDA)"
    );
}
