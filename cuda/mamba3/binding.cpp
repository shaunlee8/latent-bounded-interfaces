// Torch binding for the lane-batched wgmma recurrence kernel of the Mamba-3
// forward-mode construction (recurrence_jvp.cu).
#include <torch/extension.h>

std::vector<torch::Tensor> recurrence_full(
    torch::Tensor QR, torch::Tensor KR, torch::Tensor V,
    torch::Tensor DQRAW, torch::Tensor DKRAW, torch::Tensor DV,
    torch::Tensor DTHETA, torch::Tensor COS, torch::Tensor SIN,
    torch::Tensor SCALE, torch::Tensor DSCALE,
    torch::Tensor L, torch::Tensor DL);
std::vector<torch::Tensor> recurrence_full_fin(
    torch::Tensor QR, torch::Tensor KR, torch::Tensor V,
    torch::Tensor DQRAW, torch::Tensor DKRAW, torch::Tensor DV,
    torch::Tensor DTHETA, torch::Tensor COS, torch::Tensor SIN,
    torch::Tensor SCALE, torch::Tensor DSCALE,
    torch::Tensor L, torch::Tensor DL,
    torch::Tensor Z, torch::Tensor DZ,
    torch::Tensor QKD, torch::Tensor DQKD, torch::Tensor DSK);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("recurrence_full", &recurrence_full,
          "Lane-batched wgmma recurrence kernel, raw OUT/DOUT fields",
          pybind11::arg("QR"), pybind11::arg("KR"), pybind11::arg("V"),
          pybind11::arg("DQRAW"), pybind11::arg("DKRAW"), pybind11::arg("DV"),
          pybind11::arg("DTHETA"), pybind11::arg("COS"), pybind11::arg("SIN"),
          pybind11::arg("SCALE"), pybind11::arg("DSCALE"), pybind11::arg("L"),
          pybind11::arg("DL"));
    m.def("recurrence_full_fin", &recurrence_full_fin,
          "Lane-batched wgmma recurrence kernel, finalize folded into the epilogue -> finalized OUT/DOUT",
          pybind11::arg("QR"), pybind11::arg("KR"), pybind11::arg("V"),
          pybind11::arg("DQRAW"), pybind11::arg("DKRAW"), pybind11::arg("DV"),
          pybind11::arg("DTHETA"), pybind11::arg("COS"), pybind11::arg("SIN"),
          pybind11::arg("SCALE"), pybind11::arg("DSCALE"), pybind11::arg("L"),
          pybind11::arg("DL"), pybind11::arg("Z"), pybind11::arg("DZ"),
          pybind11::arg("QKD"), pybind11::arg("DQKD"), pybind11::arg("DSK"));
}
