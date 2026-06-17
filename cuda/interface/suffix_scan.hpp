#pragma once

#include <torch/extension.h>

torch::Tensor suffix_scan_mats_launcher(torch::Tensor mats);
torch::Tensor suffix_scan_jacobian(torch::Tensor jacobian);
