#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> attention_input_pullback_basis_simple(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output_cotangent_basis,
    double softmax_scale,
    bool causal
);
