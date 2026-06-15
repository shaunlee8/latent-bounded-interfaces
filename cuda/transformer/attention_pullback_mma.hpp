#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> attention_input_pullback_basis_mma_p1(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor logsumexp,
    torch::Tensor output_cotangent,
    double softmax_scale,
    bool causal
);
std::vector<torch::Tensor> attention_input_pullback_basis_mma_pblock4(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor logsumexp,
    torch::Tensor output_cotangent_basis,
    double softmax_scale,
    bool causal
);
