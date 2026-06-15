/*
Scalar CUDA reference implementation for the LBI-2 Transformer attention input
pullback. Tests use this path to compare the extension boundary against PyTorch
autograd; it is not a performance path.
*/
#include "attention_pullback.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 128;

__device__ inline long qkv_offset(int b, int h, int t, int d, int H, int T, int Dh) {
    return (((static_cast<long>(b) * H + h) * T + t) * Dh + d);
}

__device__ inline long basis_offset(int b, int p, int h, int t, int d, int P, int H, int T, int Dh) {
    return ((((static_cast<long>(b) * P + p) * H + h) * T + t) * Dh + d);
}

__device__ float simple_score_qk(
    const float* q,
    const float* k,
    int b,
    int h,
    int query,
    int key,
    int H,
    int T,
    int Dh,
    float scale
) {
    float acc = 0.0f;
    for (int d = 0; d < Dh; ++d) {
        acc += q[qkv_offset(b, h, query, d, H, T, Dh)] * k[qkv_offset(b, h, key, d, H, T, Dh)];
    }
    return acc * scale;
}

__device__ float simple_row_max_score(
    const float* q,
    const float* k,
    int b,
    int h,
    int query,
    int H,
    int T,
    int Dh,
    float scale,
    bool causal
) {
    const int key_end = causal ? query + 1 : T;
    float max_val = -3.4028234663852886e38f;
    for (int key = 0; key < key_end; ++key) {
        const float s = simple_score_qk(q, k, b, h, query, key, H, T, Dh, scale);
        max_val = fmaxf(max_val, s);
    }
    return max_val;
}

__device__ float simple_row_exp_sum(
    const float* q,
    const float* k,
    int b,
    int h,
    int query,
    int H,
    int T,
    int Dh,
    float scale,
    bool causal,
    float max_score
) {
    const int key_end = causal ? query + 1 : T;
    float denom = 0.0f;
    for (int key = 0; key < key_end; ++key) {
        denom += expf(simple_score_qk(q, k, b, h, query, key, H, T, Dh, scale) - max_score);
    }
    return denom;
}

__device__ float simple_dp_score(
    const float* v,
    const float* output_cotangent_basis,
    int b,
    int p,
    int h,
    int query,
    int key,
    int P,
    int H,
    int T,
    int Dh
) {
    float acc = 0.0f;
    for (int d = 0; d < Dh; ++d) {
        acc += output_cotangent_basis[basis_offset(b, p, h, query, d, P, H, T, Dh)] *
               v[qkv_offset(b, h, key, d, H, T, Dh)];
    }
    return acc;
}

__device__ float simple_row_softmax_dot_dp(
    const float* q,
    const float* k,
    const float* v,
    const float* output_cotangent_basis,
    int b,
    int p,
    int h,
    int query,
    int P,
    int H,
    int T,
    int Dh,
    float scale,
    bool causal,
    float max_score,
    float denom
) {
    const int key_end = causal ? query + 1 : T;
    float acc = 0.0f;
    for (int key = 0; key < key_end; ++key) {
        const float prob = expf(simple_score_qk(q, k, b, h, query, key, H, T, Dh, scale) - max_score) / denom;
        acc += prob * simple_dp_score(v, output_cotangent_basis, b, p, h, query, key, P, H, T, Dh);
    }
    return acc;
}

__global__ void simple_dq_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* output_cotangent_basis,
    float* dq,
    int B,
    int P,
    int H,
    int T,
    int Dh,
    float scale,
    bool causal,
    long total
) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % Dh;
    long tmp = idx / Dh;
    int query = tmp % T;
    tmp /= T;
    int h = tmp % H;
    tmp /= H;
    int p = tmp % P;
    int b = tmp / P;

    const int key_end = causal ? query + 1 : T;
    const float max_score = simple_row_max_score(q, k, b, h, query, H, T, Dh, scale, causal);
    const float denom = simple_row_exp_sum(q, k, b, h, query, H, T, Dh, scale, causal, max_score);
    const float row_dot = simple_row_softmax_dot_dp(q, k, v, output_cotangent_basis, b, p, h, query, P, H, T, Dh, scale, causal, max_score, denom);

    float acc = 0.0f;
    for (int key = 0; key < key_end; ++key) {
        const float prob = expf(simple_score_qk(q, k, b, h, query, key, H, T, Dh, scale) - max_score) / denom;
        const float dp = simple_dp_score(v, output_cotangent_basis, b, p, h, query, key, P, H, T, Dh);
        const float ds = prob * (dp - row_dot);
        acc += ds * k[qkv_offset(b, h, key, d, H, T, Dh)];
    }
    dq[basis_offset(b, p, h, query, d, P, H, T, Dh)] = acc * scale;
}

__global__ void simple_dk_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* output_cotangent_basis,
    float* dk,
    int B,
    int P,
    int H,
    int T,
    int Dh,
    float scale,
    bool causal,
    long total
) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % Dh;
    long tmp = idx / Dh;
    int key = tmp % T;
    tmp /= T;
    int h = tmp % H;
    tmp /= H;
    int p = tmp % P;
    int b = tmp / P;

    const int query_start = causal ? key : 0;
    float acc = 0.0f;
    for (int query = query_start; query < T; ++query) {
        const float max_score = simple_row_max_score(q, k, b, h, query, H, T, Dh, scale, causal);
        const float denom = simple_row_exp_sum(q, k, b, h, query, H, T, Dh, scale, causal, max_score);
        const float row_dot = simple_row_softmax_dot_dp(q, k, v, output_cotangent_basis, b, p, h, query, P, H, T, Dh, scale, causal, max_score, denom);
        const float prob = expf(simple_score_qk(q, k, b, h, query, key, H, T, Dh, scale) - max_score) / denom;
        const float dp = simple_dp_score(v, output_cotangent_basis, b, p, h, query, key, P, H, T, Dh);
        const float ds = prob * (dp - row_dot);
        acc += ds * q[qkv_offset(b, h, query, d, H, T, Dh)];
    }
    dk[basis_offset(b, p, h, key, d, P, H, T, Dh)] = acc * scale;
}

__global__ void simple_dv_kernel(
    const float* q,
    const float* k,
    const float* output_cotangent_basis,
    float* dv,
    int B,
    int P,
    int H,
    int T,
    int Dh,
    float scale,
    bool causal,
    long total
) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % Dh;
    long tmp = idx / Dh;
    int key = tmp % T;
    tmp /= T;
    int h = tmp % H;
    tmp /= H;
    int p = tmp % P;
    int b = tmp / P;

    const int query_start = causal ? key : 0;
    float acc = 0.0f;
    for (int query = query_start; query < T; ++query) {
        const float max_score = simple_row_max_score(q, k, b, h, query, H, T, Dh, scale, causal);
        const float denom = simple_row_exp_sum(q, k, b, h, query, H, T, Dh, scale, causal, max_score);
        const float prob = expf(simple_score_qk(q, k, b, h, query, key, H, T, Dh, scale) - max_score) / denom;
        acc += prob * output_cotangent_basis[basis_offset(b, p, h, query, d, P, H, T, Dh)];
    }
    dv[basis_offset(b, p, h, key, d, P, H, T, Dh)] = acc;
}

}  // namespace

std::vector<torch::Tensor> attention_input_pullback_basis_simple(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output_cotangent_basis,
    double softmax_scale,
    bool causal
) {
    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int T = static_cast<int>(q.size(2));
    const int Dh = static_cast<int>(q.size(3));
    const int P = static_cast<int>(output_cotangent_basis.size(1));
    const long total = static_cast<long>(B) * P * H * T * Dh;

    auto opts = output_cotangent_basis.options();
    auto dq = torch::empty({B, P, H, T, Dh}, opts);
    auto dk = torch::empty({B, P, H, T, Dh}, opts);
    auto dv = torch::empty({B, P, H, T, Dh}, opts);

    if (total == 0) {
        return {dq, dk, dv};
    }

    const int blocks = static_cast<int>((total + THREADS - 1) / THREADS);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float scale = static_cast<float>(softmax_scale);

    simple_dq_kernel<<<blocks, THREADS, 0, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        output_cotangent_basis.data_ptr<float>(),
        dq.data_ptr<float>(),
        B,
        P,
        H,
        T,
        Dh,
        scale,
        causal,
        total
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    simple_dk_kernel<<<blocks, THREADS, 0, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        output_cotangent_basis.data_ptr<float>(),
        dk.data_ptr<float>(),
        B,
        P,
        H,
        T,
        Dh,
        scale,
        causal,
        total
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    simple_dv_kernel<<<blocks, THREADS, 0, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        output_cotangent_basis.data_ptr<float>(),
        dv.data_ptr<float>(),
        B,
        P,
        H,
        T,
        Dh,
        scale,
        causal,
        total
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {dq, dk, dv};
}
