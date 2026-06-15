/*
CUTLASS/CuTe-assisted CUDA implementation for the LBI-2 Transformer attention
input pullback.

The fp16 Dh=64 causal path uses WMMA tensor-core tiles. The current body uses
one multi-warp CTA per query/key tile so QK and dO V^T are computed once and
reused across the four 16-wide head-dimension blocks.
*/
#include "attention_pullback_mma.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace {

constexpr int THREADS = 128;
constexpr int HDIM64 = 64;
constexpr int MAX_SEQUENCE = 1024;
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;
constexpr int WARP_THREADS = 32;
constexpr int DIM_BLOCKS = HDIM64 / TILE_N;

struct MmaP1Params {
    const at::Half* q;
    const at::Half* k;
    const at::Half* v;
    const at::Half* out;
    const at::Half* dO;
    const float* logsumexp;
    float* dsoftmax_sum;
    at::Half* dq;
    at::Half* dk;
    at::Half* dv;
    int B;
    int H;
    int T;
    float scale;
};

__device__ inline long qkv_offset(int b, int h, int t, int d, int H, int T) {
    return (((static_cast<long>(b) * H + h) * T + t) * HDIM64 + d);
}

__device__ inline long row_offset(int b, int h, int t, int H, int T) {
    return ((static_cast<long>(b) * H + h) * T + t);
}

__device__ inline float hfloat(at::Half x) {
    return static_cast<float>(x);
}

__global__ void mma_p1_dq_wmma_hdim64_causal_kernel(MmaP1Params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    const int query_block = blockIdx.x;
    const int h = blockIdx.y % params.H;
    const int b = blockIdx.y / params.H;
    const int q_start = query_block * TILE_M;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_THREADS;
    const int lane = tid % WARP_THREADS;
    (void)lane;

    extern __shared__ unsigned char raw_shared[];
    at::Half* q_tile = reinterpret_cast<at::Half*>(raw_shared);
    at::Half* do_tile = q_tile + TILE_M * HDIM64;
    float* row_sum_tile = reinterpret_cast<float*>(do_tile + TILE_M * HDIM64);
    at::Half* k_tile = reinterpret_cast<at::Half*>(row_sum_tile + TILE_M);
    at::Half* v_tile = k_tile + TILE_N * HDIM64;
    at::Half* ds_tile = v_tile + TILE_N * HDIM64;
    float* score_parts = reinterpret_cast<float*>(ds_tile + TILE_M * TILE_N);
    float* dp_parts = score_parts + DIM_BLOCKS * TILE_M * TILE_N;
    float* score_tile = dp_parts + DIM_BLOCKS * TILE_M * TILE_N;
    float* dp_tile = score_tile + TILE_M * TILE_N;
    float* dq_out = dp_tile + TILE_M * TILE_N;

    for (int idx = tid; idx < TILE_M * HDIM64; idx += blockDim.x) {
        const int m = idx / HDIM64;
        const int d = idx % HDIM64;
        const int q_abs = q_start + m;
        q_tile[idx] = params.q[qkv_offset(b, h, q_abs, d, params.H, params.T)];
        do_tile[idx] = params.dO[qkv_offset(b, h, q_abs, d, params.H, params.T)];
    }
    __syncthreads();

    for (int m = tid; m < TILE_M; m += blockDim.x) {
        const int q_abs = q_start + m;
        float acc = 0.0f;
        #pragma unroll
        for (int d = 0; d < HDIM64; ++d) {
            acc += hfloat(do_tile[m * HDIM64 + d]) * hfloat(params.out[qkv_offset(b, h, q_abs, d, params.H, params.T)]);
        }
        params.dsoftmax_sum[row_offset(b, h, q_abs, params.H, params.T)] = acc;
        row_sum_tile[m] = acc;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_dq;
    wmma::fill_fragment(acc_dq, 0.0f);

    for (int key_start = 0; key_start <= q_start + TILE_M - 1 && key_start < params.T; key_start += TILE_N) {
        for (int idx = tid; idx < TILE_N * HDIM64; idx += blockDim.x) {
            const int n = idx / HDIM64;
            const int d = idx % HDIM64;
            const int key_abs = key_start + n;
            k_tile[idx] = params.k[qkv_offset(b, h, key_abs, d, params.H, params.T)];
            v_tile[idx] = params.v[qkv_offset(b, h, key_abs, d, params.H, params.T)];
        }
        __syncthreads();

        {
            const int d_base = warp_id * TILE_K;
            wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_s;
            wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_dp;
            wmma::fill_fragment(acc_s, 0.0f);
            wmma::fill_fragment(acc_dp, 0.0f);
            wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> do_frag;
            wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::col_major> k_frag;
            wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::col_major> v_frag;
            wmma::load_matrix_sync(q_frag, reinterpret_cast<half*>(q_tile + d_base), HDIM64);
            wmma::load_matrix_sync(do_frag, reinterpret_cast<half*>(do_tile + d_base), HDIM64);
            wmma::load_matrix_sync(k_frag, reinterpret_cast<half*>(k_tile + d_base), HDIM64);
            wmma::load_matrix_sync(v_frag, reinterpret_cast<half*>(v_tile + d_base), HDIM64);
            wmma::mma_sync(acc_s, q_frag, k_frag, acc_s);
            wmma::mma_sync(acc_dp, do_frag, v_frag, acc_dp);
            wmma::store_matrix_sync(score_parts + warp_id * TILE_M * TILE_N, acc_s, TILE_N, wmma::mem_row_major);
            wmma::store_matrix_sync(dp_parts + warp_id * TILE_M * TILE_N, acc_dp, TILE_N, wmma::mem_row_major);
        }
        __syncthreads();

        for (int idx = tid; idx < TILE_M * TILE_N; idx += blockDim.x) {
            float score = 0.0f;
            float dp = 0.0f;
            #pragma unroll
            for (int part = 0; part < DIM_BLOCKS; ++part) {
                score += score_parts[part * TILE_M * TILE_N + idx];
                dp += dp_parts[part * TILE_M * TILE_N + idx];
            }
            score_tile[idx] = score;
            dp_tile[idx] = dp;
        }
        __syncthreads();

        for (int idx = tid; idx < TILE_M * TILE_N; idx += blockDim.x) {
            const int m = idx / TILE_N;
            const int n = idx % TILE_N;
            const int q_abs = q_start + m;
            const int key_abs = key_start + n;
            float ds = 0.0f;
            if (key_abs <= q_abs) {
                const float prob = expf(score_tile[idx] * params.scale - params.logsumexp[row_offset(b, h, q_abs, params.H, params.T)]);
                ds = prob * (dp_tile[idx] - row_sum_tile[m]);
            }
            ds_tile[idx] = static_cast<at::Half>(ds);
        }
        __syncthreads();

        const int d_start = warp_id * TILE_N;
        wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> ds_frag;
        wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::row_major> k_sub_frag;
        wmma::load_matrix_sync(ds_frag, reinterpret_cast<half*>(ds_tile), TILE_N);
        wmma::load_matrix_sync(k_sub_frag, reinterpret_cast<half*>(k_tile + d_start), HDIM64);
        wmma::mma_sync(acc_dq, ds_frag, k_sub_frag, acc_dq);
        __syncthreads();
    }

    float* warp_dq_out = dq_out + warp_id * TILE_M * TILE_N;
    wmma::store_matrix_sync(warp_dq_out, acc_dq, TILE_N, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < DIM_BLOCKS * TILE_M * TILE_N; idx += blockDim.x) {
        const int dim_block = idx / (TILE_M * TILE_N);
        const int rem = idx - dim_block * TILE_M * TILE_N;
        const int m = rem / TILE_N;
        const int d = rem % TILE_N;
        const int q_abs = q_start + m;
        const int d_abs = dim_block * TILE_N + d;
        params.dq[qkv_offset(b, h, q_abs, d_abs, params.H, params.T)] = static_cast<at::Half>(dq_out[idx] * params.scale);
    }
#endif
}

__global__ void mma_p1_dkdv_wmma_hdim64_causal_kernel(MmaP1Params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    const int key_block = blockIdx.x;
    const int h = blockIdx.y % params.H;
    const int b = blockIdx.y / params.H;
    const int key_start = key_block * TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_THREADS;
    const int lane = tid % WARP_THREADS;
    (void)lane;

    extern __shared__ unsigned char raw_shared[];
    at::Half* q_tile = reinterpret_cast<at::Half*>(raw_shared);
    at::Half* do_tile = q_tile + TILE_M * HDIM64;
    at::Half* k_tile = do_tile + TILE_M * HDIM64;
    at::Half* v_tile = k_tile + TILE_N * HDIM64;
    at::Half* ds_tile = v_tile + TILE_N * HDIM64;
    at::Half* prob_tile = ds_tile + TILE_M * TILE_N;
    float* score_parts = reinterpret_cast<float*>(prob_tile + TILE_M * TILE_N);
    float* dp_parts = score_parts + DIM_BLOCKS * TILE_M * TILE_N;
    float* score_tile = dp_parts + DIM_BLOCKS * TILE_M * TILE_N;
    float* dp_tile = score_tile + TILE_M * TILE_N;
    float* dk_out = dp_tile + TILE_M * TILE_N;
    float* dv_out = dk_out + DIM_BLOCKS * TILE_N * TILE_N;

    for (int idx = tid; idx < TILE_N * HDIM64; idx += blockDim.x) {
        const int n = idx / HDIM64;
        const int d = idx % HDIM64;
        const int key_abs = key_start + n;
        k_tile[idx] = params.k[qkv_offset(b, h, key_abs, d, params.H, params.T)];
        v_tile[idx] = params.v[qkv_offset(b, h, key_abs, d, params.H, params.T)];
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_dk;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_dv;
    wmma::fill_fragment(acc_dk, 0.0f);
    wmma::fill_fragment(acc_dv, 0.0f);

    for (int q_start = key_start; q_start < params.T; q_start += TILE_M) {
        for (int idx = tid; idx < TILE_M * HDIM64; idx += blockDim.x) {
            const int m = idx / HDIM64;
            const int d = idx % HDIM64;
            const int q_abs = q_start + m;
            q_tile[idx] = params.q[qkv_offset(b, h, q_abs, d, params.H, params.T)];
            do_tile[idx] = params.dO[qkv_offset(b, h, q_abs, d, params.H, params.T)];
        }
        __syncthreads();

        {
            const int d_base = warp_id * TILE_K;
            wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_s;
            wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_dp;
            wmma::fill_fragment(acc_s, 0.0f);
            wmma::fill_fragment(acc_dp, 0.0f);
            wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> do_frag;
            wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::col_major> k_frag;
            wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::col_major> v_frag;
            wmma::load_matrix_sync(q_frag, reinterpret_cast<half*>(q_tile + d_base), HDIM64);
            wmma::load_matrix_sync(do_frag, reinterpret_cast<half*>(do_tile + d_base), HDIM64);
            wmma::load_matrix_sync(k_frag, reinterpret_cast<half*>(k_tile + d_base), HDIM64);
            wmma::load_matrix_sync(v_frag, reinterpret_cast<half*>(v_tile + d_base), HDIM64);
            wmma::mma_sync(acc_s, q_frag, k_frag, acc_s);
            wmma::mma_sync(acc_dp, do_frag, v_frag, acc_dp);
            wmma::store_matrix_sync(score_parts + warp_id * TILE_M * TILE_N, acc_s, TILE_N, wmma::mem_row_major);
            wmma::store_matrix_sync(dp_parts + warp_id * TILE_M * TILE_N, acc_dp, TILE_N, wmma::mem_row_major);
        }
        __syncthreads();

        for (int idx = tid; idx < TILE_M * TILE_N; idx += blockDim.x) {
            float score = 0.0f;
            float dp = 0.0f;
            #pragma unroll
            for (int part = 0; part < DIM_BLOCKS; ++part) {
                score += score_parts[part * TILE_M * TILE_N + idx];
                dp += dp_parts[part * TILE_M * TILE_N + idx];
            }
            score_tile[idx] = score;
            dp_tile[idx] = dp;
        }
        __syncthreads();

        for (int idx = tid; idx < TILE_M * TILE_N; idx += blockDim.x) {
            const int m = idx / TILE_N;
            const int n = idx % TILE_N;
            const int q_abs = q_start + m;
            const int key_abs = key_start + n;
            float prob = 0.0f;
            float ds = 0.0f;
            if (key_abs <= q_abs) {
                prob = expf(score_tile[idx] * params.scale - params.logsumexp[row_offset(b, h, q_abs, params.H, params.T)]);
                ds = prob * (dp_tile[idx] - params.dsoftmax_sum[row_offset(b, h, q_abs, params.H, params.T)]);
            }
            prob_tile[idx] = static_cast<at::Half>(prob);
            ds_tile[idx] = static_cast<at::Half>(ds);
        }
        __syncthreads();

        const int d_start = warp_id * TILE_N;
        wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::col_major> ds_t_frag;
        wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::col_major> prob_t_frag;
        wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::row_major> q_sub_frag;
        wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::row_major> do_sub_frag;
        wmma::load_matrix_sync(ds_t_frag, reinterpret_cast<half*>(ds_tile), TILE_N);
        wmma::load_matrix_sync(prob_t_frag, reinterpret_cast<half*>(prob_tile), TILE_N);
        wmma::load_matrix_sync(q_sub_frag, reinterpret_cast<half*>(q_tile + d_start), HDIM64);
        wmma::load_matrix_sync(do_sub_frag, reinterpret_cast<half*>(do_tile + d_start), HDIM64);
        wmma::mma_sync(acc_dk, ds_t_frag, q_sub_frag, acc_dk);
        wmma::mma_sync(acc_dv, prob_t_frag, do_sub_frag, acc_dv);
        __syncthreads();
    }

    float* warp_dk_out = dk_out + warp_id * TILE_N * TILE_N;
    float* warp_dv_out = dv_out + warp_id * TILE_N * TILE_N;
    wmma::store_matrix_sync(warp_dk_out, acc_dk, TILE_N, wmma::mem_row_major);
    wmma::store_matrix_sync(warp_dv_out, acc_dv, TILE_N, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < DIM_BLOCKS * TILE_N * TILE_N; idx += blockDim.x) {
        const int dim_block = idx / (TILE_N * TILE_N);
        const int rem = idx - dim_block * TILE_N * TILE_N;
        const int n = rem / TILE_N;
        const int d = rem % TILE_N;
        const int key_abs = key_start + n;
        const int d_abs = dim_block * TILE_N + d;
        params.dk[qkv_offset(b, h, key_abs, d_abs, params.H, params.T)] = static_cast<at::Half>(dk_out[idx] * params.scale);
        params.dv[qkv_offset(b, h, key_abs, d_abs, params.H, params.T)] = static_cast<at::Half>(dv_out[idx]);
    }
#endif
}

void validate_mma_p1_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& out,
    const torch::Tensor& logsumexp,
    const torch::Tensor& output_cotangent,
    bool causal
) {
    TORCH_CHECK(causal, "fa2_mma_p1 currently supports causal=True only");
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda() && v.is_cuda() && out.is_cuda() && logsumexp.is_cuda() && output_cotangent.is_cuda(),
                "all fa2_mma_p1 inputs must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4, "q must have shape [B, H, T, Dh]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape");
    TORCH_CHECK(v.sizes() == q.sizes(), "v must match q shape");
    TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
    TORCH_CHECK(output_cotangent.sizes() == q.sizes(), "output_cotangent must match q shape");
    TORCH_CHECK(logsumexp.dim() == 3, "logsumexp must have shape [B, H, T]");
    TORCH_CHECK(logsumexp.size(0) == q.size(0) && logsumexp.size(1) == q.size(1) && logsumexp.size(2) == q.size(2),
                "logsumexp must have shape [B, H, T] matching q");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && out.is_contiguous() && output_cotangent.is_contiguous(),
                "q/k/v/out/output_cotangent must be contiguous");
    TORCH_CHECK(logsumexp.is_contiguous(), "logsumexp must be contiguous");
    TORCH_CHECK(q.scalar_type() == at::kHalf, "fa2_mma_p1 currently supports float16 only");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type() &&
                out.scalar_type() == q.scalar_type() && output_cotangent.scalar_type() == q.scalar_type(),
                "q/k/v/out/output_cotangent dtypes must match");
    TORCH_CHECK(logsumexp.scalar_type() == at::kFloat, "logsumexp must be float32");
    TORCH_CHECK(q.size(3) == HDIM64, "fa2_mma_p1 currently supports Dh=64 only");
    TORCH_CHECK(q.size(2) <= MAX_SEQUENCE, "fa2_mma_p1 current skeleton supports T <= 1024");
    TORCH_CHECK(q.size(2) % TILE_M == 0, "fa2_mma_p1 currently requires T to be a multiple of 16");
}


void validate_mma_pblock4_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& out,
    const torch::Tensor& logsumexp,
    const torch::Tensor& output_cotangent_basis,
    bool causal
) {
    TORCH_CHECK(causal, "fa2_mma_pblock4 currently supports causal=True only");
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda() && v.is_cuda() && out.is_cuda() && logsumexp.is_cuda() && output_cotangent_basis.is_cuda(),
                "all fa2_mma_pblock4 inputs must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4, "q must have shape [B, H, T, Dh]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape");
    TORCH_CHECK(v.sizes() == q.sizes(), "v must match q shape");
    TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
    TORCH_CHECK(output_cotangent_basis.dim() == 5, "output_cotangent_basis must have shape [B, P, H, T, Dh]");
    TORCH_CHECK(output_cotangent_basis.size(0) == q.size(0) && output_cotangent_basis.size(2) == q.size(1) &&
                output_cotangent_basis.size(3) == q.size(2) && output_cotangent_basis.size(4) == q.size(3),
                "output_cotangent_basis must have shape [B, P, H, T, Dh] matching q/k/v");
    TORCH_CHECK(output_cotangent_basis.size(1) > 0 && output_cotangent_basis.size(1) <= 4,
                "fa2_mma_pblock4 currently supports 1 <= P <= 4");
    TORCH_CHECK(logsumexp.dim() == 3, "logsumexp must have shape [B, H, T]");
    TORCH_CHECK(logsumexp.size(0) == q.size(0) && logsumexp.size(1) == q.size(1) && logsumexp.size(2) == q.size(2),
                "logsumexp must have shape [B, H, T] matching q");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && out.is_contiguous() &&
                output_cotangent_basis.is_contiguous(),
                "q/k/v/out/output_cotangent_basis must be contiguous");
    TORCH_CHECK(logsumexp.is_contiguous(), "logsumexp must be contiguous");
    TORCH_CHECK(q.scalar_type() == at::kHalf, "fa2_mma_pblock4 currently supports float16 only");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type() &&
                out.scalar_type() == q.scalar_type() && output_cotangent_basis.scalar_type() == q.scalar_type(),
                "q/k/v/out/output_cotangent_basis dtypes must match");
    TORCH_CHECK(logsumexp.scalar_type() == at::kFloat, "logsumexp must be float32");
    TORCH_CHECK(q.size(3) == HDIM64, "fa2_mma_pblock4 currently supports Dh=64 only");
    TORCH_CHECK(q.size(2) <= MAX_SEQUENCE, "fa2_mma_pblock4 current skeleton supports T <= 1024");
    TORCH_CHECK(q.size(2) % TILE_M == 0, "fa2_mma_pblock4 currently requires T to be a multiple of 16");
}

}  // namespace

std::vector<torch::Tensor> attention_input_pullback_basis_mma_p1(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor logsumexp,
    torch::Tensor output_cotangent,
    double softmax_scale,
    bool causal
) {
    validate_mma_p1_inputs(q, k, v, out, logsumexp, output_cotangent, causal);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    // Verify that the required CuTe/CUTLASS headers expose the symbols used by
    // this translation unit.
    static_assert(cute::rank(cute::make_shape(cute::Int<1>{})) == 1);
    static_assert(sizeof(cutlass::half_t) == 2);

    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int T = static_cast<int>(q.size(2));
    const long total_rows = static_cast<long>(B) * H * T;

    auto dq = torch::empty_like(q);
    auto dk = torch::empty_like(k);
    auto dv = torch::empty_like(v);
    auto dsoftmax_sum = torch::empty({B, H, T}, logsumexp.options());

    if (total_rows == 0) {
        return {dq, dk, dv};
    }

    MmaP1Params params{
        q.data_ptr<at::Half>(),
        k.data_ptr<at::Half>(),
        v.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        output_cotangent.data_ptr<at::Half>(),
        logsumexp.data_ptr<float>(),
        dsoftmax_sum.data_ptr<float>(),
        dq.data_ptr<at::Half>(),
        dk.data_ptr<at::Half>(),
        dv.data_ptr<at::Half>(),
        B,
        H,
        T,
        static_cast<float>(softmax_scale),
    };

    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 tile_grid((T + TILE_M - 1) / TILE_M, B * H);
    const size_t dq_shared_bytes = static_cast<size_t>(
        2 * TILE_M * HDIM64 * sizeof(at::Half) +
        TILE_M * sizeof(float) +
        2 * TILE_N * HDIM64 * sizeof(at::Half) +
        TILE_M * TILE_N * sizeof(at::Half) +
        2 * DIM_BLOCKS * TILE_M * TILE_N * sizeof(float) +
        2 * TILE_M * TILE_N * sizeof(float) +
        DIM_BLOCKS * TILE_M * TILE_N * sizeof(float)
    );
    mma_p1_dq_wmma_hdim64_causal_kernel<<<tile_grid, THREADS, dq_shared_bytes, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 key_grid((T + TILE_N - 1) / TILE_N, B * H);
    const size_t dkdv_shared_bytes = static_cast<size_t>(
        2 * TILE_M * HDIM64 * sizeof(at::Half) +
        2 * TILE_N * HDIM64 * sizeof(at::Half) +
        2 * TILE_M * TILE_N * sizeof(at::Half) +
        2 * DIM_BLOCKS * TILE_M * TILE_N * sizeof(float) +
        2 * TILE_M * TILE_N * sizeof(float) +
        2 * DIM_BLOCKS * TILE_N * TILE_N * sizeof(float)
    );
    mma_p1_dkdv_wmma_hdim64_causal_kernel<<<key_grid, THREADS, dkdv_shared_bytes, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {dq, dk, dv};
}


std::vector<torch::Tensor> attention_input_pullback_basis_mma_pblock4(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor logsumexp,
    torch::Tensor output_cotangent_basis,
    double softmax_scale,
    bool causal
) {
    validate_mma_pblock4_inputs(q, k, v, out, logsumexp, output_cotangent_basis, causal);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    const int P = static_cast<int>(output_cotangent_basis.size(1));
    auto dq = torch::empty_like(output_cotangent_basis);
    auto dk = torch::empty_like(output_cotangent_basis);
    auto dv = torch::empty_like(output_cotangent_basis);

    for (int p = 0; p < P; ++p) {
        auto dO_p = output_cotangent_basis.select(1, p).contiguous();
        auto partial = attention_input_pullback_basis_mma_p1(
            q,
            k,
            v,
            out,
            logsumexp,
            dO_p,
            softmax_scale,
            causal
        );
        dq.select(1, p).copy_(partial[0]);
        dk.select(1, p).copy_(partial[1]);
        dv.select(1, p).copy_(partial[2]);
    }

    return {dq, dk, dv};
}

