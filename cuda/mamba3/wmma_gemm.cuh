// Shared wmma (16x16x16 bf16, fp32 accum) GEMM helpers over smem/gmem
// pointers, warp-strided output tiles; used by the dual-scan opt kernels.
#pragma once

#include <cuda_bf16.h>
#include <mma.h>

namespace lbi_mamba3 {
namespace wmma_gemm {

using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

// C[M,N] fp32 = A @ B^T, A stored [M,K] row-major, B stored [N,K] row-major.
__device__ inline void gemm_ABt(float* C, int ldc, const bf16* A, int lda,
                                const bf16* B, int ldb, int M, int N, int K,
                                int warp, int nwarps, bool accumulate) {
    const int mt = M / WMMA_M, nt = N / WMMA_N, kt = K / WMMA_K;
    for (int t = warp; t < mt * nt; t += nwarps) {
        const int im = (t / nt) * WMMA_M, in = (t % nt) * WMMA_N;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        if (accumulate) wmma::load_matrix_sync(acc, C + im * ldc + in, ldc, wmma::mem_row_major);
        else wmma::fill_fragment(acc, 0.f);
        for (int k = 0; k < kt; ++k) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::col_major> b;
            wmma::load_matrix_sync(a, A + im * lda + k * WMMA_K, lda);
            wmma::load_matrix_sync(b, B + in * ldb + k * WMMA_K, ldb);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(C + im * ldc + in, acc, ldc, wmma::mem_row_major);
    }
}

// C[M,N] fp32 = A @ B, A stored [M,K] row-major, B stored [K,N] row-major.
__device__ inline void gemm_AB(float* C, int ldc, const bf16* A, int lda,
                               const bf16* B, int ldb, int M, int N, int K,
                               int warp, int nwarps, bool accumulate) {
    const int mt = M / WMMA_M, nt = N / WMMA_N, kt = K / WMMA_K;
    for (int t = warp; t < mt * nt; t += nwarps) {
        const int im = (t / nt) * WMMA_M, in = (t % nt) * WMMA_N;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        if (accumulate) wmma::load_matrix_sync(acc, C + im * ldc + in, ldc, wmma::mem_row_major);
        else wmma::fill_fragment(acc, 0.f);
        for (int k = 0; k < kt; ++k) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major> b;
            wmma::load_matrix_sync(a, A + im * lda + k * WMMA_K, lda);
            wmma::load_matrix_sync(b, B + (k * WMMA_K) * ldb + in, ldb);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(C + im * ldc + in, acc, ldc, wmma::mem_row_major);
    }
}

// C[M,N] fp32 = A^T @ B, A stored [K,M] row-major, B stored [K,N] row-major.
__device__ inline void gemm_AtB(float* C, int ldc, const bf16* A, int lda,
                                const bf16* B, int ldb, int M, int N, int K,
                                int warp, int nwarps, bool accumulate) {
    const int mt = M / WMMA_M, nt = N / WMMA_N, kt = K / WMMA_K;
    for (int t = warp; t < mt * nt; t += nwarps) {
        const int im = (t / nt) * WMMA_M, in = (t % nt) * WMMA_N;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        if (accumulate) wmma::load_matrix_sync(acc, C + im * ldc + in, ldc, wmma::mem_row_major);
        else wmma::fill_fragment(acc, 0.f);
        for (int k = 0; k < kt; ++k) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major> b;
            wmma::load_matrix_sync(a, A + (k * WMMA_K) * lda + im, lda);
            wmma::load_matrix_sync(b, B + (k * WMMA_K) * ldb + in, ldb);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(C + im * ldc + in, acc, ldc, wmma::mem_row_major);
    }
}

// JVP GEMM pairs: C (+)= A@B and dC (+)= dA@B + A@dB in one interleaved
// loop; each operand fragment loads once and the mma chains overlap.

__device__ inline void gemm_jvp_AB(float* C, float* dC, int ldc,
                                   const bf16* A, const bf16* dA, int lda,
                                   const bf16* B, const bf16* dB, int ldb,
                                   int M, int N, int K, int warp, int nwarps,
                                   bool accumulate) {
    const int mt = M / WMMA_M, nt = N / WMMA_N, kt = K / WMMA_K;
    for (int t = warp; t < mt * nt; t += nwarps) {
        const int im = (t / nt) * WMMA_M, in = (t % nt) * WMMA_N;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc, dacc;
        if (accumulate) {
            wmma::load_matrix_sync(acc, C + im * ldc + in, ldc, wmma::mem_row_major);
            wmma::load_matrix_sync(dacc, dC + im * ldc + in, ldc, wmma::mem_row_major);
        } else {
            wmma::fill_fragment(acc, 0.f);
            wmma::fill_fragment(dacc, 0.f);
        }
        for (int k = 0; k < kt; ++k) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major> a, da;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major> b, db;
            wmma::load_matrix_sync(a, A + im * lda + k * WMMA_K, lda);
            wmma::load_matrix_sync(da, dA + im * lda + k * WMMA_K, lda);
            wmma::load_matrix_sync(b, B + (k * WMMA_K) * ldb + in, ldb);
            wmma::load_matrix_sync(db, dB + (k * WMMA_K) * ldb + in, ldb);
            wmma::mma_sync(acc, a, b, acc);
            wmma::mma_sync(dacc, da, b, dacc);
            wmma::mma_sync(dacc, a, db, dacc);
        }
        wmma::store_matrix_sync(C + im * ldc + in, acc, ldc, wmma::mem_row_major);
        wmma::store_matrix_sync(dC + im * ldc + in, dacc, ldc, wmma::mem_row_major);
    }
}

__device__ inline void gemm_jvp_ABt(float* C, float* dC, int ldc,
                                    const bf16* A, const bf16* dA, int lda,
                                    const bf16* B, const bf16* dB, int ldb,
                                    int M, int N, int K, int warp, int nwarps,
                                    bool accumulate) {
    const int mt = M / WMMA_M, nt = N / WMMA_N, kt = K / WMMA_K;
    for (int t = warp; t < mt * nt; t += nwarps) {
        const int im = (t / nt) * WMMA_M, in = (t % nt) * WMMA_N;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc, dacc;
        if (accumulate) {
            wmma::load_matrix_sync(acc, C + im * ldc + in, ldc, wmma::mem_row_major);
            wmma::load_matrix_sync(dacc, dC + im * ldc + in, ldc, wmma::mem_row_major);
        } else {
            wmma::fill_fragment(acc, 0.f);
            wmma::fill_fragment(dacc, 0.f);
        }
        for (int k = 0; k < kt; ++k) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major> a, da;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::col_major> b, db;
            wmma::load_matrix_sync(a, A + im * lda + k * WMMA_K, lda);
            wmma::load_matrix_sync(da, dA + im * lda + k * WMMA_K, lda);
            wmma::load_matrix_sync(b, B + in * ldb + k * WMMA_K, ldb);
            wmma::load_matrix_sync(db, dB + in * ldb + k * WMMA_K, ldb);
            wmma::mma_sync(acc, a, b, acc);
            wmma::mma_sync(dacc, da, b, dacc);
            wmma::mma_sync(dacc, a, db, dacc);
        }
        wmma::store_matrix_sync(C + im * ldc + in, acc, ldc, wmma::mem_row_major);
        wmma::store_matrix_sync(dC + im * ldc + in, dacc, ldc, wmma::mem_row_major);
    }
}

}  // namespace wmma_gemm
}  // namespace lbi_mamba3
