#include "suffix_scan.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

namespace {

void validate_mats(const torch::Tensor& mats) {
    TORCH_CHECK(mats.is_cuda(), "mats must be a CUDA tensor");
    TORCH_CHECK(mats.is_contiguous(), "mats must be contiguous");
    TORCH_CHECK(mats.dim() == 4, "mats must have shape [B, K, R, R]");
    TORCH_CHECK(mats.size(-1) == mats.size(-2), "mats must be square in the last two dimensions");
    TORCH_CHECK(
        mats.scalar_type() == at::kFloat || mats.scalar_type() == at::kHalf || mats.scalar_type() == at::kBFloat16,
        "mats must have float, half, or bfloat16 dtype"
    );
}

void check_cublas(cublasStatus_t status, const char* what) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cuBLAS status ", static_cast<int>(status));
}

void gemm_strided_batched_row_major(
    const float* lhs,
    const float* rhs,
    float* out,
    int batch_count,
    int rank
) {
    if (batch_count == 0) {
        return;
    }

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const long long stride = static_cast<long long>(rank) * rank;

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    auto stream = at::cuda::getCurrentCUDAStream();
    check_cublas(cublasSetStream(handle, stream.stream()), "cublasSetStream");

    // cuBLAS uses column-major storage. Swapping lhs/rhs gives the correct row-major product.
    check_cublas(
        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            rank,
            rank,
            rank,
            &alpha,
            rhs,
            rank,
            stride,
            lhs,
            rank,
            stride,
            &beta,
            out,
            rank,
            stride,
            batch_count
        ),
        "cublasSgemmStridedBatched"
    );
}

}  // namespace

torch::Tensor suffix_scan_mats_launcher(torch::Tensor mats) {
    validate_mats(mats);

    const auto batch_size = mats.size(0);
    const auto n_regions = mats.size(1);
    const auto rank = mats.size(2);
    auto float_opts = mats.options().dtype(at::kFloat);

    if (n_regions == 0) {
        auto eye = torch::eye(rank, float_opts).view({1, 1, rank, rank});
        return eye.expand({batch_size, 1, rank, rank}).clone();
    }

    // Store as [K, B, R, R] so each active scan prefix is contiguous across the
    // flattened (k, b) batch dimension. This lets cuBLAS read/write directly from
    // persistent buffers without per-level contiguous materialization.
    auto work = mats.to(at::kFloat).permute({1, 0, 2, 3}).contiguous().clone();
    auto next = torch::empty_like(work);

    for (int64_t offset = 1; offset < n_regions; offset *= 2) {
        const auto active = n_regions - offset;
        auto lhs = work.narrow(0, 0, active);
        auto rhs = work.narrow(0, offset, active);
        auto dst = next.narrow(0, 0, active);

        gemm_strided_batched_row_major(
            lhs.data_ptr<float>(),
            rhs.data_ptr<float>(),
            dst.data_ptr<float>(),
            static_cast<int>(active * batch_size),
            static_cast<int>(rank)
        );

        next.narrow(0, active, offset).copy_(work.narrow(0, active, offset));

        auto tmp = work;
        work = next;
        next = tmp;
    }

    auto out = torch::empty({batch_size, n_regions + 1, rank, rank}, float_opts);
    out.narrow(1, 0, n_regions).copy_(work.permute({1, 0, 2, 3}));
    auto eye = torch::eye(rank, float_opts).view({1, 1, rank, rank}).expand({batch_size, 1, rank, rank});
    out.narrow(1, n_regions, 1).copy_(eye);
    return out;
}
