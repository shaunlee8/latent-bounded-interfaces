#include "suffix_scan.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>

#include <algorithm>

namespace {

void validate_mats(const torch::Tensor &mats) {
	TORCH_CHECK(mats.is_cuda(), "mats must be a CUDA tensor");
	TORCH_CHECK(mats.is_contiguous(), "mats must be contiguous");
	TORCH_CHECK(mats.dim() == 4, "mats must have shape [B, K, R, R]");
	TORCH_CHECK(mats.size(-1) == mats.size(-2),
				"mats must be square in the last two dimensions");
	TORCH_CHECK(mats.scalar_type() == at::kFloat ||
					mats.scalar_type() == at::kHalf ||
					mats.scalar_type() == at::kBFloat16,
				"mats must have float, half, or bfloat16 dtype");
}

void check_cublas(cublasStatus_t status, const char *what) {
	TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what,
				" failed with cuBLAS status ", static_cast<int>(status));
}

void gemm_strided_batched_row_major(const float *lhs, const float *rhs,
									float *out, int batch_count, int rank) {
	if (batch_count == 0) {
		return;
	}

	constexpr float alpha = 1.0f;
	constexpr float beta = 0.0f;
	const long long stride = static_cast<long long>(rank) * rank;

	auto handle = at::cuda::getCurrentCUDABlasHandle();
	auto stream = at::cuda::getCurrentCUDAStream();
	check_cublas(cublasSetStream(handle, stream.stream()), "cublasSetStream");

	// cuBLAS uses column-major storage. Swapping lhs/rhs gives the correct
	// row-major product.
	check_cublas(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, rank,
										   rank, rank, &alpha, rhs, rank, stride,
										   lhs, rank, stride, &beta, out, rank,
										   stride, batch_count),
				 "cublasSgemmStridedBatched");
}

constexpr int kJacobianTile = 16;

__global__ void suffix_scan_jacobian_step_kernel(const float *__restrict__ work,
												 float *__restrict__ next,
												 int batch_size, int n_regions,
												 int rank, int offset) {
	const int tile_col = blockIdx.x;
	const int tile_row = blockIdx.y;
	const int batch_region = blockIdx.z;

	const int b = batch_region / n_regions;
	const int region = batch_region - b * n_regions;

	const int row = tile_row * kJacobianTile + threadIdx.y;
	const int col = tile_col * kJacobianTile + threadIdx.x;

	const int matrix_elems = rank * rank;
	const int lhs_base = (b * n_regions + region) * matrix_elems;

	if (region + offset >= n_regions) {
		if (row < rank && col < rank) {
			next[lhs_base + row * rank + col] = work[lhs_base + row * rank + col];
		}
		return;
	}

	const int rhs_base = (b * n_regions + region + offset) * matrix_elems;

	__shared__ float lhs_tile[kJacobianTile][kJacobianTile + 1];
	__shared__ float rhs_tile[kJacobianTile][kJacobianTile + 1];

	float acc = 0.0f;

	for (int k0 = 0; k0 < rank; k0 += kJacobianTile) {
		const int lhs_k = k0 + threadIdx.x;
		const int rhs_k = k0 + threadIdx.y;

		if (row < rank && lhs_k < rank) {
			lhs_tile[threadIdx.y][threadIdx.x] = work[lhs_base + row * rank + lhs_k];
		} else {
			lhs_tile[threadIdx.y][threadIdx.x] = 0.0f;
		}

		if (rhs_k < rank && col < rank) {
			rhs_tile[threadIdx.y][threadIdx.x] = work[rhs_base + rhs_k * rank + col];
		} else {
			rhs_tile[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

#pragma unroll
		for (int k = 0; k < kJacobianTile; ++k) {
			acc += lhs_tile[threadIdx.y][k] * rhs_tile[k][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < rank && col < rank) {
		next[lhs_base + row * rank + col] = acc;
	}
}

// specialized R=64 path, 1 threadblock computes 64x64 matmul, 1 thread computes 4x4 output tile
constexpr int kR64 = 64;
constexpr int kR64Tile = 16;

__global__ __launch_bounds__(256, 2) void suffix_scan_jacobian_r64_step_kernel(
	const float *__restrict__ work, float *__restrict__ next, int n_regions,
	int offset) {
	const int region = blockIdx.x;
	const int b = blockIdx.y;

	const int tx = threadIdx.x; // 0..15
	const int ty = threadIdx.y; // 0..15
	const int tid = ty * kR64Tile + tx;

	constexpr int matrix_elems = kR64 * kR64;

	const int lhs_base = (b * n_regions + region) * matrix_elems;

	if (region + offset >= n_regions) {
		constexpr int vec_elems = matrix_elems / 4;

		const float4 *work4 = reinterpret_cast<const float4 *>(work + lhs_base);
		float4 *next4 = reinterpret_cast<float4 *>(next + lhs_base);

#pragma unroll
		for (int idx = tid; idx < vec_elems; idx += kR64Tile * kR64Tile) {
			next4[idx] = work4[idx];
		}

		return;
	}

	const int rhs_base = (b * n_regions + region + offset) * matrix_elems;

	float acc[4][4];

#pragma unroll
	for (int i = 0; i < 4; ++i) {
#pragma unroll
		for (int j = 0; j < 4; ++j) {
			acc[i][j] = 0.0f;
		}
	}

	__shared__ float lhs_tile[kR64][kR64Tile + 1]; // 64 x 16
	__shared__ float rhs_tile[kR64Tile][kR64 + 1]; // 16 x 64

	for (int k0 = 0; k0 < kR64; k0 += kR64Tile) {

#pragma unroll
		for (int idx = tid; idx < kR64 * kR64Tile; idx += kR64Tile * kR64Tile) {
			const int row = idx >> 4; // idx / 16
			const int k = idx & 15;	  // idx % 16
			lhs_tile[row][k] = work[lhs_base + row * kR64 + (k0 + k)];
		}

#pragma unroll
		for (int idx = tid; idx < kR64Tile * kR64; idx += kR64Tile * kR64Tile) {
			const int k = idx >> 6;	  // idx / 64
			const int col = idx & 63; // idx % 64
			rhs_tile[k][col] = work[rhs_base + (k0 + k) * kR64 + col];
		}

		__syncthreads();

#pragma unroll
		for (int k = 0; k < kR64Tile; ++k) {
			const float a0 = lhs_tile[ty + 0 * kR64Tile][k];
			const float a1 = lhs_tile[ty + 1 * kR64Tile][k];
			const float a2 = lhs_tile[ty + 2 * kR64Tile][k];
			const float a3 = lhs_tile[ty + 3 * kR64Tile][k];

			const float b0 = rhs_tile[k][tx + 0 * kR64Tile];
			const float b1 = rhs_tile[k][tx + 1 * kR64Tile];
			const float b2 = rhs_tile[k][tx + 2 * kR64Tile];
			const float b3 = rhs_tile[k][tx + 3 * kR64Tile];

			acc[0][0] += a0 * b0;
			acc[0][1] += a0 * b1;
			acc[0][2] += a0 * b2;
			acc[0][3] += a0 * b3;

			acc[1][0] += a1 * b0;
			acc[1][1] += a1 * b1;
			acc[1][2] += a1 * b2;
			acc[1][3] += a1 * b3;

			acc[2][0] += a2 * b0;
			acc[2][1] += a2 * b1;
			acc[2][2] += a2 * b2;
			acc[2][3] += a2 * b3;

			acc[3][0] += a3 * b0;
			acc[3][1] += a3 * b1;
			acc[3][2] += a3 * b2;
			acc[3][3] += a3 * b3;
		}

		__syncthreads();
	}

#pragma unroll
	for (int i = 0; i < 4; ++i) {
		const int row = ty + i * kR64Tile;

#pragma unroll
		for (int j = 0; j < 4; ++j) {
			const int col = tx + j * kR64Tile;
			next[lhs_base + row * kR64 + col] = acc[i][j];
		}
	}
}

__global__ void suffix_scan_jacobian_write_output_kernel(
	const float *__restrict__ work, float *__restrict__ out, int batch_size,
	int n_regions, int rank) {
	const int64_t matrix_elems = static_cast<int64_t>(rank) * rank;
	const int64_t total =
		static_cast<int64_t>(batch_size) * (n_regions + 1) * matrix_elems;

	for (int64_t linear =
			 static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		 linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
		const int inner = static_cast<int>(linear % matrix_elems);
		const int col = inner % rank;
		const int row = inner / rank;

		const int suffix_idx =
			static_cast<int>((linear / matrix_elems) % (n_regions + 1));
		const int b = static_cast<int>(linear / (matrix_elems * (n_regions + 1)));

		if (suffix_idx < n_regions) {
			const int64_t src =
				(static_cast<int64_t>(b) * n_regions + suffix_idx) * matrix_elems +
				inner;
			out[linear] = work[src];
		} else {
			out[linear] = row == col ? 1.0f : 0.0f;
		}
	}
}

__global__ void suffix_scan_jacobian_r64_write_output_kernel(
	const float *__restrict__ work, float *__restrict__ out, int n_regions) {
	constexpr int matrix_elems = kR64 * kR64;
	constexpr int vec_elems = matrix_elems / 4;

	const int out_matrix = blockIdx.x;
	const int tid = threadIdx.x;

	const int suffix_stride = n_regions + 1;
	const int b = out_matrix / suffix_stride;
	const int suffix_idx = out_matrix - b * suffix_stride;

	const int64_t out_base = static_cast<int64_t>(out_matrix) * matrix_elems;

	if (suffix_idx < n_regions) {
		const int64_t src_base =
			(static_cast<int64_t>(b) * n_regions + suffix_idx) * matrix_elems;

		const float4 *__restrict__ work4 =
			reinterpret_cast<const float4 *>(work + src_base);
		float4 *__restrict__ out4 = reinterpret_cast<float4 *>(out + out_base);

#pragma unroll
		for (int idx = tid; idx < vec_elems; idx += blockDim.x) {
			out4[idx] = work4[idx];
		}
	} else {
#pragma unroll
		for (int idx = tid; idx < matrix_elems; idx += blockDim.x) {
			const int row = idx >> 6; // idx / 64
			const int col = idx & 63; // idx % 64
			out[out_base + idx] = row == col ? 1.0f : 0.0f;
		}
	}
}

} // namespace

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
			lhs.data_ptr<float>(), rhs.data_ptr<float>(), dst.data_ptr<float>(),
			static_cast<int>(active * batch_size), static_cast<int>(rank));

		next.narrow(0, active, offset).copy_(work.narrow(0, active, offset));

		auto tmp = work;
		work = next;
		next = tmp;
	}

	auto out = torch::empty({batch_size, n_regions + 1, rank, rank}, float_opts);
	out.narrow(1, 0, n_regions).copy_(work.permute({1, 0, 2, 3}));
	auto eye = torch::eye(rank, float_opts)
				   .view({1, 1, rank, rank})
				   .expand({batch_size, 1, rank, rank});
	out.narrow(1, n_regions, 1).copy_(eye);
	return out;
}

torch::Tensor suffix_scan_jacobian(torch::Tensor jacobian) {
	validate_mats(jacobian);

	const auto batch_size = jacobian.size(0);
	const auto n_regions = jacobian.size(1);
	const auto rank = jacobian.size(2);
	auto float_opts = jacobian.options().dtype(at::kFloat);

	if (jacobian.scalar_type() != at::kFloat || batch_size == 0 ||
		n_regions == 0 || n_regions > 32 || rank <= 0 || rank > 64) {
		return suffix_scan_mats_launcher(jacobian);
	}

	const int batch_size_i = static_cast<int>(batch_size);
	const int n_regions_i = static_cast<int>(n_regions);
	const int rank_i = static_cast<int>(rank);

	auto stream = at::cuda::getCurrentCUDAStream();

	torch::Tensor work;

	if (rank_i == 64) {
		auto tmp0 = torch::empty_like(jacobian);
		auto tmp1 = torch::empty_like(jacobian);

		const dim3 block(kR64Tile, kR64Tile);
		const dim3 grid(n_regions_i, batch_size_i);

		int level = 0;

		for (int offset = 1; offset < n_regions_i; offset *= 2, ++level) {
			const float *src_ptr = nullptr;
			torch::Tensor dst;

			if (level == 0) {
				src_ptr = jacobian.data_ptr<float>();
				dst = tmp0;
			} else if (level % 2 == 1) {
				src_ptr = tmp0.data_ptr<float>();
				dst = tmp1;
			} else {
				src_ptr = tmp1.data_ptr<float>();
				dst = tmp0;
			}

			suffix_scan_jacobian_r64_step_kernel<<<grid, block, 0, stream>>>(
				src_ptr, dst.data_ptr<float>(), n_regions_i, offset);
			C10_CUDA_KERNEL_LAUNCH_CHECK();

			work = dst;
		}

		// handle K == 1
		if (level == 0) {
			work = jacobian;
		}
	} else {

		auto tmp0 = torch::empty_like(jacobian);
		auto tmp1 = torch::empty_like(jacobian);

		const dim3 block(kJacobianTile, kJacobianTile);
		const dim3 grid((rank_i + kJacobianTile - 1) / kJacobianTile,
						(rank_i + kJacobianTile - 1) / kJacobianTile,
						batch_size_i * n_regions_i);

		int level = 0;

		for (int offset = 1; offset < n_regions_i; offset *= 2, ++level) {
			const float *src_ptr = nullptr;
			torch::Tensor dst;

			if (level == 0) {
				src_ptr = jacobian.data_ptr<float>();
				dst = tmp0;
			} else if (level % 2 == 1) {
				src_ptr = tmp0.data_ptr<float>();
				dst = tmp1;
			} else {
				src_ptr = tmp1.data_ptr<float>();
				dst = tmp0;
			}

			suffix_scan_jacobian_step_kernel<<<grid, block, 0, stream>>>(
				src_ptr, dst.data_ptr<float>(), batch_size_i, n_regions_i, rank_i,
				offset);
			C10_CUDA_KERNEL_LAUNCH_CHECK();

			work = dst;
		}

		if (level == 0) {
			work = jacobian;
		}
	}

	auto out = torch::empty({batch_size, n_regions + 1, rank, rank}, float_opts);

	const int threads = 256;

	if (rank_i == 64) {
		const int blocks = batch_size_i * (n_regions_i + 1);

		suffix_scan_jacobian_r64_write_output_kernel<<<blocks, threads, 0,
													   stream>>>(
			work.data_ptr<float>(), out.data_ptr<float>(), n_regions_i);
		C10_CUDA_KERNEL_LAUNCH_CHECK();
	} else {
		const int64_t total = batch_size * (n_regions + 1) * rank * rank;
		const int blocks = static_cast<int>(
			std::min<int64_t>((total + threads - 1) / threads, 4096));

		suffix_scan_jacobian_write_output_kernel<<<blocks, threads, 0, stream>>>(
			work.data_ptr<float>(), out.data_ptr<float>(), batch_size_i,
			n_regions_i, rank_i);
		C10_CUDA_KERNEL_LAUNCH_CHECK();
	}

	return out;
}
