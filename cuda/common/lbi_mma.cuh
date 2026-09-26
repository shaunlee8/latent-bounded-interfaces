// The MMA issue seam of the recurrence and flash-JVP kernels: Hopper wgmma,
// warpgroup-collective with register accumulators.
#pragma once

#include <cstdint>

#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>

namespace lbi {
namespace mma {

// Epilogues read an accumulator tile in windows of ACC_WINDOW registers.
constexpr int ACC_WINDOW = 8;

// A per-warpgroup accumulator tile of NREG fp32 registers.
template <int NREG>
struct Accum {
    static_assert(NREG % 2 == 0, "accumulator tiles are 32-bit register pairs");
    static_assert(NREG % ACC_WINDOW == 0, "tile must divide into windows");
    float v[NREG];

    __device__ inline float& operator[](int i) { return v[i]; }
    __device__ inline const float& operator[](int i) const { return v[i]; }
    __device__ inline uint32_t* raw() { return reinterpret_cast<uint32_t*>(v); }

    // The whole tile is register-resident, so windowing is a no-op.
    __device__ inline void window(int b) { (void)b; }
    __device__ inline void load() {}
};

// Batch control: the warpgroup fence/arrive/commit/wait protocol.
__device__ inline void arrive() {
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
}

__device__ inline void commit() { tl::warpgroup_commit_batch(); }

template <int PENDING>
__device__ inline void wait() { tl::warpgroup_wait<PENDING>(); }

template <int NREG>
__device__ inline void fence_operand(Accum<NREG>& acc) {
    tl::warpgroup_fence_operand(acc.v, NREG);
}

// Completion pipeline over warpgroup batches: wait<P> leaves at most P
// batches pending.
struct Pipeline {
    static constexpr int NBARS = 4;
    __device__ inline void init(uint64_t* smem_bars) { (void)smem_bars; }
    __device__ inline void commit() { tl::warpgroup_commit_batch(); }
    template <int PENDING>
    __device__ inline void wait() { tl::warpgroup_wait<PENDING>(); }
};

// C = A @ B accumulated in `acc`, both operands addressed by shared-memory
// descriptors. Template parameters carry the tile shape and the operand
// transposes; `accumulate` selects accumulate-into versus overwrite.
template <int M, int N, int K, bool TRANS_A, bool TRANS_B,
          int SCALE_A = 1, int SCALE_B = 1>
__device__ inline void mma_ss(uint64_t desc_a, uint64_t desc_b,
                              Accum<(M * N) / 128>& acc, bool accumulate) {
    tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                 tl::DataType::kFloat32, M, N, K, TRANS_A, TRANS_B,
                 SCALE_A, SCALE_B>(desc_a, desc_b, acc.raw(), accumulate);
}

}  // namespace mma
}  // namespace lbi
