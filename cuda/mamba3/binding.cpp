// Torch binding for the hand-CUDA chunk-parallel pass C, the forward
// dual-scan passes, and the lane-batched wgmma dual walk.
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include "chunkparallel_pass_c.hpp"
#include "fwd_dualscan.hpp"

namespace {

using lbi_mamba3::ChunkParallelPassCParams;
using lbi_mamba3::FwdDualscanParams;

#define CHECK_IN(t, d)                                                     \
    TORCH_CHECK((t).is_cuda(), #t " must be CUDA");                        \
    TORCH_CHECK((t).is_contiguous(), #t " must be contiguous");            \
    TORCH_CHECK((t).scalar_type() == (d), #t " has wrong dtype")

void chunkparallel_pass_c_simple(
    torch::Tensor dout, torch::Tensor gate, torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor q_bias, torch::Tensor k_bias,
    torch::Tensor dk, torch::Tensor dv, torch::Tensor states, torch::Tensor dq,
    torch::Tensor dstates_in, torch::Tensor angles, torch::Tensor da_cs,
    torch::Tensor da_cs_rev, torch::Tensor dt, torch::Tensor trap,
    torch::Tensor dfactor, torch::Tensor dgamma_diag, torch::Tensor dangles,
    torch::Tensor d_skip, torch::Tensor dd, torch::Tensor qk_dot,
    torch::Tensor dssda, torch::Tensor dda_cs_rev, torch::Tensor dda_cs,
    torch::Tensor segsum, bool use_mma) {
    const auto bf16 = torch::kBFloat16;
    const auto f32 = torch::kFloat32;
    CHECK_IN(dout, bf16);
    CHECK_IN(gate, bf16);
    CHECK_IN(q, bf16);
    CHECK_IN(k, bf16);
    CHECK_IN(v, bf16);
    CHECK_IN(q_bias, f32);
    CHECK_IN(k_bias, f32);
    CHECK_IN(dk, f32);
    CHECK_IN(dv, bf16);
    CHECK_IN(states, bf16);
    CHECK_IN(dq, f32);
    CHECK_IN(dstates_in, bf16);
    CHECK_IN(angles, f32);
    CHECK_IN(da_cs, f32);
    CHECK_IN(da_cs_rev, f32);
    CHECK_IN(dt, f32);
    CHECK_IN(trap, bf16);
    CHECK_IN(dfactor, f32);
    CHECK_IN(dgamma_diag, f32);
    CHECK_IN(dangles, f32);
    CHECK_IN(d_skip, f32);
    CHECK_IN(dd, f32);
    CHECK_IN(qk_dot, bf16);
    CHECK_IN(dssda, f32);
    CHECK_IN(dda_cs_rev, f32);
    CHECK_IN(dda_cs, f32);
    CHECK_IN(segsum, f32);

    ChunkParallelPassCParams p{};
    p.batch = dout.size(0);
    p.lanes = dout.size(1);
    p.seqlen = dout.size(2);
    p.heads = dout.size(3);
    p.headdim = dout.size(4);
    p.groups = q.size(3);
    p.d_state = q.size(4);
    p.nchunks = segsum.size(2);
    p.chunk_size = segsum.size(3);
    p.n_rot = angles.size(3);
    TORCH_CHECK(p.seqlen == p.nchunks * p.chunk_size, "seqlen/chunk mismatch");
    TORCH_CHECK(dk.size(-1) == p.d_state && dk.dim() == 4, "DK must be [B,LG,S,N]");

    p.DOUT = reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr());
    p.GATE = reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr());
    p.Q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    p.K = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
    p.V = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
    p.Q_BIAS = q_bias.data_ptr<float>();
    p.K_BIAS = k_bias.data_ptr<float>();
    p.STATES = reinterpret_cast<const __nv_bfloat16*>(states.data_ptr());
    p.DSTATES_IN = reinterpret_cast<const __nv_bfloat16*>(dstates_in.data_ptr());
    p.ANGLES = angles.data_ptr<float>();
    p.DA_CS = da_cs.data_ptr<float>();
    p.DA_CS_REV = da_cs_rev.data_ptr<float>();
    p.DT = dt.data_ptr<float>();
    p.TRAP = reinterpret_cast<const __nv_bfloat16*>(trap.data_ptr());
    p.D = d_skip.data_ptr<float>();
    p.QK_DOT = reinterpret_cast<const __nv_bfloat16*>(qk_dot.data_ptr());
    p.SEGSUM = segsum.data_ptr<float>();
    p.DK = dk.data_ptr<float>();
    p.DQ = dq.data_ptr<float>();
    p.DV = reinterpret_cast<__nv_bfloat16*>(dv.data_ptr());
    p.DANGLES = dangles.data_ptr<float>();
    p.DFACTOR = dfactor.data_ptr<float>();
    p.DGAMMA_DIAG = dgamma_diag.data_ptr<float>();
    p.DD = dd.data_ptr<float>();
    p.DSSDA = dssda.data_ptr<float>();
    p.DDA_CS_REV = dda_cs_rev.data_ptr<float>();
    p.DDA_CS = dda_cs.data_ptr<float>();

    if (use_mma) {
        lbi_mamba3::launch_chunkparallel_pass_c_mma(
            p, c10::cuda::getCurrentCUDAStream().stream());
    } else {
        lbi_mamba3::launch_chunkparallel_pass_c_simple(
            p, c10::cuda::getCurrentCUDAStream().stream());
    }
}

const __nv_bfloat16* bfp(const torch::Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}

FwdDualscanParams fwd_params(
    const torch::Tensor& QR, const torch::Tensor& KR, const torch::Tensor& V,
    const torch::Tensor& DQRAW, const torch::Tensor& DKRAW, const torch::Tensor& DV,
    const torch::Tensor& DTHETA, const torch::Tensor& COS, const torch::Tensor& SIN,
    const torch::Tensor& SCALE, const torch::Tensor& DSCALE,
    const torch::Tensor& L, const torch::Tensor& DL, int64_t chunk_size) {
    const auto bf16 = torch::kBFloat16;
    const auto f32 = torch::kFloat32;
    CHECK_IN(QR, bf16); CHECK_IN(KR, bf16); CHECK_IN(V, bf16);
    CHECK_IN(DQRAW, bf16); CHECK_IN(DKRAW, bf16); CHECK_IN(DV, bf16);
    CHECK_IN(DTHETA, bf16); CHECK_IN(COS, bf16); CHECK_IN(SIN, bf16);
    CHECK_IN(SCALE, f32); CHECK_IN(DSCALE, f32);
    CHECK_IN(L, f32); CHECK_IN(DL, f32);
    FwdDualscanParams p{};
    p.batch = (int)QR.size(0);
    p.seqlen = (int)QR.size(1);
    p.heads = (int)QR.size(2);
    p.d_state = (int)QR.size(3);
    p.groups = (int)DQRAW.size(3);
    p.n_rot = (int)COS.size(3);
    p.headdim = (int)V.size(3);
    p.chunk_size = (int)chunk_size;
    p.lanes = (int)DQRAW.size(0);
    p.s_true = p.seqlen;
    TORCH_CHECK(p.seqlen % p.chunk_size == 0, "seqlen must be chunk-padded");
    p.QR = bfp(QR); p.KR = bfp(KR); p.V = bfp(V);
    p.DQRAW = bfp(DQRAW); p.DKRAW = bfp(DKRAW); p.DV = bfp(DV);
    p.DTHETA = bfp(DTHETA); p.COS = bfp(COS); p.SIN = bfp(SIN);
    p.SCALE = SCALE.data_ptr<float>(); p.DSCALE = DSCALE.data_ptr<float>();
    p.L = L.data_ptr<float>(); p.DL = DL.data_ptr<float>();
    return p;
}

void fwd_passA(torch::Tensor QR, torch::Tensor KR, torch::Tensor V,
               torch::Tensor DQRAW, torch::Tensor DKRAW, torch::Tensor DV,
               torch::Tensor DTHETA, torch::Tensor COS, torch::Tensor SIN,
               torch::Tensor SCALE, torch::Tensor DSCALE, torch::Tensor L,
               torch::Tensor DL, torch::Tensor SC, torch::Tensor DSC,
               int64_t chunk_size, bool opt) {
    auto p = fwd_params(QR, KR, V, DQRAW, DKRAW, DV, DTHETA, COS, SIN,
                        SCALE, DSCALE, L, DL, chunk_size);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    if (opt) {
        // The opt pass A emits the bf16 state stream directly.
        TORCH_CHECK(p.chunk_size % 16 == 0 && p.d_state % 16 == 0 &&
                    p.headdim % 16 == 0, "opt kernels need dims % 16");
        CHECK_IN(SC, torch::kBFloat16);
        CHECK_IN(DSC, torch::kBFloat16);
        p.SCB = reinterpret_cast<__nv_bfloat16*>(SC.data_ptr());
        p.DSCB = reinterpret_cast<__nv_bfloat16*>(DSC.data_ptr());
        lbi_mamba3::launch_fwd_passA_primal_opt(p, stream);
        lbi_mamba3::launch_fwd_passA_tan_opt(p, stream);
    } else {
        CHECK_IN(SC, torch::kFloat32);
        CHECK_IN(DSC, torch::kFloat32);
        p.SC = SC.data_ptr<float>();
        p.DSC = DSC.data_ptr<float>();
        lbi_mamba3::launch_fwd_passA_primal(p, stream);
        lbi_mamba3::launch_fwd_passA_tan(p, stream);
    }
}

void fwd_passB(torch::Tensor SC, torch::Tensor DSC, torch::Tensor L,
               torch::Tensor DL, torch::Tensor S_IN, torch::Tensor DS_IN,
               int64_t seqlen, int64_t chunk_size) {
    const auto bf16 = torch::kBFloat16;
    CHECK_IN(SC, bf16); CHECK_IN(DSC, bf16);
    CHECK_IN(S_IN, bf16); CHECK_IN(DS_IN, bf16);
    CHECK_IN(L, torch::kFloat32); CHECK_IN(DL, torch::kFloat32);
    FwdDualscanParams p{};
    p.batch = (int)SC.size(0);
    p.heads = (int)SC.size(1);
    p.d_state = (int)SC.size(3);
    p.headdim = (int)SC.size(4);
    p.lanes = (int)DSC.size(0);
    p.seqlen = (int)seqlen;
    p.chunk_size = (int)chunk_size;
    const long nc = SC.size(2);
    TORCH_CHECK(p.seqlen == nc * p.chunk_size, "seqlen/chunk mismatch");
    TORCH_CHECK(((long)p.d_state * p.headdim) % 64 == 0, "N*P must be % 64");
    p.SCB = reinterpret_cast<__nv_bfloat16*>(SC.data_ptr());
    p.DSCB = reinterpret_cast<__nv_bfloat16*>(DSC.data_ptr());
    p.SB_OUT = reinterpret_cast<__nv_bfloat16*>(S_IN.data_ptr());
    p.DSB_OUT = reinterpret_cast<__nv_bfloat16*>(DS_IN.data_ptr());
    p.L = L.data_ptr<float>();
    p.DL = DL.data_ptr<float>();
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    lbi_mamba3::launch_fwd_passB_opt(p, stream);
}

void fwd_passC(torch::Tensor QR, torch::Tensor KR, torch::Tensor V,
               torch::Tensor DQRAW, torch::Tensor DKRAW, torch::Tensor DV,
               torch::Tensor DTHETA, torch::Tensor COS, torch::Tensor SIN,
               torch::Tensor SCALE, torch::Tensor DSCALE, torch::Tensor L,
               torch::Tensor DL, torch::Tensor S_IN, torch::Tensor DS_IN,
               torch::Tensor OUT, torch::Tensor DOUT, int64_t chunk_size,
               bool opt) {
    auto p = fwd_params(QR, KR, V, DQRAW, DKRAW, DV, DTHETA, COS, SIN,
                        SCALE, DSCALE, L, DL, chunk_size);
    CHECK_IN(OUT, torch::kFloat32); CHECK_IN(DOUT, torch::kFloat32);
    if (opt) {
        CHECK_IN(S_IN, torch::kBFloat16); CHECK_IN(DS_IN, torch::kBFloat16);
        p.S_INB = bfp(S_IN); p.DS_INB = bfp(DS_IN);
    } else {
        CHECK_IN(S_IN, torch::kFloat32); CHECK_IN(DS_IN, torch::kFloat32);
        p.S_IN = S_IN.data_ptr<float>();
        p.DS_IN = DS_IN.data_ptr<float>();
    }
    p.OUT = OUT.data_ptr<float>();
    p.DOUT = DOUT.data_ptr<float>();
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    if (opt) {
        TORCH_CHECK(p.chunk_size % 16 == 0 && p.d_state % 16 == 0 &&
                    p.headdim % 16 == 0 && p.d_state >= 2 * p.headdim &&
                    p.headdim <= p.chunk_size,
                    "opt passC needs dims % 16, N >= 2P, P <= cs");
        lbi_mamba3::launch_fwd_passC_opt(p, stream);
    } else {
        lbi_mamba3::launch_fwd_passC(p, stream);
    }
}

void fwd_passC_mean(torch::Tensor QR, torch::Tensor KR, torch::Tensor V,
                    torch::Tensor DQRAW, torch::Tensor DKRAW, torch::Tensor DV,
                    torch::Tensor DTHETA, torch::Tensor COS, torch::Tensor SIN,
                    torch::Tensor SCALE, torch::Tensor DSCALE, torch::Tensor L,
                    torch::Tensor DL, torch::Tensor Z, torch::Tensor DZ,
                    torch::Tensor QKDOT, torch::Tensor DQKDOT, torch::Tensor DSKIP,
                    torch::Tensor S_IN, torch::Tensor DS_IN, torch::Tensor PART,
                    int64_t chunk_size, int64_t s_true, bool opt) {
    auto p = fwd_params(QR, KR, V, DQRAW, DKRAW, DV, DTHETA, COS, SIN,
                        SCALE, DSCALE, L, DL, chunk_size);
    const auto bf16 = torch::kBFloat16;
    const auto f32 = torch::kFloat32;
    CHECK_IN(Z, bf16); CHECK_IN(DZ, bf16);
    CHECK_IN(QKDOT, f32); CHECK_IN(DQKDOT, f32); CHECK_IN(DSKIP, f32);
    CHECK_IN(PART, f32);
    p.Z = bfp(Z); p.DZ = bfp(DZ);
    p.QKDOT = QKDOT.data_ptr<float>();
    p.DQKDOT = DQKDOT.data_ptr<float>();
    p.DSKIP = DSKIP.data_ptr<float>();
    if (opt) {
        CHECK_IN(S_IN, torch::kBFloat16); CHECK_IN(DS_IN, torch::kBFloat16);
        p.S_INB = bfp(S_IN); p.DS_INB = bfp(DS_IN);
    } else {
        CHECK_IN(S_IN, f32); CHECK_IN(DS_IN, f32);
        p.S_IN = S_IN.data_ptr<float>();
        p.DS_IN = DS_IN.data_ptr<float>();
    }
    p.PART = PART.data_ptr<float>();
    p.s_true = (int)s_true;
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    if (opt) {
        TORCH_CHECK(p.chunk_size % 16 == 0 && p.d_state % 16 == 0 &&
                    p.headdim % 16 == 0 && p.d_state >= 2 * p.headdim &&
                    p.headdim <= p.chunk_size,
                    "opt passC needs dims % 16, N >= 2P, P <= cs");
        TORCH_CHECK(4 * p.chunk_size * (p.d_state - p.headdim) >= 8 * 2048,
                    "opt passC_mean warp scratch needs 4*cs*(N-P) >= 16KB");
        lbi_mamba3::launch_fwd_passC_mean_opt(p, stream);
    } else {
        lbi_mamba3::launch_fwd_passC_mean(p, stream);
    }
}

void fwd_fused_mean(torch::Tensor QR, torch::Tensor KR, torch::Tensor V,
                    torch::Tensor DQRAW, torch::Tensor DKRAW, torch::Tensor DV,
                    torch::Tensor DTHETA, torch::Tensor COS, torch::Tensor SIN,
                    torch::Tensor SCALE, torch::Tensor DSCALE, torch::Tensor L,
                    torch::Tensor DL, torch::Tensor Z, torch::Tensor DZ,
                    torch::Tensor QKDOT, torch::Tensor DQKDOT,
                    torch::Tensor DSKIP, torch::Tensor PART,
                    int64_t chunk_size, int64_t s_true) {
    auto p = fwd_params(QR, KR, V, DQRAW, DKRAW, DV, DTHETA, COS, SIN,
                        SCALE, DSCALE, L, DL, chunk_size);
    const auto bf16 = torch::kBFloat16;
    const auto f32 = torch::kFloat32;
    CHECK_IN(Z, bf16); CHECK_IN(DZ, bf16);
    CHECK_IN(QKDOT, f32); CHECK_IN(DQKDOT, f32); CHECK_IN(DSKIP, f32);
    CHECK_IN(PART, f32);
    p.Z = bfp(Z); p.DZ = bfp(DZ);
    p.QKDOT = QKDOT.data_ptr<float>();
    p.DQKDOT = DQKDOT.data_ptr<float>();
    p.DSKIP = DSKIP.data_ptr<float>();
    p.PART = PART.data_ptr<float>();
    p.s_true = (int)s_true;
    TORCH_CHECK(p.chunk_size % 16 == 0 && p.d_state % 16 == 0 &&
                p.headdim % 16 == 0 && p.headdim <= p.d_state,
                "fused mean needs dims % 16, P <= N");
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    lbi_mamba3::launch_fwd_fused_mean(p, stream);
}

void fwd_r8a_mean(torch::Tensor QR, torch::Tensor KR, torch::Tensor V,
                  torch::Tensor DQRAW, torch::Tensor DKRAW, torch::Tensor DV,
                  torch::Tensor DTHETA, torch::Tensor COS, torch::Tensor SIN,
                  torch::Tensor SCALE, torch::Tensor DSCALE, torch::Tensor L,
                  torch::Tensor DL, torch::Tensor Z, torch::Tensor DZ,
                  torch::Tensor QKDOT, torch::Tensor DQKDOT,
                  torch::Tensor DSKIP, torch::Tensor SOUT, torch::Tensor DSOUT,
                  torch::Tensor FLAGS, torch::Tensor PART,
                  int64_t chunk_size, int64_t s_true) {
    auto p = fwd_params(QR, KR, V, DQRAW, DKRAW, DV, DTHETA, COS, SIN,
                        SCALE, DSCALE, L, DL, chunk_size);
    const auto bf16 = torch::kBFloat16;
    const auto f32 = torch::kFloat32;
    CHECK_IN(Z, bf16); CHECK_IN(DZ, bf16);
    CHECK_IN(QKDOT, f32); CHECK_IN(DQKDOT, f32); CHECK_IN(DSKIP, f32);
    CHECK_IN(SOUT, bf16); CHECK_IN(DSOUT, bf16);
    CHECK_IN(PART, f32);
    TORCH_CHECK(FLAGS.is_cuda() && FLAGS.is_contiguous() &&
                FLAGS.scalar_type() == torch::kInt32,
                "FLAGS must be contiguous CUDA int32");
    p.Z = bfp(Z); p.DZ = bfp(DZ);
    p.QKDOT = QKDOT.data_ptr<float>();
    p.DQKDOT = DQKDOT.data_ptr<float>();
    p.DSKIP = DSKIP.data_ptr<float>();
    p.SB_OUT = reinterpret_cast<__nv_bfloat16*>(SOUT.data_ptr());
    p.DSB_OUT = reinterpret_cast<__nv_bfloat16*>(DSOUT.data_ptr());
    p.FLAGS = FLAGS.data_ptr<int>();
    p.PART = PART.data_ptr<float>();
    p.s_true = (int)s_true;
    TORCH_CHECK(p.chunk_size % 16 == 0 && p.d_state % 16 == 0 &&
                p.headdim % 16 == 0 && p.headdim <= p.d_state,
                "r8a mean needs dims % 16, P <= N");
    TORCH_CHECK((long)p.lanes * p.batch <= 65535, "grid.z limit");
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    lbi_mamba3::launch_fwd_r8a_mean(p, stream);
}

void fwd_fused_mean_native(torch::Tensor QR, torch::Tensor KR, torch::Tensor V,
                           torch::Tensor DQ32, torch::Tensor DK32,
                           torch::Tensor DV32, torch::Tensor DZ32,
                           torch::Tensor DTHETA, torch::Tensor COS,
                           torch::Tensor SIN, torch::Tensor SCALE,
                           torch::Tensor DSCALE, torch::Tensor L,
                           torch::Tensor DL, torch::Tensor Z,
                           torch::Tensor QKDOT, torch::Tensor DQKDOT,
                           torch::Tensor DSKIP, torch::Tensor PART,
                           int64_t chunk_size, int64_t s_true) {
    // native fold: bulk tangents arrive unpadded as strided views (inner
    // dim contiguous, 16B-aligned); pad rows zero-fill on chip.
    const auto bf16 = torch::kBFloat16;
    const auto f32 = torch::kFloat32;
    CHECK_IN(QR, bf16); CHECK_IN(KR, bf16); CHECK_IN(V, bf16);
    CHECK_IN(DTHETA, bf16); CHECK_IN(COS, bf16); CHECK_IN(SIN, bf16);
    CHECK_IN(Z, bf16);
    auto chk_nat = [&](const torch::Tensor& a, const torch::Tensor& b2,
                       const char* nm) {
        TORCH_CHECK(a.is_cuda() && b2.is_cuda(), nm, " must be CUDA");
        TORCH_CHECK(a.scalar_type() == b2.scalar_type() &&
                    (a.scalar_type() == bf16 || a.scalar_type() == f32),
                    nm, " pair must be bf16 or fp32, matching");
        TORCH_CHECK(a.sizes() == b2.sizes() && a.strides() == b2.strides(),
                    nm, " pair must share one layout");
        TORCH_CHECK(a.stride(4) == 1, nm, " inner dim must be contiguous");
        const long es = (a.scalar_type() == bf16) ? 8 : 4;
        for (int d = 0; d < 4; ++d)
            TORCH_CHECK(a.stride(d) % es == 0, nm, " strides must be 16B-aligned");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(b2.data_ptr()) % 16 == 0,
                    nm, " base must be 16B-aligned");
    };
    chk_nat(DQ32, DK32, "dq/dk");
    chk_nat(DV32, DZ32, "dv/dz");
    CHECK_IN(SCALE, f32); CHECK_IN(DSCALE, f32);
    CHECK_IN(L, f32); CHECK_IN(DL, f32);
    CHECK_IN(QKDOT, f32); CHECK_IN(DQKDOT, f32); CHECK_IN(DSKIP, f32);
    CHECK_IN(PART, f32);
    FwdDualscanParams p{};
    p.batch = (int)QR.size(0);
    p.seqlen = (int)QR.size(1);
    p.heads = (int)QR.size(2);
    p.d_state = (int)QR.size(3);
    p.groups = (int)DQ32.size(3);
    p.n_rot = (int)COS.size(3);
    p.headdim = (int)V.size(3);
    p.chunk_size = (int)chunk_size;
    p.lanes = (int)DQ32.size(0);
    p.s_true = (int)s_true;
    TORCH_CHECK(p.seqlen % p.chunk_size == 0, "seqlen must be chunk-padded");
    TORCH_CHECK(p.chunk_size % 16 == 0 && p.d_state % 16 == 0 &&
                p.headdim % 16 == 0 && p.headdim <= p.d_state,
                "fused mean needs dims % 16, P <= N");
    TORCH_CHECK(DQ32.size(2) == s_true && DK32.size(2) == s_true &&
                DV32.size(2) == s_true && DZ32.size(2) == s_true,
                "native tangents must be UNPADDED (seq dim == s_true)");
    TORCH_CHECK(s_true <= p.seqlen && p.seqlen - s_true < p.chunk_size,
                "padded seqlen must cover s_true within one chunk");
    TORCH_CHECK(DQ32.size(4) == p.d_state && DK32.size(4) == p.d_state &&
                DV32.size(3) == p.heads && DV32.size(4) == p.headdim &&
                DZ32.size(3) == p.heads && DZ32.size(4) == p.headdim,
                "native tangent inner dims mismatch");
    p.QR = bfp(QR); p.KR = bfp(KR); p.V = bfp(V);
    p.DTHETA = bfp(DTHETA); p.COS = bfp(COS); p.SIN = bfp(SIN);
    p.Z = bfp(Z);
    p.native = 1;
    p.nat_qk32 = (DQ32.scalar_type() == f32);
    p.nat_vz32 = (DV32.scalar_type() == f32);
    if (p.nat_qk32) {
        p.DQRAW32 = DQ32.data_ptr<float>();
        p.DKRAW32 = DK32.data_ptr<float>();
    } else {
        p.DQRAW = bfp(DQ32); p.DKRAW = bfp(DK32);
    }
    if (p.nat_vz32) {
        p.DV32 = DV32.data_ptr<float>();
        p.DZ32 = DZ32.data_ptr<float>();
    } else {
        p.DV = bfp(DV32); p.DZ = bfp(DZ32);
    }
    p.qk_sl = DQ32.stride(0); p.qk_sb = DQ32.stride(1);
    p.qk_ss = DQ32.stride(2); p.qk_sg = DQ32.stride(3);
    p.vz_sl = DV32.stride(0); p.vz_sb = DV32.stride(1);
    p.vz_ss = DV32.stride(2); p.vz_sh = DV32.stride(3);
    p.SCALE = SCALE.data_ptr<float>(); p.DSCALE = DSCALE.data_ptr<float>();
    p.L = L.data_ptr<float>(); p.DL = DL.data_ptr<float>();
    p.QKDOT = QKDOT.data_ptr<float>();
    p.DQKDOT = DQKDOT.data_ptr<float>();
    p.DSKIP = DSKIP.data_ptr<float>();
    p.PART = PART.data_ptr<float>();
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    lbi_mamba3::launch_fwd_fused_mean(p, stream);
}

}  // namespace

// Lane-batched wgmma dual walk (gwalk_dual.cu)
torch::Tensor gwalk_mean(
    torch::Tensor QR, torch::Tensor KR, torch::Tensor V,
    torch::Tensor DQRAW, torch::Tensor DKRAW, torch::Tensor DV,
    torch::Tensor DTHETA, torch::Tensor COS, torch::Tensor SIN,
    torch::Tensor SCALE, torch::Tensor DSCALE,
    torch::Tensor L, torch::Tensor DL,
    torch::Tensor Z, torch::Tensor DZ,
    torch::Tensor QKD, torch::Tensor DQKD, torch::Tensor DSK,
    int64_t s_true);
std::vector<torch::Tensor> gwalk_full(
    torch::Tensor QR, torch::Tensor KR, torch::Tensor V,
    torch::Tensor DQRAW, torch::Tensor DKRAW, torch::Tensor DV,
    torch::Tensor DTHETA, torch::Tensor COS, torch::Tensor SIN,
    torch::Tensor SCALE, torch::Tensor DSCALE,
    torch::Tensor L, torch::Tensor DL);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gwalk_mean", &gwalk_mean,
          "Lane-batched wgmma dual walk, finalize-folded mean epilogue -> ACC [r,B,H,P]",
          pybind11::arg("QR"), pybind11::arg("KR"), pybind11::arg("V"),
          pybind11::arg("DQRAW"), pybind11::arg("DKRAW"), pybind11::arg("DV"),
          pybind11::arg("DTHETA"), pybind11::arg("COS"), pybind11::arg("SIN"),
          pybind11::arg("SCALE"), pybind11::arg("DSCALE"), pybind11::arg("L"),
          pybind11::arg("DL"), pybind11::arg("Z"), pybind11::arg("DZ"),
          pybind11::arg("QKD"), pybind11::arg("DQKD"), pybind11::arg("DSK"),
          pybind11::arg("s_true"));
    m.def("gwalk_full", &gwalk_full,
          "Lane-batched wgmma dual walk, raw OUT/DOUT fields",
          pybind11::arg("QR"), pybind11::arg("KR"), pybind11::arg("V"),
          pybind11::arg("DQRAW"), pybind11::arg("DKRAW"), pybind11::arg("DV"),
          pybind11::arg("DTHETA"), pybind11::arg("COS"), pybind11::arg("SIN"),
          pybind11::arg("SCALE"), pybind11::arg("DSCALE"), pybind11::arg("L"),
          pybind11::arg("DL"));
    m.def("fwd_passA", &fwd_passA,
          "Forward dual-scan pass A (primal SC + per-lane DSC)",
          pybind11::arg("QR"), pybind11::arg("KR"), pybind11::arg("V"),
          pybind11::arg("DQRAW"), pybind11::arg("DKRAW"), pybind11::arg("DV"),
          pybind11::arg("DTHETA"), pybind11::arg("COS"), pybind11::arg("SIN"),
          pybind11::arg("SCALE"), pybind11::arg("DSCALE"), pybind11::arg("L"),
          pybind11::arg("DL"), pybind11::arg("SC"), pybind11::arg("DSC"),
          pybind11::arg("chunk_size"), pybind11::arg("opt") = false);
    m.def("fwd_passB", &fwd_passB,
          "Forward dual-scan pass B (inter-chunk tri scan, bf16 states)",
          pybind11::arg("SC"), pybind11::arg("DSC"), pybind11::arg("L"),
          pybind11::arg("DL"), pybind11::arg("S_IN"), pybind11::arg("DS_IN"),
          pybind11::arg("seqlen"), pybind11::arg("chunk_size"));
    m.def("fwd_fused_mean", &fwd_fused_mean,
          "Fused A+B+C(mean) chunk walk (states smem-resident)",
          pybind11::arg("QR"), pybind11::arg("KR"), pybind11::arg("V"),
          pybind11::arg("DQRAW"), pybind11::arg("DKRAW"), pybind11::arg("DV"),
          pybind11::arg("DTHETA"), pybind11::arg("COS"), pybind11::arg("SIN"),
          pybind11::arg("SCALE"), pybind11::arg("DSCALE"), pybind11::arg("L"),
          pybind11::arg("DL"), pybind11::arg("Z"), pybind11::arg("DZ"),
          pybind11::arg("QKDOT"), pybind11::arg("DQKDOT"),
          pybind11::arg("DSKIP"), pybind11::arg("PART"),
          pybind11::arg("chunk_size"), pybind11::arg("s_true"));
    m.def("fwd_r8a_mean", &fwd_r8a_mean,
          "Chunk-parallel ticket-chained fused walk (mean epilogue)",
          pybind11::arg("QR"), pybind11::arg("KR"), pybind11::arg("V"),
          pybind11::arg("DQRAW"), pybind11::arg("DKRAW"), pybind11::arg("DV"),
          pybind11::arg("DTHETA"), pybind11::arg("COS"), pybind11::arg("SIN"),
          pybind11::arg("SCALE"), pybind11::arg("DSCALE"), pybind11::arg("L"),
          pybind11::arg("DL"), pybind11::arg("Z"), pybind11::arg("DZ"),
          pybind11::arg("QKDOT"), pybind11::arg("DQKDOT"),
          pybind11::arg("DSKIP"), pybind11::arg("SOUT"), pybind11::arg("DSOUT"),
          pybind11::arg("FLAGS"), pybind11::arg("PART"),
          pybind11::arg("chunk_size"), pybind11::arg("s_true"));
    m.def("fwd_fused_mean_native", &fwd_fused_mean_native,
          "Fused walk, native-layout fold: bulk tangents fp32/unpadded "
          "preprocess-native (no emission pass)",
          pybind11::arg("QR"), pybind11::arg("KR"), pybind11::arg("V"),
          pybind11::arg("DQ32"), pybind11::arg("DK32"), pybind11::arg("DV32"),
          pybind11::arg("DZ32"), pybind11::arg("DTHETA"), pybind11::arg("COS"),
          pybind11::arg("SIN"), pybind11::arg("SCALE"), pybind11::arg("DSCALE"),
          pybind11::arg("L"), pybind11::arg("DL"), pybind11::arg("Z"),
          pybind11::arg("QKDOT"), pybind11::arg("DQKDOT"),
          pybind11::arg("DSKIP"), pybind11::arg("PART"),
          pybind11::arg("chunk_size"), pybind11::arg("s_true"));
    m.def("fwd_passC", &fwd_passC,
          "Forward dual-scan pass C (full OUT/DOUT)",
          pybind11::arg("QR"), pybind11::arg("KR"), pybind11::arg("V"),
          pybind11::arg("DQRAW"), pybind11::arg("DKRAW"), pybind11::arg("DV"),
          pybind11::arg("DTHETA"), pybind11::arg("COS"), pybind11::arg("SIN"),
          pybind11::arg("SCALE"), pybind11::arg("DSCALE"), pybind11::arg("L"),
          pybind11::arg("DL"), pybind11::arg("S_IN"), pybind11::arg("DS_IN"),
          pybind11::arg("OUT"), pybind11::arg("DOUT"),
          pybind11::arg("chunk_size"), pybind11::arg("opt") = false);
    m.def("fwd_passC_mean", &fwd_passC_mean,
          "Forward dual-scan pass C with fused mean epilogue (PART)",
          pybind11::arg("QR"), pybind11::arg("KR"), pybind11::arg("V"),
          pybind11::arg("DQRAW"), pybind11::arg("DKRAW"), pybind11::arg("DV"),
          pybind11::arg("DTHETA"), pybind11::arg("COS"), pybind11::arg("SIN"),
          pybind11::arg("SCALE"), pybind11::arg("DSCALE"), pybind11::arg("L"),
          pybind11::arg("DL"), pybind11::arg("Z"), pybind11::arg("DZ"),
          pybind11::arg("QKDOT"), pybind11::arg("DQKDOT"), pybind11::arg("DSKIP"),
          pybind11::arg("S_IN"), pybind11::arg("DS_IN"), pybind11::arg("PART"),
          pybind11::arg("chunk_size"), pybind11::arg("s_true"),
          pybind11::arg("opt") = false);
    m.def("chunkparallel_pass_c", &chunkparallel_pass_c_simple,
          "Chunk-parallel P-batched SISO backward pass C (use_mma selects tensor cores)",
          pybind11::arg("dout"), pybind11::arg("gate"), pybind11::arg("q"),
          pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("q_bias"),
          pybind11::arg("k_bias"), pybind11::arg("dk"), pybind11::arg("dv"),
          pybind11::arg("states"), pybind11::arg("dq"), pybind11::arg("dstates_in"),
          pybind11::arg("angles"), pybind11::arg("da_cs"), pybind11::arg("da_cs_rev"),
          pybind11::arg("dt"), pybind11::arg("trap"), pybind11::arg("dfactor"),
          pybind11::arg("dgamma_diag"), pybind11::arg("dangles"), pybind11::arg("d_skip"),
          pybind11::arg("dd"), pybind11::arg("qk_dot"), pybind11::arg("dssda"),
          pybind11::arg("dda_cs_rev"), pybind11::arg("dda_cs"), pybind11::arg("segsum"),
          pybind11::arg("use_mma") = false);
}
