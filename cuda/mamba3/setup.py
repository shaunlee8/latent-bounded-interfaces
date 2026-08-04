from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).resolve().parent

# gwalk_dual.cu uses tilelang's shipped wgmma_ss
# descriptor discipline headers; sm_90a is required for wgmma issue.
import tilelang  # noqa: E402  (build-time dependency, already a project dep)

_tl_dir = Path(tilelang.__file__).resolve().parent
tl_includes = [str(_tl_dir / "src"), str(_tl_dir / "3rdparty" / "cutlass" / "include")]

setup(
    name="mamba3_lbi_cuda",
    ext_modules=[
        CUDAExtension(
            name="mamba3_lbi_cuda",
            sources=[
                str(this_dir / "binding.cpp"),
                str(this_dir / "chunkparallel_pass_c_simple.cu"),
                str(this_dir / "chunkparallel_pass_c_mma.cu"),
                str(this_dir / "fwd_dualscan_simple.cu"),
                str(this_dir / "fwd_dualscan_opt.cu"),
                str(this_dir / "gwalk_dual.cu"),
            ],
            include_dirs=[str(this_dir)] + tl_includes,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3", "--use_fast_math", "-lineinfo", "-arch=sm_90a",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
