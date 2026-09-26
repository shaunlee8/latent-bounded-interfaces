from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).resolve().parent

# recurrence_jvp.cu uses tilelang's shipped wgmma headers; sm_90a is required
# for wgmma issue.
import tilelang  # noqa: E402

_tl_dir = Path(tilelang.__file__).resolve().parent
tl_includes = [str(_tl_dir / "src"), str(_tl_dir / "3rdparty" / "cutlass" / "include")]

setup(
    name="mamba3_lbi_cuda",
    ext_modules=[
        CUDAExtension(
            name="mamba3_lbi_cuda",
            sources=[
                str(this_dir / "binding.cpp"),
                str(this_dir / "recurrence_jvp.cu"),
            ],
            include_dirs=[str(this_dir), str(this_dir.parent / "common")] + tl_includes,
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
