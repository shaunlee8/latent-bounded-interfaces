from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).resolve().parent

setup(
    name="interface_scan_cuda",
    ext_modules=[
        CUDAExtension(
            name="interface_scan_cuda",
            sources=[
                str(this_dir / "binding.cpp"),
                str(this_dir / "suffix_scan.cu"),
            ],
            include_dirs=[str(this_dir)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
