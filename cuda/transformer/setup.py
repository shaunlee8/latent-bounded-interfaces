import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).resolve().parent
repo_root = this_dir.parents[1]
cutlass_root = Path(os.environ.get("LBI_CUTLASS_ROOT", repo_root / "third_party" / "cutlass")).expanduser().resolve()
cutlass_include = cutlass_root / "include"
cutlass_util_include = cutlass_root / "tools" / "util" / "include"

cutlass_required = [
    cutlass_include / "cutlass" / "cutlass.h",
    cutlass_include / "cute" / "tensor.hpp",
]
missing_cutlass = [str(path) for path in cutlass_required if not path.exists()]
if missing_cutlass:
    raise RuntimeError(
        "Transformer CUDA kernels require CUTLASS/CuTe. "
        "Run `git submodule update --init --recursive third_party/cutlass` or set LBI_CUTLASS_ROOT. Missing: "
        + ", ".join(missing_cutlass)
    )

include_dirs = [str(this_dir), str(cutlass_include), str(cutlass_util_include)]
sources = [
    str(this_dir / "binding.cpp"),
    str(this_dir / "attention_pullback_simple.cu"),
    str(this_dir / "attention_pullback_mma.cu"),
]
nvcc_args = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]
cxx_args = ["-O3"]

setup(
    name="transformer_lbi_cuda",
    ext_modules=[
        CUDAExtension(
            name="transformer_lbi_cuda",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
