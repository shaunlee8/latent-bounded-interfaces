#!/usr/bin/env bash
# Prewarm the JIT build of the transformer flash-JVP extension (the kernel
# otherwise builds on first use). Requires ninja on PATH and a Hopper toolchain.
set -euo pipefail
cd "$(dirname "$0")/../.."
${PYTHON_BIN:-python} -c "from cuda.transformer import _module; _module(); print('lbi_transformer_flash_jvp build OK')"
