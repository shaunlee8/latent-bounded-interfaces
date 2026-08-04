#!/usr/bin/env bash
# Dense-baseline paper training launcher; all knobs are environment overrides
# and the shared body lives in paper_train_common.sh.
set -euo pipefail
VARIANT=dense
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paper_train_common.sh"
