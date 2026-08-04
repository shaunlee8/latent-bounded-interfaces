#!/usr/bin/env bash
# LBI paper training launcher; all knobs are environment overrides and the
# shared body lives in paper_train_common.sh.
set -euo pipefail
VARIANT=lbi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paper_train_common.sh"
