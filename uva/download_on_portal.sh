#!/usr/bin/env bash
# Fallback if compute nodes can't reach the internet (preflight T6 = WARN).
# Runs the two download steps of 00_prepare.sbatch on the portal; the sbatch job then only
# converts the checkpoint (it skips files that already exist). From the repo root:
#     bash uva/download_on_portal.sh      # ~20-40 min; keep the SSH session open (or use tmux)
set -euo pipefail
source "$(dirname "$0")/env.sh"

hf download "$HF_BASE_MODEL" --local-dir "$HF_BASE_LOCAL"
if [ ! -f "$SFT_DATA/token_ids_part_0000.npy" ]; then
    python uva/prep_sft_data.py --out "$SFT_DATA" --num_examples "${NUM_EXAMPLES:-5000}"
fi
echo ">>> downloads done; now: bash uva/launch_from_mac.sh <id> full   (or sbatch the jobs by hand)"
