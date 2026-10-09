#!/usr/bin/env bash
# CPU job: base model -> OLMo-core format, tokenize + label-mask the data, pack it.
# Each step is skipped if already done, so resubmitting is cheap. Submitted by scripts/submit.sh:
#     sbatch [--output=...] finetune/slurm/prepare.sbatch finetune/configs/<run>.yaml
#SBATCH --job-name=ft-prepare
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192000
#SBATCH --time=08:00:00
#SBATCH --output=ft-prepare-%j.out
set -euo pipefail
CONFIG="${1:?usage: sbatch finetune/slurm/prepare.sbatch CONFIG.yaml}"
source finetune/scripts/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} config=$CONFIG start=$(date)"
ft_load "$CONFIG"

python "$FT/ft.py" prepare "$CONFIG"
python "$FT/ft.py" validate "$CONFIG"
du -sh "$FT_MODEL_DIR" "$FT_DATA_DIR" 2>/dev/null || true
echo "PREPARE OK  done=$(date)"
