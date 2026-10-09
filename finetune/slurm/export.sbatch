#!/usr/bin/env bash
# CPU job: convert the run's latest checkpoint to Hugging Face format
# ($FT_RUN_DIR/hf-stepN: bf16 safetensors + tokenizer + chat template; ~15 GB for 7B).
#     sbatch finetune/slurm/export.sbatch CONFIG.yaml [--step N]
#SBATCH --job-name=ft-export
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --time=02:00:00
#SBATCH --output=ft-export-%j.out
set -euo pipefail
CONFIG="${1:?usage: sbatch finetune/slurm/export.sbatch CONFIG.yaml [--step N]}"
shift
source finetune/scripts/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} config=$CONFIG start=$(date)"
python "$FT/ft.py" export "$CONFIG" "$@"
echo "EXPORT OK  done=$(date)"
