#!/usr/bin/env bash
# GPU job: full-parameter fine-tuning with FSDP on one node. Submitted by scripts/submit.sh, which
# sets --gres/--constraint/--time/--mem from the YAML's slurm: section, e.g.
#     sbatch --gres=gpu:2 --constraint=h100_94gb finetune/slurm/train.sbatch CONFIG.yaml [--fit-check]
#
# Safety (all lessons from the uva/ feasibility runs):
#   * prechecks (model, data, CUDA) fail in seconds, before torchrun starts
#   * data packing runs once in a single process (not in every rank)
#   * watchdog kills the job if neither the log nor the checkpoint folder changes for slurm.stall_min
#   * re-submitting the same config RESUMES from its latest checkpoint
#SBATCH --job-name=ft-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100_94gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240000
#SBATCH --time=04:00:00
#SBATCH --output=ft-train-%j.out
set -euo pipefail
CONFIG="${1:?usage: sbatch finetune/slurm/train.sbatch CONFIG.yaml [--fit-check] [extra ft.py train args]}"
shift
source finetune/scripts/env.sh
ft_load "$CONFIG"

MODE=train
RUN_DIR="$FT_RUN_DIR"
for a in "$@"; do [ "$a" = --fit-check ] && MODE=fit-check && RUN_DIR="$FT_RUN_DIR-fitcheck"; done
NGPU=$(nvidia-smi -L | wc -l)
echo "host=$(hostname) job=$SLURM_JOB_ID mode=$MODE gpus=$NGPU config=$CONFIG run=$RUN_DIR start=$(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

fail() { echo "PRECHECK FAILED: $*  (no GPU time used for training)"; exit 2; }
echo "== prechecks"
[ "$NGPU" = "$FT_GPUS" ] || fail "job has $NGPU GPUs but the config says slurm.gpus=$FT_GPUS"
[ -f "$FT_MODEL_DIR/model_and_optim/.metadata" ] || fail "base model not prepared: $FT_MODEL_DIR"
[ -f "$FT_DATA_DIR/manifest.json" ] || fail "data not prepared: $FT_DATA_DIR"
python -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == $NGPU; print('   torch', torch.__version__, 'cuda', torch.version.cuda)" \
    || fail "CUDA not usable (driver vs torch wheel mismatch?)"
if [ ! -f "$FT_DATA_DIR/pack.json" ]; then
    echo "== packing data (single process)"
    python "$FT/ft.py" pack "$CONFIG" || fail "data packing failed"
fi
python - "$FT_DATA_DIR/pack.json" <<'PY'
import json, sys; p = json.load(open(sys.argv[1]))
print(f"   data: {p['instances']:,} sequences of {p['seq_len']} tokens = {p['steps_per_epoch']} steps/epoch")
PY
if [ "$MODE" = train ] && ls -d "$RUN_DIR"/step* >/dev/null 2>&1; then
    echo "   found checkpoints in $RUN_DIR: the run will RESUME"
fi

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/train-$SLURM_JOB_ID.log"
echo "== training (log: $LOG; watchdog ${FT_STALL_MIN} min)"
export WATCH_DIR="$RUN_DIR" STALL_MIN="$FT_STALL_MIN"   # checkpoint writes count as progress
bash "$FT/scripts/run_with_watchdog.sh" "$LOG" \
    torchrun --standalone --nproc-per-node="$NGPU" "$FT/ft.py" train "$CONFIG" "$@"
echo "TRAIN OK ($MODE)  done=$(date)"
