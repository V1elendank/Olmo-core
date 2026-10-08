#!/usr/bin/env bash
# Step 2 (GPU): short full-parameter SFT of OLMo-2 7B with FSDP on one node.
# Default: 4x A100 80GB. Submit from the repo root, e.g.
#     sbatch uva/01_sft.sbatch
#     sbatch --gres=gpu:2 --constraint=h100_94gb uva/01_sft.sbatch          # 2x H100 NVL 94GB
#
# Env knobs (pass with --export=ALL,KEY=VAL,...):
#   STEPS=30          optimizer steps
#   RUN_NAME=...      run folder under $RUNS_DIR (re-using a name RESUMES from its checkpoint)
#   CKPT_EVERY=10     overwritten checkpoint every N steps + final checkpoint (0 = off)
#   STALL_MIN=12      watchdog: kill the run if the log is silent this long
#   EXTRA="--full_ac" extra args for uva/sft_olmo2_7b_uva.py
#
# Safety: everything that can fail without a GPU is checked first (checkpoint, data, CUDA,
# data packing), and the training run is wrapped in a watchdog, so a broken or hung job ends
# within minutes instead of holding the GPUs until the time limit.
#SBATCH --job-name=olmo2-7b-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100_80gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240000
#SBATCH --time=04:00:00
#SBATCH --output=uva/logs/%x-%j.out
set -euo pipefail
source uva/env.sh

NGPU=$(nvidia-smi -L | wc -l)
STEPS="${STEPS:-30}"
RUN_NAME="${RUN_NAME:-olmo2-7b-sft-${SLURM_JOB_ID}}"
CKPT_EVERY="${CKPT_EVERY:-0}"
SAVE="$RUNS_DIR/$RUN_NAME"

echo "host=$(hostname) gpus=$NGPU job=$SLURM_JOB_ID run=$RUN_NAME steps=$STEPS ckpt_every=$CKPT_EVERY start=$(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

fail() {
    echo "PRECHECK FAILED: $*  (no GPU time used for training)"
    exit 2
}

echo "== prechecks"
[ -f "$OLMO_CKPT/model_and_optim/.metadata" ] || fail "no converted checkpoint at $OLMO_CKPT/model_and_optim"
ls "$SFT_DATA"/token_ids_part_*.npy >/dev/null 2>&1 || fail "no tokenized data in $SFT_DATA"
python -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == $NGPU; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count())" \
    || fail "CUDA not usable (driver vs TORCH_CUDA mismatch?)"

CKPT_ARGS=()
if [ "$CKPT_EVERY" -gt 0 ]; then
    CKPT_ARGS=(--ephemeral_interval="$CKPT_EVERY" --save_checkpoints)
    [ -d "$SAVE" ] && ls -d "$SAVE"/step* >/dev/null 2>&1 && echo "   found checkpoints in $SAVE: run will RESUME"
fi

COMMON=(--run_name="$RUN_NAME" --checkpoint="$OLMO_CKPT/model_and_optim" --dataset_path="$SFT_DATA"
    --save_folder="$SAVE" --data_work_dir="$OLMO_ROOT/dataset-cache" --seq_len=4096)

# Pack the SFT data into 4096-token instances in ONE process (cached; instant if the prepare job did it).
echo "== data packing (single process)"
python uva/sft_olmo2_7b_uva.py prep_data "${COMMON[@]}" >"$SAVE.prep_data.log" 2>&1 \
    || { tail -n 40 "$SAVE.prep_data.log"; fail "data packing failed (log: $SAVE.prep_data.log)"; }
grep "Dataset ready" "$SAVE.prep_data.log"

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

echo "== training (watchdog: ${STALL_MIN:-12} min without log output or checkpoint writes -> kill)"
export WATCH_DIR="$SAVE" # checkpoint saves (~88 GB, silent for minutes) count as progress
# shellcheck disable=SC2086
bash uva/run_with_watchdog.sh "$SAVE.train.log" \
    torchrun --standalone --nproc-per-node="$NGPU" uva/sft_olmo2_7b_uva.py train \
    "${COMMON[@]}" \
    --global_batch_seqs=32 \
    --microbatch_seqs=1 \
    --steps="$STEPS" \
    --dist_timeout_min=10 \
    ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} \
    ${EXTRA:-}

echo "done=$(date)"
