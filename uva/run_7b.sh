#!/usr/bin/env bash
# One command on the PORTAL: CPU checks first, then submit the 7B GPU jobs.
#
#     cd /bigtemp/$USER/OLMo-core && bash uva/run_7b.sh
#
#   1. uva/local_test.sh: tiny-model CPU test of the whole pipeline (~3-5 min)
#   2. checks on the REAL inputs: converted 7B checkpoint exists, real data packs on CPU
#   3. only if 1+2 pass: submit 7B fit check (5 steps) -> 7B SFT (30 steps, checkpoint every 15)
#
# Options (env vars):
#   GPU_ARGS="--gres=gpu:4 --constraint=a100_80gb"   default: 2x H100 NVL
#   SKIP_LOCAL_TEST=1                                 skip step 1
#   SFT_STEPS=30  CKPT_EVERY=15                       for the SFT job
set -euo pipefail
cd "$(dirname "$0")/.."
source uva/env.sh

GPU="${GPU_ARGS:---gres=gpu:2 --constraint=h100_94gb}"
MAIL="--mail-type=END,FAIL --mail-user=$USER@virginia.edu"
PREP_LOG="$RUNS_DIR/prep_data.portal.log"

echo "== 1/3 CPU pipeline test (tiny model)"
if [ "${SKIP_LOCAL_TEST:-0}" != 1 ]; then
    bash uva/local_test.sh || { echo ">>> local test FAILED: nothing submitted"; exit 1; }
fi

echo "== 2/3 checks on the real 7B inputs (CPU)"
[ -f "$OLMO_CKPT/model_and_optim/.metadata" ] \
    || { echo ">>> no converted 7B checkpoint: run  sbatch uva/00_prepare.sbatch  first"; exit 1; }
echo "   7B checkpoint: OK ($OLMO_CKPT)"
if python uva/sft_olmo2_7b_uva.py prep_data --checkpoint=none --dataset_path="$SFT_DATA" \
    --save_folder="$RUNS_DIR/prep" --data_work_dir="$OLMO_ROOT/dataset-cache" --seq_len=4096 \
    >"$PREP_LOG" 2>&1; then
    echo "   real data: $(grep -o 'Dataset ready.*' "$PREP_LOG")"
else
    tail -n 40 "$PREP_LOG"
    echo ">>> packing the real data FAILED (full log: $PREP_LOG): nothing submitted"
    exit 1
fi

echo "== 3/3 submitting GPU jobs ($GPU)"
# shellcheck disable=SC2086
fit=$(sbatch --parsable $MAIL $GPU --time=00:45:00 \
    --export=ALL,STEPS=5,RUN_NAME=fit-check uva/01_sft.sbatch)
# shellcheck disable=SC2086
sft=$(sbatch --parsable $MAIL $GPU --dependency=afterok:"$fit" \
    --export=ALL,STEPS="${SFT_STEPS:-30}",CKPT_EVERY="${CKPT_EVERY:-15}",RUN_NAME=sft-30step uva/01_sft.sbatch)
echo ">>> fit check: $fit   SFT (starts only if fit check succeeds): $sft"
echo ">>> logs: uva/logs/olmo2-7b-sft-$fit.out  uva/logs/olmo2-7b-sft-$sft.out"
squeue -u "$USER"
