#!/usr/bin/env bash
# Step 2 (GPU): short full-parameter SFT of OLMo-2 7B with FSDP on one node.
# Default: 4x A100 80GB (cheetah04, NVLink). Submit from the repo root:
#     sbatch uva/01_sft.sbatch
# Other GPU shapes (command-line flags override the #SBATCH lines below):
#     sbatch --gres=gpu:2 --constraint=h100_94gb uva/01_sft.sbatch          # 2x H100 NVL 94GB
#     sbatch --gres=gpu:4 --constraint=a100_40gb --export=ALL,EXTRA="--full_ac" uva/01_sft.sbatch
#     sbatch --gres=gpu:4 --constraint=a40       --export=ALL,EXTRA="--full_ac" uva/01_sft.sbatch
#SBATCH --job-name=olmo2-7b-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100_80gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240000
#SBATCH --time=04:00:00
#SBATCH --output=uva/logs/%x-%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<computing-id>@virginia.edu
set -euo pipefail
source uva/env.sh

NGPU=$(nvidia-smi -L | wc -l)
STEPS="${STEPS:-30}"
RUN_NAME="${RUN_NAME:-olmo2-7b-sft-${SLURM_JOB_ID}}"

echo "host=$(hostname) gpus=$NGPU job=$SLURM_JOB_ID start=$(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available: check TORCH_CUDA vs driver'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count())"

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

COMMON=(--run_name="$RUN_NAME" --checkpoint="$OLMO_CKPT/model_and_optim" --dataset_path="$SFT_DATA"
        --save_folder="$RUNS_DIR/$RUN_NAME" --seq_len=4096)

# Pack the SFT data into 4096-token instances in ONE process first (cached in save_folder).
# If this fails, the job stops here in seconds instead of hanging the GPU ranks at a barrier.
echo "== packing data (single process)"
python uva/sft_olmo2_7b_uva.py prep_data "${COMMON[@]}"

# shellcheck disable=SC2086
torchrun --standalone --nproc-per-node="$NGPU" uva/sft_olmo2_7b_uva.py train \
    --run_name="$RUN_NAME" \
    --checkpoint="$OLMO_CKPT/model_and_optim" \
    --dataset_path="$SFT_DATA" \
    --save_folder="$RUNS_DIR/$RUN_NAME" \
    --seq_len=4096 \
    --global_batch_seqs=32 \
    --microbatch_seqs=1 \
    --steps="$STEPS" \
    ${EXTRA:-}

echo "done=$(date)"
