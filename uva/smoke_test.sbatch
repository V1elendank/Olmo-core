#!/usr/bin/env bash
# GPU smoke test (~5 min of GPU time): checks the driver, CUDA, NCCL across 2 GPUs, FSDP and
# the OLMo-core training loop, using a tiny random-init 190M OLMo-2 model on 50 tokenized
# conversations from preflight.sh. It does NOT download or load the 7B model. From the repo root:
#     sbatch uva/smoke_test.sbatch
#SBATCH --job-name=olmo-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100_80gb|a100_40gb|h100_94gb|a40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=00:30:00
#SBATCH --output=uva/logs/%x-%j.out
set -euo pipefail
source uva/env.sh
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"

echo "== S1  driver / GPUs"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== S2  torch sees CUDA, bf16 matmul on every GPU"
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available: rebuild venv with TORCH_CUDA=cu126"
n = torch.cuda.device_count()
for i in range(n):
    a = torch.randn(4096, 4096, device=f"cuda:{i}", dtype=torch.bfloat16)
    (a @ a).sum().item()
    print(f"   cuda:{i} {torch.cuda.get_device_name(i)} OK")
print(f"   torch {torch.__version__} (CUDA {torch.version.cuda}), {n} GPUs")
EOF

echo "== S3  tiny OLMo-core FSDP training run (190M, random init, 5 steps)"
NGPU=$(nvidia-smi -L | wc -l)
torchrun --standalone --nproc-per-node="$NGPU" uva/sft_olmo2_7b_uva.py train \
    --model_size=190M \
    --checkpoint=none \
    --dataset_path="$OLMO_ROOT/data/smoke" \
    --save_folder="$RUNS_DIR/smoke-$SLURM_JOB_ID" \
    --seq_len=1024 \
    --global_batch_seqs=4 \
    --microbatch_seqs=1 \
    --steps=5

echo "SMOKE TEST PASSED  $(date)"
