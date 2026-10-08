#!/usr/bin/env bash
# Step 1 (CPU only): download OLMo-2 7B from Hugging Face, convert it to an OLMo-core
# checkpoint, and tokenize a small SFT dataset.  Submit from the repo root:
#     sbatch uva/00_prepare.sbatch
#SBATCH --job-name=olmo-prep
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192000
#SBATCH --time=06:00:00
#SBATCH --output=uva/logs/%x-%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<computing-id>@virginia.edu
set -euo pipefail
source uva/env.sh
echo "host=$(hostname) start=$(date)"

# 1. Download the HF base model (~29 GB). Skips files already present.
if [ ! -f "$HF_BASE_LOCAL/config.json" ] || ! ls "$HF_BASE_LOCAL"/*.safetensors >/dev/null 2>&1; then
    hf download "$HF_BASE_MODEL" --local-dir "$HF_BASE_LOCAL"
fi

# 2. Convert HF -> OLMo-core (writes $OLMO_CKPT/model_and_optim). Validation compares
#    logits of the two models on CPU; add --skip-validation to save ~20 min.
if [ ! -d "$OLMO_CKPT/model_and_optim" ]; then
    python src/examples/huggingface/convert_checkpoint_from_hf.py \
        -i "$HF_BASE_LOCAL" -m olmo2_7b -t dolma2 -o "$OLMO_CKPT"
fi

# 3. Tokenize a small SFT sample (Tulu 3 SFT mixture, 5k conversations).
if [ ! -f "$SFT_DATA/token_ids_part_0000.npy" ]; then
    python uva/prep_sft_data.py --out "$SFT_DATA" --num_examples "${NUM_EXAMPLES:-5000}"
fi

ls -la "$OLMO_CKPT" "$SFT_DATA"
echo "done=$(date)"
