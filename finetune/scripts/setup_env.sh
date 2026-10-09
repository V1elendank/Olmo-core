#!/usr/bin/env bash
# One-time environment setup on portal.cs.virginia.edu (from the repo root):
#     bash finetune/scripts/setup_env.sh            # REINSTALL=1 to rebuild the venv
# Creates the venv on /bigtemp with uv. torch uses CUDA 12.8 wheels (work with the cluster's
# 580 driver); if a GPU job reports CUDA unusable, rerun with TORCH_CUDA=cu126.
# The venv from the uva/ feasibility test is the same one: nothing to do if it exists.
set -euo pipefail
TORCH_CUDA="${TORCH_CUDA:-cu128}"
source "$(dirname "$0")/env.sh"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if [ "${REINSTALL:-0}" = 1 ]; then rm -rf "$OLMO_VENV"; fi
if [ ! -d "$OLMO_VENV" ]; then
    uv venv --python 3.12 "$OLMO_VENV"
fi
# shellcheck disable=SC1091
source "$OLMO_VENV/bin/activate"

echo ">>> torch ($TORCH_CUDA wheels)"
uv pip install "torch>=2.10.0,<2.14" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
echo ">>> OLMo-core (editable) + data / conversion deps"
uv pip install -e "$FT_REPO[transformers]" datasets huggingface_hub jinja2 pyyaml pandas pyarrow

python - <<'PY'
import torch, transformers, datasets, yaml
from olmo_core.version import VERSION
print("torch", torch.__version__, "cuda", torch.version.cuda, "| olmo_core", VERSION,
      "| transformers", transformers.__version__, "| datasets", datasets.__version__)
PY
echo ">>> done: $OLMO_VENV"
