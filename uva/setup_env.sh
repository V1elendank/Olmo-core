#!/usr/bin/env bash
# One-time environment setup. Run on portal.cs.virginia.edu from the repo root:
#     bash uva/setup_env.sh
# Uses uv (installed to ~/.local/bin if missing) and puts the venv + caches on /bigtemp.
#
# TORCH_CUDA picks the PyTorch wheel flavor. The default PyPI torch wheel targets CUDA 13
# (needs NVIDIA driver >= 580), so we use cu128 wheels, which run on any driver >= 525 via CUDA
# minor-version compatibility. 01_sft.sbatch prints the driver version and fails fast if CUDA
# isn't usable; if so, rerun with TORCH_CUDA=cu126.
set -euo pipefail
TORCH_CUDA="${TORCH_CUDA:-cu128}"

source "$(dirname "$0")/env.sh"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$OLMO_VENV" ]; then
    uv venv --python 3.12 "$OLMO_VENV"
fi
# shellcheck disable=SC1091
source "$OLMO_VENV/bin/activate"

echo ">>> Installing torch ($TORCH_CUDA wheels)"
uv pip install "torch>=2.10.0,<2.14" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

echo ">>> Installing OLMo-core (editable) + data/conversion deps"
uv pip install -e "$OLMO_REPO[transformers]" datasets "huggingface_hub[cli]" jinja2

python - <<'EOF'
import torch, olmo_core, transformers
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("olmo_core", olmo_core.__version__, "| transformers", transformers.__version__)
EOF
echo ">>> Done. venv: $OLMO_VENV"
