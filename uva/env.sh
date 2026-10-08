# Shared settings for the UVA CS cluster scripts. Source this, don't run it.
# Everything big lives on /bigtemp (20 TB/user scratch, NOT backed up, purged after 150 days idle).
# Home dirs are ~100 GB and should not hold models/datasets.

export OLMO_ROOT="${OLMO_ROOT:-/bigtemp/$USER/olmo}"
export OLMO_REPO="${OLMO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export OLMO_VENV="$OLMO_ROOT/venv"

export HF_HOME="$OLMO_ROOT/hf_cache"
export UV_CACHE_DIR="$OLMO_ROOT/uv_cache"
export PIP_CACHE_DIR="$OLMO_ROOT/pip_cache"
export TRITON_CACHE_DIR="$OLMO_ROOT/triton_cache"

# Base model + data used for the feasibility run
export HF_BASE_MODEL="${HF_BASE_MODEL:-allenai/OLMo-2-1124-7B}"
export HF_BASE_LOCAL="$OLMO_ROOT/hf/OLMo-2-1124-7B"
export OLMO_CKPT="$OLMO_ROOT/ckpt/OLMo-2-1124-7B"          # converted OLMo-core checkpoint
export SFT_DATA="$OLMO_ROOT/data/tulu3-sample"              # tokenized SFT data
export RUNS_DIR="$OLMO_ROOT/runs"

export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OLMO_ROOT"/{hf,ckpt,data,runs,hf_cache} 2>/dev/null || true

if [ -f "$OLMO_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$OLMO_VENV/bin/activate"
fi
