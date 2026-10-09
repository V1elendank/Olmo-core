# Shared settings for every finetune/ script. Source it, don't run it:   source finetune/scripts/env.sh
# Big files live on /bigtemp (20 TB scratch per user, NOT backed up, purged after 150 days unused):
#   $OLMO_ROOT/models/   prepared base models        $OLMO_ROOT/data/   tokenized datasets
#   $OLMO_ROOT/runs/     checkpoints, logs, metrics  $OLMO_ROOT/hf/     raw Hugging Face downloads
export OLMO_ROOT="${OLMO_ROOT:-/bigtemp/${USER:-$(id -un)}/olmo}"
FT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export FT_REPO
export FT="$FT_REPO/finetune"
export OLMO_VENV="${OLMO_VENV:-$OLMO_ROOT/venv}"

export HF_HOME="$OLMO_ROOT/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export UV_CACHE_DIR="$OLMO_ROOT/uv_cache"
export PIP_CACHE_DIR="$OLMO_ROOT/pip_cache"
export TRITON_CACHE_DIR="$OLMO_ROOT/triton_cache"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OLMO_ROOT"/{models,data,runs,hf,hf_cache} 2>/dev/null || true

if [ -f "$OLMO_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$OLMO_VENV/bin/activate"
fi

# ft_load CONFIG -> exports FT_NAME FT_MODEL_DIR FT_DATA_DIR FT_RUN_DIR FT_GPUS FT_TIME ... (fails on a bad YAML)
ft_load() {
    local exports
    exports=$(python "$FT/ft.py" paths "$1") || return 1
    eval "$exports"
}
