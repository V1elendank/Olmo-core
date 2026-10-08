#!/usr/bin/env bash
# Quick checks (~5-10 min) to run on the portal BEFORE the multi-hour jobs. From the repo root:
#     bash uva/preflight.sh
# Prints PASS/WARN/FAIL per check and a summary. Nothing here uses a GPU.
set -uo pipefail
source "$(dirname "$0")/env.sh"

FAILS=0
WARNS=0
pass() { echo "PASS  $*"; }
warn() { echo "WARN  $*"; WARNS=$((WARNS + 1)); }
fail() { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }

echo "== T1  Python environment"
if python -c "import torch, olmo_core, transformers, datasets; print('   torch', torch.__version__, 'cuda', torch.version.cuda, '| olmo_core', olmo_core.__version__)"; then
    pass "venv imports torch / olmo_core / transformers / datasets"
else
    fail "venv broken: rerun  bash uva/setup_env.sh"
fi

echo "== T2  Storage"
if touch "$OLMO_ROOT/.write_test" 2>/dev/null && rm -f "$OLMO_ROOT/.write_test"; then
    pass "$OLMO_ROOT is writable"
else
    fail "cannot write to $OLMO_ROOT"
fi
df -h /bigtemp | tail -n 1 | awk '{print "   /bigtemp free: " $4}'
echo "   need ~200 GB: HF model 29 GB + converted ckpt 29 GB + venv/caches"

echo "== T3  Training config builds (dry run)"
if python uva/sft_olmo2_7b_uva.py dry_run --checkpoint=none --dataset_path=/dev/null \
    --save_folder=/tmp/olmo-dry-run-$USER >/tmp/olmo-dry-run-$USER.log 2>&1; then
    pass "7B SFT config OK (full config in /tmp/olmo-dry-run-$USER.log)"
else
    fail "dry run failed: see /tmp/olmo-dry-run-$USER.log"
fi

echo "== T4  Hugging Face reachable from the portal"
if python -c "from huggingface_hub import HfApi; i=HfApi().model_info('$HF_BASE_MODEL'); print('   ', i.id, 'OK')"; then
    pass "can see $HF_BASE_MODEL"
else
    fail "cannot reach huggingface.co from the portal"
fi

echo "== T5  Tokenize 50 conversations (checks tokenizer + chat template + label masks)"
SMOKE_DATA="$OLMO_ROOT/data/smoke"
if python uva/prep_sft_data.py --out "$SMOKE_DATA" --num_examples 50 >/tmp/olmo-prep-smoke-$USER.log 2>&1 \
    && python - "$SMOKE_DATA" <<'EOF'
import json, sys, numpy as np
d = sys.argv[1]
s = json.load(open(f"{d}/stats.json"))
ids = np.memmap(f"{d}/token_ids_part_0000.npy", dtype=np.uint32, mode="r")
m = np.memmap(f"{d}/labels_mask_0000.npy", dtype=np.bool_, mode="r")
assert ids.size == m.size == s["total_tokens"], "size mismatch"
assert ids.max() < 100278, "token id out of vocab"
frac = m.mean()
assert 0.05 < frac < 0.95, f"odd label fraction {frac:.2f}"
print(f"    {s['examples']} examples, {s['total_tokens']} tokens, {frac:.0%} trainable")
EOF
then
    pass "tokenized smoke data at $SMOKE_DATA"
else
    fail "tokenization failed: see /tmp/olmo-prep-smoke-$USER.log"
fi

echo "== T6  Internet from a compute node (decides where downloads run)"
CODE=$(srun -p cpu -t 00:03:00 --mem=1000 --immediate=300 \
    curl -s -o /dev/null -w '%{http_code}' https://huggingface.co 2>/dev/null || echo "none")
if [ "$CODE" = "200" ]; then
    pass "compute nodes can reach Hugging Face: 00_prepare.sbatch can download"
else
    warn "compute node got '$CODE': run  bash uva/download_on_portal.sh  before the full run"
fi

echo "== T7  Target GPUs in the gpu partition"
sinfo -p gpu -h -o '%N %t %f' | grep -E 'a100_80gb|h100_94gb' | sed 's/^/    /' || true

echo
echo "Preflight: $FAILS fail, $WARNS warn"
[ "$FAILS" -eq 0 ] && echo ">>> OK to submit the GPU smoke test:  sbatch uva/smoke_test.sbatch"
exit "$FAILS"
