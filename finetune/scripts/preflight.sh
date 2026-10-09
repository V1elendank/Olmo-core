#!/usr/bin/env bash
# Quick checks on the portal before spending GPU hours (~2 min, no GPU):
#     bash finetune/scripts/preflight.sh [CONFIG.yaml]
set -uo pipefail
cd "$(dirname "$0")/../.."
source finetune/scripts/env.sh
FAILS=0 WARNS=0
pass() { echo "PASS  $*"; }
warn() { echo "WARN  $*"; WARNS=$((WARNS + 1)); }
fail() { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }

echo "== P1  Python environment"
python -c "import torch, transformers, datasets, yaml; from olmo_core.version import VERSION; print('   torch', torch.__version__, 'cuda', torch.version.cuda, '| olmo_core', VERSION, '| transformers', transformers.__version__)" \
    && pass "venv OK ($OLMO_VENV)" || fail "venv broken: bash finetune/scripts/setup_env.sh"

echo "== P2  Storage"
touch "$OLMO_ROOT/.w" 2>/dev/null && rm -f "$OLMO_ROOT/.w" && pass "$OLMO_ROOT writable" || fail "cannot write $OLMO_ROOT"
df -h "$OLMO_ROOT" | tail -n 1 | awk '{print "   free on " $6 ": " $4 "   (7B: base model 15-29 GB, each checkpoint ~88 GB, HF export ~15 GB)"}'

echo "== P3  Configs"
for c in finetune/configs/*.yaml; do
    case "$(basename "$c")" in _*) continue ;; esac
    if out=$(python "$FT/ft.py" validate "$c" 2>&1); then pass "$(basename "$c")"; else fail "$(basename "$c"): $(echo "$out" | tail -n 3 | tr '\n' ' ')"; fi
done

if [ -n "${1:-}" ]; then
    echo "== P4  $1: tokenizer + first rows of every dataset reachable"
    python - "$1" <<'PY' && pass "tokenizer and data sources readable" || fail "see the error above"
import sys, itertools
sys.path.insert(0, "finetune")
from ftlib.config import load_config
from ftlib.data import iter_rows, load_tokenizer
cfg = load_config(sys.argv[1]); tok = load_tokenizer(cfg)
print(f"   tokenizer {cfg.tokenizer}: {len(tok)} tokens, chat template {'yes' if tok.chat_template else 'no'}")
for d in cfg.data:
    rows = list(itertools.islice(iter_rows(d), 2))
    print(f"   {d.name}: columns {sorted(rows[0])}")
PY
fi

echo "== P5  Internet from a compute node (downloads run in the prepare job)"
CODE=$(srun -p cpu -t 00:03:00 --mem=1000 --immediate=300 curl -s -o /dev/null -w '%{http_code}' https://huggingface.co 2>/dev/null || echo none)
[ "$CODE" = 200 ] && pass "compute nodes reach Hugging Face" || warn "compute node got '$CODE': run  python finetune/ft.py prepare CONFIG  on the portal instead"

echo "== P6  GPUs (gpu partition)"
sinfo -p gpu -h -N -o '%N %t %G %f' 2>/dev/null | grep -E 'a100|h100|h200' | sed 's/^/    /' || true
echo
echo "Preflight: $FAILS fail, $WARNS warn"
exit "$FAILS"
