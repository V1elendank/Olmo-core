#!/usr/bin/env bash
# CPU-only, offline test of the whole fine-tuning pipeline (~5 min). Runs on the portal, a laptop,
# or any Linux box with the venv; no GPU, no internet. Uses a tiny random model and a byte-level
# stand-in for the dolma2 tokenizer, but the SAME code, data format and checkpoints as real runs.
#
#     bash finetune/scripts/local_test.sh          # from anywhere in the repo
#
#   T1 config validation           T6  fit-check mode (no checkpoints)
#   T2 data prep: chat / ShareGPT / T7  STOP file ends a run cleanly
#      prompt-response / text      T8  watchdog kills a hung run
#   T3 packing                     T9  export to Hugging Face + load with transformers
#   T4 train from a base model     T10 control-panel data collection
#   T5 resume after interruption
set -uo pipefail
cd "$(dirname "$0")/../.."
source finetune/scripts/env.sh
T="${LOCAL_TEST_DIR:-/tmp/ft-local-test-${USER:-$(id -un)}}"
rm -rf "$T" && mkdir -p "$T"
export OLMO_ROOT="$T/root" FT_CPU_TEST=1 OLMO_SHARED_FS=1 OMP_NUM_THREADS=2
NPROC="${NPROC:-2}"
PASS=0 FAIL=0
ok() { echo "PASS  $*"; PASS=$((PASS + 1)); }
bad() { echo "FAIL  $*"; FAIL=$((FAIL + 1)); }
FTPY="$FT/ft.py"
C="$T/tiny.yaml"
trainrun() { torchrun --standalone --nproc-per-node="$NPROC" "$FTPY" train "$C" "$@"; }

echo "== fixtures (tokenizer, datasets in 4 formats, tiny base model)"
python - "$T" <<'EOF' >"$T/fixtures.log" 2>&1 || { tail "$T/fixtures.log"; exit 1; }
import sys; from pathlib import Path
sys.path.insert(0, "finetune")
from ftlib.testing import make_test_tokenizer, make_test_datasets
from ftlib.models import save_test_checkpoint
t = Path(sys.argv[1])
make_test_tokenizer(t / "tok"); make_test_datasets(t / "ds"); save_test_checkpoint("tiny", t / "base")
EOF
cat >"$C" <<EOF
name: tiny-cpu
model: {source: olmo_core, path: $T/base}
tokenizer: $T/tok
data:
  - {name: chat, source: $T/ds/chat.jsonl, format: chat, system_prompt: "You are a helpful assistant."}
  - {name: sharegpt, source: $T/ds/sharegpt.json, format: chat}
  - {name: alpaca, source: $T/ds/alpaca.csv, format: prompt_response, num_examples: 200}
  - {name: text, source: $T/ds/text.jsonl, format: text}
train: {seq_len: 512, global_batch_seqs: 4, steps: 4, lr: 1.0e-3}
checkpoint: {every: 2}
slurm: {gpus: $NPROC, fit_check_steps: 2}
EOF

echo "== T1  config validation"
python "$FTPY" validate "$C" >"$T/t1.log" 2>&1 && ok "valid config accepted" || { cat "$T/t1.log"; bad "valid config rejected"; }
sed 's/format: text/format: txt/; s/steps: 4/steps: 4, epochs: 1/' "$C" >"$T/bad.yaml"
if python "$FTPY" validate "$T/bad.yaml" >"$T/t1b.log" 2>&1; then bad "bad config accepted"; else
    grep -q "format must be" "$T/t1b.log" && grep -q "exactly one of steps" "$T/t1b.log" \
        && ok "bad config rejected with clear errors" || { cat "$T/t1b.log"; bad "unclear error"; }
fi

echo "== T2  tokenize + label-mask 4 dataset formats"
if python "$FTPY" prepare-data "$C" >"$T/t2.log" 2>&1; then
    python - "$T" <<'EOF' && ok "4 datasets tokenized, masks correct" || bad "manifest/mask checks"
import glob, json, sys, numpy as np
d = glob.glob(f"{sys.argv[1]}/root/data/*/manifest.json")[0]; m = json.load(open(d)); base = d.rsplit("/", 1)[0]
assert [x["name"] for x in m["datasets"]] == ["chat", "sharegpt", "alpaca", "text"]
for x in m["datasets"]:
    ids = np.memmap(f"{base}/{x['files'][0]}", dtype=np.uint32, mode="r"); msk = np.memmap(f"{base}/{x['files'][1]}", dtype=np.bool_, mode="r")
    assert ids.size == msk.size == x["tokens"] and msk.sum() == x["trainable_tokens"], x["name"]
    assert ids.max() < 100278 and ids[-1] == 100257
    if x["format"] == "text": assert x["trainable_fraction"] == 1.0
    else:
        assert 0.2 < x["trainable_fraction"] < 0.95, x
        seg = x["samples"][0]["segments"]; assert not seg[0]["train"] and seg[-1]["train"] and seg[-1]["t"].endswith("<|endoftext|>")
    print(f"      {x['name']:9s} {x['format']:16s} {x['examples']:4d} ex  {x['tokens']:7d} tok  {x['trainable_fraction']:.0%} trainable")
assert m["datasets"][0]["samples"][0]["segments"][0]["t"].startswith("<|im_start|>system")
EOF
else tail -n 20 "$T/t2.log"; bad "prepare-data crashed"; fi

echo "== T3  packing (single process)"
python "$FTPY" pack "$C" >"$T/t3.log" 2>&1 && ok "$(grep -o 'Packed: .*' "$T/t3.log")" || { tail "$T/t3.log"; bad "pack failed"; }

echo "== T4  train 4 steps from the base model (checkpoint every 2)"
R="$T/root/runs/tiny-cpu"
if trainrun >"$T/t4.log" 2>&1; then
    grep -q "Checkpoint successfully loaded" "$T/t4.log" && ok "base model loaded" || bad "base model not loaded"
    [ -f "$R/step4/model_and_optim/.metadata" ] && ok "final checkpoint step4" || bad "no step4"
    python -c "import json,sys; s=json.load(open('$R/status.json')); assert s['state']=='finished' and s['step']==4, s" \
        && [ "$(grep -c '"train/CE loss"' "$R/metrics.jsonl")" -ge 3 ] && ok "status.json + metrics.jsonl written" || bad "monitor files"
else tail -n 30 "$T/t4.log"; bad "training crashed"; fi

echo "== T5  same config with --steps 6: must RESUME at step 4"
if trainrun --steps 6 >"$T/t5.log" 2>&1; then
    grep -q "RESUMED from checkpoint" "$T/t5.log" && ok "resumed at step 4" || bad "did not resume"
    grep -q "step=1/6" "$T/t5.log" && bad "restarted from step 1" || true
    [ -d "$R/step6" ] && [ ! -d "$R/step4" ] && [ ! -d "$R/step2" ] && ok "kept only the final checkpoint" || bad "checkpoint pruning: $(ls "$R")"
else tail -n 30 "$T/t5.log"; bad "resume crashed"; fi

echo "== T6  fit check (2 steps, separate folder, no checkpoints)"
if trainrun --fit-check >"$T/t6.log" 2>&1; then
    F="$T/root/runs/tiny-cpu-fitcheck"
    grep -q "step=2/2" "$T/t6.log" && ! ls -d "$F"/step* >/dev/null 2>&1 && ok "fit check ran without checkpoints" || bad "fit check output"
else tail -n 30 "$T/t6.log"; bad "fit check crashed"; fi

echo "== T7  STOP file ends a run cleanly (with a checkpoint to resume from)"
S="$T/root/runs/stoptest"
(for _ in $(seq 150); do grep -qs '"running"' "$S/status.json" && break; sleep 2; done
    sleep 8; python "$FTPY" stop "$C" --run-name stoptest >/dev/null) &
trainrun --run-name stoptest --steps 60 >"$T/t7.log" 2>&1
wait
python -c "import json; s=json.load(open('$S/status.json')); assert s['state']=='stopped' and s['step']<60, s" 2>/dev/null \
    && ls -d "$S"/step* >/dev/null 2>&1 && ok "stopped at step $(python -c "import json;print(json.load(open('$S/status.json'))['step'])") with checkpoint" \
    || { tail -n 15 "$T/t7.log"; bad "STOP not honored"; }

echo "== T8  watchdog kills a run that goes silent"
start=$(date +%s)
FT_TEST_HANG_SEC=600 STALL_SEC=25 CHECK_SEC=5 bash "$FT/scripts/run_with_watchdog.sh" "$T/t8.log" \
    torchrun --standalone --nproc-per-node="$NPROC" "$FTPY" train "$C" --run-name hang >"$T/t8.out" 2>&1
rc=$?
took=$(($(date +%s) - start))
[ "$rc" = 124 ] && [ "$took" -lt 180 ] && ok "hung run killed after ${took}s (exit 124)" || { tail "$T/t8.out"; bad "watchdog rc=$rc after ${took}s"; }
sleep 2
pgrep -f "run-name hang" >/dev/null && bad "hung processes left behind" || ok "no leftover processes"

echo "== T9  export to Hugging Face format"
if python "$FTPY" export "$C" >"$T/t9.log" 2>&1; then
    python - "$R/hf-step6" <<'EOF' >>"$T/t9.log" 2>&1 && ok "hf-step6 loads in transformers, chat template kept" || { tail "$T/t9.log"; bad "HF model unusable"; }
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(sys.argv[1]); t = AutoTokenizer.from_pretrained(sys.argv[1])
assert t.chat_template and "<|im_start|>" in t.chat_template
ids = t.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True, return_tensors="pt", return_dict=False)
out = m.generate(ids if torch.is_tensor(ids) else ids["input_ids"], max_new_tokens=3, do_sample=False)
assert out.shape[1] > 0
EOF
else tail -n 20 "$T/t9.log"; bad "export failed"; fi

echo "== T10 control-panel data"
if python "$FTPY" status --out "$T/data.js" >"$T/t10.log" 2>&1; then
    python - "$T/data.js" <<'EOF' && ok "data.js has runs, metrics, dataset samples" || bad "data.js content"
import json, sys
d = json.loads(open(sys.argv[1]).read().split("=", 1)[1].rstrip().rstrip(";"))
names = {r["name"] for r in d["runs"]}; assert {"tiny-cpu", "stoptest"} <= names, names
r = next(r for r in d["runs"] if r["name"] == "tiny-cpu"); assert len(r["metrics"]["series"]["train/CE loss"]) >= 5
assert d["datasets"][0]["datasets"][0]["samples"]
EOF
else tail "$T/t10.log"; bad "status crashed"; fi

echo
echo "Local test: $PASS pass, $FAIL fail   (logs in $T)"
[ "$FAIL" -eq 0 ] && echo ">>> pipeline OK; safe to submit GPU jobs"
exit "$FAIL"
