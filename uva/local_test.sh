#!/usr/bin/env bash
# CPU-only end-to-end test of the training pipeline, before spending any GPU time.
# Runs on a laptop, the portal, or any Linux box with the venv (no GPU, no internet needed).
#
#     bash uva/local_test.sh            # from the repo root, ~3-6 min
#
# Uses a tiny random-init OLMo-2-style model (~55M params) and synthetic token data, but the
# SAME script, data format, checkpoint format and code paths as the real 7B run:
#   L1 checkpoint save -> load        L4 checkpointing during training
#   L2 data packing (prep_data)       L5 resume after interruption
#   L3 bad input fails fast           L6 watchdog kills a hung run
set -uo pipefail
cd "$(dirname "$0")/.."
T="${LOCAL_TEST_DIR:-/tmp/olmo-local-test-${USER:-$(id -un)}}"
rm -rf "$T" && mkdir -p "$T"
export UVA_CPU_TEST=1 OLMO_SHARED_FS=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
NPROC="${NPROC:-2}"
PASS=0
FAIL=0
ok() { echo "PASS  $*"; PASS=$((PASS + 1)); }
bad() { echo "FAIL  $*"; FAIL=$((FAIL + 1)); }
S=uva/sft_olmo2_7b_uva.py
COMMON=(--model_size=tiny --dataset_path="$T/data" --data_work_dir="$T/cache" --seq_len=512
    --global_batch_seqs=4 --microbatch_seqs=1 --data_loader.num_workers=0)

echo "== L0  synthetic SFT data (same raw uint32/bool format as prep_sft_data.py)"
python - "$T/data" <<'EOF'
import os, sys, numpy as np
out = sys.argv[1]; os.makedirs(out)
rng = np.random.default_rng(0); ids = []; mask = []
for _ in range(400):
    n = int(rng.integers(64, 700)); d = rng.integers(0, 100_000, n).astype(np.uint32)
    d[0] = d[-1] = 100257; m = np.zeros(n, bool); m[n // 2:] = True; ids.append(d); mask.append(m)
ids, mask = np.concatenate(ids), np.concatenate(mask)
a = np.memmap(f"{out}/token_ids_part_0000.npy", dtype=np.uint32, mode="w+", shape=ids.shape); a[:] = ids; a.flush()
b = np.memmap(f"{out}/labels_mask_0000.npy", dtype=np.bool_, mode="w+", shape=mask.shape); b[:] = mask; b.flush()
print(f"   {ids.size:,} tokens")
EOF

echo "== L1  save a base checkpoint in OLMo-core format, then load it in training"
if python - "$T/base/model_and_optim" <<'EOF' >"$T/l1.log" 2>&1; then
import sys
sys.path.insert(0, "uva")
import sft_olmo2_7b_uva as S
from olmo_core.distributed.checkpoint import save_model_and_optim_state
args, ov = S.get_parser().parse_known_args(["dry_run", "--checkpoint=none", "--dataset_path=x",
                                            "--save_folder=x", "--model_size=tiny", "--world_size=2"])
model = S.build_config(args, ov).model.build(init_device="cpu")
save_model_and_optim_state(sys.argv[1], model, save_overwrite=True)
EOF
    [ -f "$T/base/model_and_optim/.metadata" ] && ok "base checkpoint written" || bad "no .metadata in base checkpoint"
else
    tail -n 20 "$T/l1.log"; bad "could not write base checkpoint"
fi

echo "== L2  data packing in a single process"
if python $S prep_data --checkpoint=none --save_folder="$T/runA" "${COMMON[@]}" >"$T/l2.log" 2>&1 \
    && grep -q "Dataset ready" "$T/l2.log"; then
    ok "$(grep -o 'Dataset ready.*' "$T/l2.log")"
else
    tail -n 20 "$T/l2.log"; bad "prep_data failed"
fi

echo "== L3  bad input fails fast (missing data dir)"
start=$(date +%s)
if python $S prep_data --checkpoint=none --save_folder="$T/bad" --dataset_path="$T/nope" \
    --model_size=tiny --seq_len=512 --data_work_dir="$T/cache" >"$T/l3.log" 2>&1; then
    bad "prep_data succeeded on a missing dataset"
else
    ok "missing data -> non-zero exit after $(($(date +%s) - start))s"
fi

echo "== L4  train from the base checkpoint, checkpoint every 2 steps (4 steps)"
if torchrun --standalone --nproc-per-node="$NPROC" $S train --checkpoint="$T/base/model_and_optim" \
    --save_folder="$T/runA" --steps=4 --ephemeral_interval=2 --save_checkpoints "${COMMON[@]}" \
    >"$T/l4.log" 2>&1; then
    grep -q "Checkpoint successfully loaded" "$T/l4.log" && ok "base checkpoint loaded" || bad "base checkpoint not loaded"
    grep -q "Training complete" "$T/l4.log" && ok "4 steps trained" || bad "training did not complete"
    [ -f "$T/runA/step4/model_and_optim/.metadata" ] && ok "checkpoint saved at step 4" || bad "no step4 checkpoint"
    grep -oE "train/CE loss=[0-9.]+" "$T/l4.log" | head -n 4 | sed 's/^/      /'
else
    tail -n 30 "$T/l4.log"; bad "training crashed"
fi

echo "== L5  rerun the same run with --steps=6: must RESUME at step 4, not restart"
if torchrun --standalone --nproc-per-node="$NPROC" $S train --checkpoint="$T/base/model_and_optim" \
    --save_folder="$T/runA" --steps=6 --ephemeral_interval=2 --save_checkpoints "${COMMON[@]}" \
    >"$T/l5.log" 2>&1; then
    grep -q "RESUMED from checkpoint" "$T/l5.log" && ok "$(grep -o 'RESUMED.*' "$T/l5.log")" || bad "did not resume"
    grep -q "step=6/6" "$T/l5.log" && ok "continued to step 6" || bad "did not reach step 6"
    grep -q "step=1/6" "$T/l5.log" && bad "restarted from step 1" || true
else
    tail -n 30 "$T/l5.log"; bad "resume run crashed"
fi

echo "== L6  watchdog kills a run that goes silent (simulated hang)"
start=$(date +%s)
UVA_TEST_HANG_SEC=600 STALL_SEC=25 CHECK_SEC=5 bash uva/run_with_watchdog.sh "$T/l6.log" \
    torchrun --standalone --nproc-per-node="$NPROC" $S train --checkpoint=none \
    --save_folder="$T/runHang" --steps=2 "${COMMON[@]}" >"$T/l6.out" 2>&1
rc=$?
took=$(($(date +%s) - start))
if [ "$rc" = 124 ] && [ "$took" -lt 180 ]; then
    ok "hung run killed after ${took}s (exit 124)"
else
    tail -n 10 "$T/l6.out"; bad "watchdog: rc=$rc after ${took}s"
fi
sleep 2
pgrep -f "runHang" >/dev/null && bad "hung processes left behind" || ok "no leftover processes"

echo
echo "Local test: $PASS pass, $FAIL fail   (logs in $T)"
[ "$FAIL" -eq 0 ] && echo ">>> pipeline OK; safe to submit GPU jobs"
exit "$FAIL"
