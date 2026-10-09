#!/usr/bin/env bash
# Submit a whole fine-tuning run from one YAML, on the portal (from anywhere in the repo):
#
#     bash finetune/scripts/submit.sh finetune/configs/<run>.yaml [options]
#
# Chain (each job starts only if the previous one succeeded, so a failure never wastes GPU time):
#     prepare (CPU, skipped if done) -> fit check (GPU, few steps) -> train (GPU) -> export to HF (CPU)
#
# Options:
#     --test            run the CPU pipeline test (scripts/local_test.sh, ~5 min) first
#     --no-fit-check    skip the short GPU fit check (e.g. when resuming a run that already worked)
#     --no-export       don't convert the final checkpoint to Hugging Face format
#     --prepare-only    only the CPU prepare job
#     --export-only     only the export job (latest checkpoint of the run)
#     --after JOBID     make the first job wait for another job (afterok)
#     --print           show the sbatch commands without submitting
#
# Logs: $OLMO_ROOT/runs/<name>/logs/   Status: python finetune/ft.py list   or the control panel
set -euo pipefail
cd "$(dirname "$0")/../.."
CONFIG="${1:?usage: bash finetune/scripts/submit.sh CONFIG.yaml [--test] [--no-fit-check] [--no-export] [--prepare-only] [--export-only] [--after JOBID] [--print]}"
shift
CONFIG="$(realpath "$CONFIG")"
TEST=0 FIT=1 EXPORT=1 ONLY="" AFTER="" PRINT=0
while [ $# -gt 0 ]; do
    case "$1" in
    --test) TEST=1 ;;
    --no-fit-check) FIT=0 ;;
    --no-export) EXPORT=0 ;;
    --prepare-only) ONLY=prepare ;;
    --export-only) ONLY=export ;;
    --after) AFTER="$2"; shift ;;
    --print) PRINT=1 ;;
    *) echo "unknown option $1"; exit 2 ;;
    esac
    shift
done

source finetune/scripts/env.sh
echo "== 1. config"
python "$FT/ft.py" validate "$CONFIG"
ft_load "$CONFIG"
[ "$FT_EXPORT_HF" = 1 ] || EXPORT=0
[ "$FT_FIT_STEPS" -gt 0 ] || FIT=0

if [ "$TEST" = 1 ]; then
    echo "== 2. CPU pipeline test"
    bash "$FT/scripts/local_test.sh" || { echo ">>> local test FAILED: nothing submitted"; exit 1; }
fi

MODEL_READY=0 DATA_READY=0
[ -f "$FT_MODEL_DIR/model_and_optim/.metadata" ] && MODEL_READY=1
[ -f "$FT_DATA_DIR/manifest.json" ] && [ -f "$FT_DATA_DIR/pack.json" ] && DATA_READY=1

echo "== 3. training config builds (dry run, no GPU)"
DRY_LOG="$(mktemp)"
if python "$FT/ft.py" dry-run "$CONFIG" >"$DRY_LOG" 2>&1; then
    tail -n 1 "$DRY_LOG" | sed 's/^/   /'
elif [ "$MODEL_READY" = 1 ]; then
    tail -n 25 "$DRY_LOG"; echo ">>> dry run FAILED: nothing submitted"; exit 1
else
    echo "   (skipped: the model config is only known after prepare; set model.arch to check it now)"
fi
rm -f "$DRY_LOG"

LOGDIR="$FT_RUN_DIR/logs"
mkdir -p "$LOGDIR"
COMMON=(--output="$LOGDIR/%x-%j.out" --job-name)
MAIL=()
[ -n "$FT_MAIL" ] && MAIL=(--mail-type=END,FAIL --mail-user="$FT_MAIL")
GPU=(--partition="$FT_PARTITION" --gres="gpu:$FT_GPUS" --cpus-per-task="$FT_CPUS" --mem="$FT_MEM")
[ -n "$FT_CONSTRAINT" ] && GPU+=(--constraint="$FT_CONSTRAINT")

LAST="$AFTER"
submit() { # submit STAGE NAME [sbatch args...] -- SCRIPT ARGS
    local stage="$1" name="$2"; shift 2
    local dep=()
    [ -n "$LAST" ] && dep=(--dependency="afterok:$LAST")
    if [ "$PRINT" = 1 ]; then
        echo "   sbatch ${COMMON[*]} $name ${MAIL[*]} ${dep[*]} $*"
        LAST="<$stage>"
        return
    fi
    local id
    id=$(sbatch --parsable "${COMMON[@]}" "$name" "${MAIL[@]}" "${dep[@]}" "$@")
    printf '{"time": "%s", "stage": "%s", "job_id": "%s", "after": "%s"}\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "$stage" "$id" "$LAST" >>"$FT_RUN_DIR/jobs.jsonl"
    echo "   $stage: job $id${LAST:+ (after $LAST)}"
    LAST="$id"
}

echo "== 4. submitting ($FT_NAME)"
if [ "$ONLY" = export ]; then
    submit export "ft-export-$FT_NAME" finetune/slurm/export.sbatch "$CONFIG"
else
    if [ "$MODEL_READY$DATA_READY" != 11 ] || [ "$ONLY" = prepare ]; then
        submit prepare "ft-prep-$FT_NAME" finetune/slurm/prepare.sbatch "$CONFIG"
    else
        echo "   prepare: skipped (model, data and packing already done)"
    fi
    if [ "$ONLY" != prepare ]; then
        if [ "$FIT" = 1 ]; then
            submit fit-check "ft-fit-$FT_NAME" "${GPU[@]}" --time=00:45:00 finetune/slurm/train.sbatch "$CONFIG" --fit-check
        fi
        submit train "ft-train-$FT_NAME" "${GPU[@]}" --time="$FT_TIME" finetune/slurm/train.sbatch "$CONFIG"
        if [ "$EXPORT" = 1 ]; then
            submit export "ft-export-$FT_NAME" finetune/slurm/export.sbatch "$CONFIG"
        fi
    fi
fi
[ "$PRINT" = 1 ] && exit 0
echo
echo ">>> logs:     $LOGDIR/"
echo ">>> progress: python finetune/ft.py list      (or the control panel: bash finetune/scripts/panel.sh)"
echo ">>> stop:     python finetune/ft.py stop $CONFIG   (clean stop + checkpoint)   or  scancel <job>"
squeue -u "$USER" -o '%.10i %.28j %.8T %.10M %.11l %R' 2>/dev/null || true
