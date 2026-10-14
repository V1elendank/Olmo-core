#!/usr/bin/env bash
# Drive the UVA CS cluster from your Mac (run from anywhere in the repo). You type your CS password
# (and Duo) yourself. Most commands reuse the connection for 2 hours; the live panel owns its tunnel.
#
#   bash finetune/scripts/launch_from_mac.sh <id> setup                 push code, update the cluster copy,
#                                                                       build the venv if missing, preflight
#   bash finetune/scripts/launch_from_mac.sh <id> submit CONFIG [opts]  push + copy CONFIG, then submit.sh
#                                                                       (opts: --test --no-fit-check ...)
#   bash finetune/scripts/launch_from_mac.sh <id> status                queue + runs; snapshot of the control
#                                                                       panel opened in your browser
#   bash finetune/scripts/launch_from_mac.sh <id> panel                 LIVE control panel at localhost:8765
#   bash finetune/scripts/launch_from_mac.sh <id> logs RUN              copy RUN's logs to finetune/logs/RUN/
#   bash finetune/scripts/launch_from_mac.sh <id> download RUN [DIR]    copy RUN's Hugging Face export here
#   bash finetune/scripts/launch_from_mac.sh <id> stop CONFIG           clean stop of a running job
set -euo pipefail
ID="${1:?usage: bash finetune/scripts/launch_from_mac.sh <computing-id> setup|submit|status|panel|logs|download|stop ...}"
MODE="${2:-status}"
shift 2 || true
HOST="$ID@portal.cs.virginia.edu"
FORK_URL="${FORK_URL:-https://github.com/V1elendank/Olmo-core.git}"
BRANCH="${BRANCH:-uva-cs-finetune}"
REMOTE_REPO="/bigtemp/$ID/OLMo-core"
REMOTE_ROOT="/bigtemp/$ID/olmo"
PORT="${PORT:-8765}"
[[ "$ID" =~ ^[a-z][a-z0-9]*$ ]] || { echo "invalid UVA computing ID"; exit 2; }
[[ "$PORT" =~ ^[0-9]{1,5}$ ]] && [ "$((10#$PORT))" -ge 1024 ] && [ "$((10#$PORT))" -le 65535 ] \
    || { echo "PORT must be a number between 1024 and 65535"; exit 2; }
PORT="$((10#$PORT))"

cd "$(dirname "$0")/../.."
mkdir -p ~/.ssh finetune/logs
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" -o ControlPersist=2h -o ServerAliveInterval=60)
remote() { ssh "${SSH_OPTS[@]}" "$HOST" "bash -l -s"; } # script on stdin, login shell (Slurm in PATH)

SYNC="set -euo pipefail
mkdir -p /bigtemp/$ID
if [ -d '$REMOTE_REPO/.git' ]; then
    cd '$REMOTE_REPO' && git fetch -q origin && git checkout -q $BRANCH && git pull -q --ff-only
else
    git clone -q -b $BRANCH '$FORK_URL' '$REMOTE_REPO' && cd '$REMOTE_REPO'
fi
echo \">>> cluster repo at \$(git rev-parse --short HEAD)\""

push() {
    if ! git diff --quiet HEAD -- finetune src 2>/dev/null; then
        echo ">>> NOTE: uncommitted changes in finetune/ or src/ are NOT sent (commit them first); configs are copied directly"
    fi
    git push -q origin "$BRANCH"
}

case "$MODE" in
setup)
    push
    remote <<EOF
$SYNC
if [ -f $REMOTE_ROOT/venv/bin/activate ] && [ "${REINSTALL:-0}" != 1 ]; then
    echo ">>> venv exists (REINSTALL=1 to rebuild)"
else
    bash finetune/scripts/setup_env.sh
fi
bash finetune/scripts/preflight.sh
EOF
    ;;

submit)
    CONFIG="${1:?usage: ... submit finetune/configs/<run>.yaml [submit.sh options]}"
    shift
    [ -f "$CONFIG" ] || { echo "no such file: $CONFIG"; exit 1; }
    push
    remote <<<"$SYNC" >/dev/null
    scp "${SSH_OPTS[@]}" -q "$CONFIG" "$HOST:$REMOTE_REPO/finetune/configs/$(basename "$CONFIG")"
    remote <<EOF
cd $REMOTE_REPO && bash finetune/scripts/submit.sh finetune/configs/$(basename "$CONFIG") $*
EOF
    ;;

status)
    remote <<EOF
cd $REMOTE_REPO
source finetune/scripts/env.sh
squeue -u $ID -o '%.10i %.28j %.8T %.10M %.11l %R'
echo
python finetune/ft.py list
python finetune/ft.py status >/dev/null
EOF
    scp "${SSH_OPTS[@]}" -q "$HOST:$REMOTE_REPO/finetune/dashboard/data.js" finetune/dashboard/data.js
    echo ">>> control panel snapshot: finetune/dashboard/index.html"
    command -v open >/dev/null && open finetune/dashboard/index.html || true
    ;;

panel)
    echo ">>> starting the live control panel on the portal (Ctrl-C to stop)"
    echo ">>> after authentication and the server's ready message, open http://localhost:$PORT/launch.html"
    echo ">>> panel uses the code already on the cluster; it does not sync local changes"
    # A dedicated connection closes its forward with the panel instead of leaving it on the
    # two-hour control master. Fail before starting a remote panel if the local port is occupied.
    ssh -S none -o ControlMaster=no -o ControlPersist=no -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=60 -t -L "127.0.0.1:$PORT:127.0.0.1:$PORT" "$HOST" \
        "bash -lc 'cd $REMOTE_REPO && test -f finetune/ftlib/web.py && PORT=$PORT bash finetune/scripts/panel.sh'" \
        || { rc=$?; echo ">>> panel stopped with an error; check SSH, port $PORT, and the cluster code/environment above"; exit "$rc"; }
    ;;

logs)
    RUN="${1:?usage: ... logs RUN_NAME}"
    mkdir -p "finetune/logs/$RUN"
    rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude='step*/' --exclude='hf-*/' --include='*/' --include='*.log' \
        --include='*.out' --include='*.json' --include='*.jsonl' --exclude='*' --prune-empty-dirs \
        "$HOST:$REMOTE_ROOT/runs/$RUN/" "finetune/logs/$RUN/"
    echo ">>> finetune/logs/$RUN/"; ls "finetune/logs/$RUN"
    ;;

download)
    RUN="${1:?usage: ... download RUN_NAME [DEST_DIR]}"
    DEST="${2:-$HOME/Desktop/project/models}"
    SRC=$(ssh "${SSH_OPTS[@]}" "$HOST" "ls -d $REMOTE_ROOT/runs/$RUN/hf-step* 2>/dev/null | sort -V | tail -n 1")
    [ -n "$SRC" ] || { echo "no Hugging Face export in $REMOTE_ROOT/runs/$RUN (run: submit.sh CONFIG --export-only)"; exit 1; }
    mkdir -p "$DEST/$RUN"
    echo ">>> $SRC -> $DEST/$RUN/ (7B: ~15 GB)"
    rsync -az --progress -e "ssh ${SSH_OPTS[*]}" "$SRC/" "$DEST/$RUN/"
    ;;

stop)
    CONFIG="${1:?usage: ... stop finetune/configs/<run>.yaml}"
    remote <<EOF
cd $REMOTE_REPO && source finetune/scripts/env.sh && python finetune/ft.py stop finetune/configs/$(basename "$CONFIG")
EOF
    ;;

*)
    echo "unknown mode '$MODE' (setup | submit | status | panel | logs | download | stop)"; exit 2 ;;
esac
