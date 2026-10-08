#!/usr/bin/env bash
# Run on your Mac from the repo root. Three modes:
#
#   bash uva/launch_from_mac.sh <computing-id>          # STAGE 1 (test, ~30 min): push branch, clone/update
#                                                       #   on cluster, set up env, run preflight checks,
#                                                       #   submit the 2-GPU smoke test
#   bash uva/launch_from_mac.sh <computing-id> status   # queue + copy cluster logs to uva/logs/remote/
#   bash uva/launch_from_mac.sh <computing-id> full     # STAGE 2 (hours): only after the smoke test PASSED.
#                                                       #   prepare -> 7B fit check (5 steps) -> 7B SFT (30 steps)
#
# You'll be asked for your CS password (and Duo, if enabled); the SSH connection is reused for 2 h.
# Everything printed is also saved to uva/logs/launch-*.log.
set -euo pipefail

ID="${1:?usage: bash uva/launch_from_mac.sh <computing-id> [status|full]}"
MODE="${2:-test}"
HOST="$ID@portal.cs.virginia.edu"
FORK_URL="${FORK_URL:-https://github.com/V1elendank/Olmo-core.git}"
BRANCH="uva-cs-finetune"
REMOTE_DIR="/bigtemp/$ID/OLMo-core"

cd "$(dirname "$0")/.."
mkdir -p uva/logs/remote ~/.ssh
LOG="uva/logs/launch-$(date +%Y%m%d-%H%M%S)-$MODE.log"
exec > >(tee -a "$LOG") 2>&1

SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" -o ControlPersist=2h
          -o ServerAliveInterval=60)

# Shell snippet run on the portal first in every non-status mode: get the repo up to date.
SYNC_REPO="set -euo pipefail
mkdir -p /bigtemp/$ID
if [ -d '$REMOTE_DIR/.git' ]; then
    cd '$REMOTE_DIR' && git fetch -q origin && git checkout -q $BRANCH && git pull -q --ff-only
else
    git clone -q -b $BRANCH '$FORK_URL' '$REMOTE_DIR' && cd '$REMOTE_DIR'
fi
mkdir -p uva/logs
echo \">>> repo at \$(pwd) @ \$(git rev-parse --short HEAD)\""

case "$MODE" in
status)
    ssh "${SSH_OPTS[@]}" "$HOST" "squeue -u $ID -o '%.10i %.16j %.8T %.10M %.12l %R'; \
        sacct -u $ID -S now-2days -o JobID%12,JobName%16,State,Elapsed,ExitCode | grep -v '\.ba\|\.ex' | tail -n 10"
    scp "${SSH_OPTS[@]}" -q "$HOST:$REMOTE_DIR/uva/logs/*.out" uva/logs/remote/ 2>/dev/null \
        && echo ">>> copied cluster logs to uva/logs/remote/" || echo ">>> no cluster logs yet"
    ;;

test)
    echo ">>> [1/4] Pushing $BRANCH to $FORK_URL"
    git push -u origin "$BRANCH"
    echo ">>> [2/4] Connecting to $HOST (enter your CS password / Duo if asked)"
    ssh "${SSH_OPTS[@]}" "$HOST" "bash -l -s" <<EOF
$SYNC_REPO
echo ">>> [3/4] Environment setup (5-10 min the first time)"
if [ -f /bigtemp/$ID/olmo/venv/bin/activate ] && [ "${REINSTALL:-0}" != 1 ]; then
    echo "   venv exists, skipping (rerun with REINSTALL=1 to rebuild)"
else
    bash uva/setup_env.sh
fi
echo ">>> [4/4] Preflight checks"
if bash uva/preflight.sh; then
    jid=\$(sbatch --parsable uva/smoke_test.sbatch)
    echo ">>> submitted GPU smoke test: job \$jid  (log: uva/logs/olmo-smoke-\$jid.out)"
else
    echo ">>> preflight FAILED: fix the FAIL lines above before submitting anything"
fi
squeue -u $ID
EOF
    echo ">>> Next: wait ~10-30 min, then  bash uva/launch_from_mac.sh $ID status"
    echo ">>> When the smoke log ends with 'SMOKE TEST PASSED':  bash uva/launch_from_mac.sh $ID full"
    ;;

full)
    git push -q origin "$BRANCH" || true
    ssh "${SSH_OPTS[@]}" "$HOST" "bash -l -s" <<EOF
$SYNC_REPO
if ! grep -qs "SMOKE TEST PASSED" uva/logs/olmo-smoke-*.out; then
    echo ">>> No passing smoke test found in uva/logs/. Run stage 1 first (or FORCE=1)."
    [ "${FORCE:-0}" = 1 ] || exit 1
fi
j1=\$(sbatch --parsable uva/00_prepare.sbatch)
j2=\$(sbatch --parsable --dependency=afterok:\$j1 --time=01:00:00 \
      --export=ALL,STEPS=5,RUN_NAME=olmo2-7b-fitcheck uva/01_sft.sbatch)
j3=\$(sbatch --parsable --dependency=afterok:\$j2 uva/01_sft.sbatch)
echo ">>> submitted: prepare \$j1 -> 7B fit check (5 steps) \$j2 -> 7B SFT (30 steps) \$j3"
echo ">>> each job only starts if the previous one succeeded"
squeue -u $ID
EOF
    echo ">>> Check progress with:  bash uva/launch_from_mac.sh $ID status"
    ;;

*)
    echo "unknown mode '$MODE' (use: test | status | full)"; exit 2 ;;
esac
