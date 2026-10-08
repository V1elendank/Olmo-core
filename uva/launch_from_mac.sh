#!/usr/bin/env bash
# Run on your Mac from the repo root.
#
#   bash uva/launch_from_mac.sh <computing-id>          # push branch, set up env on cluster, submit jobs
#   bash uva/launch_from_mac.sh <computing-id> status   # show queue + copy cluster logs to uva/logs/remote/
#
# You'll be asked for your CS password (and Duo, if enabled) once; the SSH connection is
# reused for 2 hours. Everything printed is also saved to uva/logs/launch-*.log.
set -euo pipefail

ID="${1:?usage: bash uva/launch_from_mac.sh <computing-id> [status]}"
MODE="${2:-launch}"
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

if [ "$MODE" = "status" ]; then
    ssh "${SSH_OPTS[@]}" "$HOST" "squeue -u $ID -o '%.10i %.14j %.8T %.10M %.12l %R'; \
        sacct -u $ID -S now-2days -o JobID%12,JobName%14,State,Elapsed,ExitCode | grep -v '\.ba\|\.ex' | tail -n 8"
    scp "${SSH_OPTS[@]}" -q "$HOST:$REMOTE_DIR/uva/logs/*.out" uva/logs/remote/ 2>/dev/null \
        && echo ">>> copied cluster logs to uva/logs/remote/" || echo ">>> no cluster logs yet"
    exit 0
fi

echo ">>> [1/3] Pushing $BRANCH to $FORK_URL"
git push -u origin "$BRANCH"

echo ">>> [2/3] Connecting to $HOST (enter your CS password if asked)"
ssh "${SSH_OPTS[@]}" "$HOST" "bash -l -s" <<EOF
set -euo pipefail
mkdir -p /bigtemp/$ID
if [ -d "$REMOTE_DIR/.git" ]; then
    cd "$REMOTE_DIR" && git fetch -q origin && git checkout -q $BRANCH && git pull -q --ff-only
else
    git clone -q -b $BRANCH "$FORK_URL" "$REMOTE_DIR" && cd "$REMOTE_DIR"
fi
mkdir -p uva/logs
echo ">>> repo at \$(pwd) @ \$(git rev-parse --short HEAD)"
sinfo -p gpu -o '%.12N %.6t %.30f %G' | grep -E 'a100_80gb|h100' || true

echo ">>> [3/3] Environment setup (5-10 min) then job submission"
bash uva/setup_env.sh
jid=\$(sbatch --parsable uva/00_prepare.sbatch)
jid2=\$(sbatch --parsable --dependency=afterok:\$jid uva/01_sft.sbatch)
echo ">>> submitted prepare job \$jid and SFT job \$jid2 (starts after prepare succeeds)"
squeue -u $ID
EOF

echo ">>> All done. Check progress later with: bash uva/launch_from_mac.sh $ID status"
