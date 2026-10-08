#!/usr/bin/env bash
# Run a command and kill it if it goes quiet, so a hung job releases its GPUs instead of
# sitting idle until the Slurm time limit.
#
#   bash uva/run_with_watchdog.sh <logfile> <command> [args...]
#
# Output is written to <logfile> and mirrored to stdout (the Slurm log). If <logfile> gets no
# new output for STALL_MIN minutes (default 12; or STALL_SEC seconds), the whole process group
# is terminated and the script exits with code 124.
set -uo pipefail

LOG="$1"
shift
STALL_SEC="${STALL_SEC:-$((${STALL_MIN:-12} * 60))}"
CHECK_SEC="${CHECK_SEC:-30}"
export PYTHONUNBUFFERED=1 # so log timestamps reflect real progress

mkdir -p "$(dirname "$LOG")"
: >"$LOG"
setsid "$@" >>"$LOG" 2>&1 &
PID=$!
tail -n +1 -f --pid="$PID" "$LOG" &
TAIL_PID=$!

STALLED=0
while kill -0 "$PID" 2>/dev/null; do
    sleep "$CHECK_SEC"
    kill -0 "$PID" 2>/dev/null || break
    idle=$(($(date +%s) - $(stat -c %Y "$LOG")))
    if [ "$idle" -ge "$STALL_SEC" ]; then
        echo "WATCHDOG: no output for ${idle}s (limit ${STALL_SEC}s); killing the run to free the GPUs" | tee -a "$LOG"
        kill -TERM -- "-$PID" 2>/dev/null
        sleep 20
        kill -KILL -- "-$PID" 2>/dev/null
        STALLED=1
        break
    fi
done

wait "$PID"
RC=$?
wait "$TAIL_PID" 2>/dev/null
[ "$STALLED" = 1 ] && RC=124
echo "WATCHDOG: command exited with code $RC"
exit "$RC"
