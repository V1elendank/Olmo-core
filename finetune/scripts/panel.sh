#!/usr/bin/env bash
# Control panel for your fine-tuning runs (finetune/dashboard/index.html).
#
# Live, from your laptop (one command; refreshes every minute; Ctrl-C to stop):
#     bash finetune/scripts/launch_from_mac.sh <computing-id> panel      -> open http://localhost:8765
# On the portal itself:
#     bash finetune/scripts/panel.sh            # serve on 127.0.0.1:8765 (tunnel it with ssh -L)
#     bash finetune/scripts/panel.sh snapshot   # just write finetune/dashboard/data.js
set -euo pipefail
cd "$(dirname "$0")/../.."
source finetune/scripts/env.sh
if [ "${1:-serve}" = snapshot ]; then
    python "$FT/ft.py" status
else
    python "$FT/ft.py" status --serve "${PORT:-8765}" --interval "${INTERVAL:-60}"
fi
