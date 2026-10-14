# Control panel

`index.html` is a self-contained page that reads `data.js` (written by `python finetune/ft.py status`).

* **Live** (refreshes every minute), from your Mac:
  `bash finetune/scripts/launch_from_mac.sh <id> panel` → http://localhost:8765
* **Snapshot** opened from disk: `bash finetune/scripts/launch_from_mac.sh <id> status`
* On the portal: `bash finetune/scripts/panel.sh` (serve) or `bash finetune/scripts/panel.sh snapshot`.

Tabs: Overview (runs, queue, free GPUs, disk) · Runs (loss / throughput / memory / LR / grad-norm
charts, ETA, checkpoints, pipeline jobs, events, log tail, copy-paste commands) · Data (token stats,
length histograms, decoded examples with the trained-on tokens highlighted) · Models & configs ·
Cluster (GPU nodes, queue, 7-day history).

`data.js` is generated and not committed.

**New fine-tune** opens `launch.html`: independently upload a model folder and dataset files,
select the completed inputs, configure training, and click Start. It uses a loopback-only API
served by `ft.py status --serve`; static snapshots remain read-only. Uploads are resumable and
versioned, and unused uploads can be removed. Every web launch retains the GPU fit-check gate.
See the parent README for accepted files, storage locations, and testing instructions.
