# OLMo-2 7B fine-tuning on the UVA CS Slurm cluster

Feasibility test: can a short OLMo-core SFT of a 7B model run on department GPUs?
Base model `allenai/OLMo-2-1124-7B`, 5k conversations from `allenai/tulu-3-sft-mixture`,
30 optimizer steps × 32 seqs × 4096 tokens (~3.9M tokens). Full-parameter FSDP, no checkpoints saved by default.

| File | What it does |
|---|---|
| `launch_from_mac.sh` | Mac-side launcher: `test` / `status` / `full` |
| `preflight.sh` | Portal checks before using GPUs |
| `smoke_test.sbatch` | 2-GPU tiny-model test of the whole stack |
| `download_on_portal.sh` | Fallback downloads if compute nodes are offline |
| `run_7b.sh` | Portal: CPU tests then submit 7B fit check + SFT |
| `local_test.sh` | CPU end-to-end pipeline test (tiny model) |
| `run_with_watchdog.sh` | Kills a run whose log goes silent |
| `env.sh` | Paths (all under `/bigtemp/$USER/olmo`), caches, venv activation |
| `setup_env.sh` | One-time: uv venv with torch (CUDA wheels) + OLMo-core |
| `00_prepare.sbatch` | CPU job: download HF model → convert to OLMo-core → tokenize SFT data |
| `01_sft.sbatch` | GPU job: `torchrun` FSDP SFT on one node (default 4× A100 80GB) |
| `prep_sft_data.py` | Chat-template tokenization with assistant-only label masks |
| `sft_olmo2_7b_uva.py` | Beaker-free version of `src/scripts/train/sft/Olmo-2-7B-SFT.py` |

## Run it: test first, then the long jobs

From your Mac, in `~/Desktop/project/OLMo-core`:

```bash
# Stage 1: tests, ~30 min. Push, clone, env setup, preflight (portal), then a 2-GPU smoke test
bash uva/launch_from_mac.sh <computing-id>
bash uva/launch_from_mac.sh <computing-id> status   # wait for "SMOKE TEST PASSED" in uva/logs/remote/

# Stage 2: hours. prepare -> 7B fit check (5 steps) -> 7B SFT (30 steps), each gated on the previous
bash uva/launch_from_mac.sh <computing-id> full
```

| Test | Where | Time | Checks |
|---|---|---|---|
| `preflight.sh` T1-T7 | portal | 5-10 min | venv imports, /bigtemp writable, 7B config dry run, HF reachable, tokenizer + label masks on 50 convos, compute-node internet, GPU nodes up |
| `smoke_test.sbatch` | 2 GPUs | ~5 min + queue | driver, CUDA, bf16 matmul per GPU, NCCL/FSDP, full OLMo-core loop on a random-init 190M model |
| 7B fit check | target GPUs | ~10 min | real 7B weights load, 5 steps fit in memory, throughput |

If preflight T6 warns that compute nodes have no internet, run `bash uva/download_on_portal.sh` on the portal before Stage 2.

By hand on the cluster (same steps):

```bash
cd /bigtemp/$USER/OLMo-core
bash uva/setup_env.sh && bash uva/preflight.sh && sbatch uva/smoke_test.sbatch
jid=$(sbatch --parsable uva/00_prepare.sbatch)
fit=$(sbatch --parsable --dependency=afterok:$jid --export=ALL,STEPS=5 uva/01_sft.sbatch)
sbatch --dependency=afterok:$fit uva/01_sft.sbatch
```

## Safety: not holding GPUs for nothing

- **`run_7b.sh`** (portal) runs `local_test.sh` + checks on the real inputs on CPU, and only then submits the GPU jobs.
- **`local_test.sh`** is a CPU-only end-to-end test (~3-5 min) with a tiny model: checkpoint load, data packing, fail-fast on bad input,
  checkpoint save, resume after interruption, and the watchdog killing a hung run.
- **`01_sft.sbatch` prechecks** fail in seconds, before training starts, if the checkpoint, the data or CUDA is missing,
  or if data packing fails.
- **Watchdog** (`run_with_watchdog.sh`) kills the run if its log is silent for `STALL_MIN` (default 12) minutes.
- **Distributed timeout** is 10 min (was 15) so a stuck GPU rank errors out sooner.
- **Checkpoints**: `CKPT_EVERY=N` saves an overwritten checkpoint every N steps plus a final one (7B ≈ 88 GB each).
  Resubmitting with the same `RUN_NAME` **resumes** from the latest checkpoint instead of restarting.

## What "success" looks like

In `uva/logs/olmo2-7b-sft-<jobid>.out`:
- `Checkpoint successfully loaded` (base weights went in)
- per-step lines with `train/CE loss` starting around 1.0-1.5 and trending down
- `GPU memory` from the gpu_monitor callback (report peak per GPU)
- `throughput/device/TPS` (tokens/sec/GPU), which lets you estimate full-run cost
- the job ends with `Training complete` and no OOM / NCCL errors

## GPU options (from the CS wiki)

| Request | Hardware | Notes |
|---|---|---|
| `--gres=gpu:4 --constraint=a100_80gb` | cheetah04, 4× A100 80GB NVLink | default, most headroom |
| `--gres=gpu:2 --constraint=h100_94gb` | serval03/06-09, 2× H100 NVL 94GB | params+optimizer ≈ 58 GB/GPU; add `--full_ac` if OOM |
| `--gres=gpu:4 --constraint=a100_40gb` | cheetah01 | needs `EXTRA="--full_ac"`, may still be tight |
| `--gres=gpu:4 --constraint=a40` | jaguar01, 4× A40 48GB | needs `EXTRA="--full_ac"` |

Memory math: full fine-tuning with AdamW needs about 16 bytes/param (fp32 params, grads, 2 Adam moments)
≈ 117 GB for 7.3B params, sharded across GPUs by FSDP, plus activations.

## Knobs

- `STEPS=100 sbatch uva/01_sft.sbatch`: longer run
- `EXTRA="--attn_backend=flash_2"`: needs `flash-attn` installed; enables per-document attention masking
- `EXTRA="--compile"`: torch.compile (faster after warm-up)
- `EXTRA="--save_checkpoints"`: writes checkpoints (~120 GB each with optimizer state)
- Any OLMo-core override works too, for example `EXTRA="--train_module.optim.lr=1e-5"`

With the default `torch` (SDPA) attention backend, packed conversations can attend to each other.
That doesn't matter for a feasibility/throughput test, but use `flash_2` for a real SFT run.
