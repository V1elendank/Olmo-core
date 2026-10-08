# OLMo-2 7B fine-tuning on the UVA CS Slurm cluster

Feasibility test: can a short OLMo-core SFT of a 7B model run on department GPUs?
Base model `allenai/OLMo-2-1124-7B`, 5k conversations from `allenai/tulu-3-sft-mixture`,
30 optimizer steps × 32 seqs × 4096 tokens (~3.9M tokens). Full-parameter FSDP, no checkpoints saved by default.

| File | What it does |
|---|---|
| `env.sh` | Paths (all under `/bigtemp/$USER/olmo`), caches, venv activation |
| `setup_env.sh` | One-time: uv venv with torch (CUDA wheels) + OLMo-core |
| `00_prepare.sbatch` | CPU job: download HF model → convert to OLMo-core → tokenize SFT data |
| `01_sft.sbatch` | GPU job: `torchrun` FSDP SFT on one node (default 4× A100 80GB) |
| `prep_sft_data.py` | Chat-template tokenization with assistant-only label masks |
| `sft_olmo2_7b_uva.py` | Beaker-free version of `src/scripts/train/sft/Olmo-2-7B-SFT.py` |

## Run it

From your Mac, one command does everything below (push branch, clone on cluster, setup, submit):

```bash
cd ~/Desktop/project/OLMo-core
bash uva/launch_from_mac.sh <computing-id>          # launch
bash uva/launch_from_mac.sh <computing-id> status   # later: queue + copy logs to uva/logs/remote/
```

Or by hand on the cluster:

```bash
ssh <computing-id>@portal.cs.virginia.edu
mkdir -p /bigtemp/$USER && cd /bigtemp/$USER
git clone <your fork URL> OLMo-core && cd OLMo-core && git checkout uva-cs-finetune
mkdir -p uva/logs

TORCH_CUDA=cu128 bash uva/setup_env.sh           # ~5-10 min, on the portal node
jid=$(sbatch --parsable uva/00_prepare.sbatch)   # ~1-2 h (download + convert + tokenize)
sbatch --dependency=afterok:$jid uva/01_sft.sbatch

squeue -u $USER
tail -f uva/logs/olmo2-7b-sft-*.out
```

If compute nodes can't reach Hugging Face, run the three steps from `00_prepare.sbatch` by hand on the portal
(`source uva/env.sh` first). The conversion step needs ~100 GB RAM, so keep it in a CPU job if you can.

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
