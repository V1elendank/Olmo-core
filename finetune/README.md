# finetune/: fine-tuning OLMo 2 / OLMo 3 on the UVA CS cluster

Each run is described by **one YAML file** in `finetune/configs/`: which model to start from,
which data to use, and how to train. One command then runs the whole pipeline on Slurm. A
control panel shows progress, loss curves, GPU use and exactly what the model is trained on.

```
finetune/
  configs/          one YAML per run (_template.yaml documents every option; examples/ has more)
  data_examples/    tiny files showing the accepted data formats
  ft.py             the CLI: new / validate / prepare / train / export / stop / status / list
  ftlib/            config, model prep, data prep, training, export, status collection
  slurm/            prepare.sbatch (CPU), train.sbatch (GPU), export.sbatch (CPU)
  scripts/          submit.sh, launch_from_mac.sh, local_test.sh, preflight.sh, panel.sh, setup_env.sh
  dashboard/        index.html control panel (reads data.js produced by `ft.py status`)
```

The `uva/` folder is the earlier feasibility test (job 6701929). It is kept for reference; new
work goes through `finetune/`.

## Quick start

From your Mac, in the repo:

```bash
bash finetune/scripts/launch_from_mac.sh dgg6b setup                    # once, or after pulling changes
cp finetune/configs/_template.yaml finetune/configs/my-run.yaml          # or: python finetune/ft.py new my-run
#   edit my-run.yaml: model, tokenizer, data, steps, GPUs
bash finetune/scripts/launch_from_mac.sh dgg6b submit finetune/configs/my-run.yaml --test
bash finetune/scripts/launch_from_mac.sh dgg6b panel                    # live control panel at localhost:8765
bash finetune/scripts/launch_from_mac.sh dgg6b download my-run          # HF model -> ~/Desktop/project/models/
```

Or directly on the portal (`ssh dgg6b@portal.cs.virginia.edu`, `cd /bigtemp/dgg6b/OLMo-core`):

```bash
source finetune/scripts/env.sh
python finetune/ft.py validate finetune/configs/my-run.yaml   # checks the YAML, shows paths + state
bash finetune/scripts/submit.sh finetune/configs/my-run.yaml   # prepare -> fit check -> train -> export
python finetune/ft.py list                                     # what is prepared / running / done
python finetune/ft.py stop finetune/configs/my-run.yaml        # clean stop with a checkpoint
```

## What `submit.sh` does

| stage | where | what | skipped when |
|---|---|---|---|
| validate + dry run | portal | YAML errors, model config builds | never |
| `--test` (optional) | portal CPU | `local_test.sh`: the whole pipeline on a tiny model, ~5 min | without `--test` |
| prepare | CPU node | model → OLMo-core format, tokenize + mask data, pack | already prepared |
| fit check | GPU | `slurm.fit_check_steps` steps of the real model, no checkpoint | `--no-fit-check` |
| train | GPU | the run; resumes from its latest checkpoint if one exists | |
| export | CPU node | latest checkpoint → Hugging Face (`runs/<name>/hf-stepN`) | `export.hf: false` |

Each job waits for the previous one (`afterok`), so a failure never holds a GPU. Within a GPU
job, prechecks run in seconds before `torchrun`, and a watchdog ends the job if neither the log nor
the checkpoint folder changes for `slurm.stall_min` minutes.

## Models (`model:`)

| `source` | `path` | notes |
|---|---|---|
| `hf` | Hub id, e.g. `allenai/OLMo-2-1124-7B`, `allenai/Olmo-3-1025-7B` | downloaded once to `$OLMO_ROOT/hf/`, converted, logits validated |
| `hf_local` | a folder you uploaded (`config.json`, `*.safetensors`, tokenizer) | your own model; nothing fetched from the Hub |
| `olmo_core` | `$OLMO_ROOT/runs/<run>/stepN`, or a converted model folder | used in place: continue from an earlier fine-tune |
| `olmo_core` | URL, e.g. `https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage2/step47684` | AI2's intermediate checkpoints; only the weights are copied |

`arch` (`olmo2_7b`, `olmo3_7b`, `olmo2_1b_v2`, …) is required for `hf`/`hf_local`. Prepared
models live in `$OLMO_ROOT/models/` and are reused by every config that points at the same source.

Upload your own model or data from the Mac:

```bash
rsync -av ~/models/my-olmo/ dgg6b@portal.cs.virginia.edu:/bigtemp/dgg6b/my_models/my-olmo/
rsync -av ~/data/mydata.jsonl dgg6b@portal.cs.virginia.edu:/bigtemp/dgg6b/datasets/
```

## Data (`data:`)

A list of datasets that are mixed together. Each item is a Hub dataset id or a local
`.jsonl/.json/.csv/.parquet` file (or a folder of them), in one of three formats:

* `chat`: a messages list (`role`/`content`, or ShareGPT `from`/`value`). Loss on assistant turns.
* `prompt_response`: prompt + response columns (+ optional `input`). Loss on the response.
* `text`: plain text for continued pretraining. Loss on every token.

Column names are auto-detected (see `data_examples/README.md`); `num_examples` takes a seeded random
sample. Tokenized data goes to `$OLMO_ROOT/data/<name>-<hash>/`. The hash covers the tokenizer,
template, sequence length and data settings, so changing any of them re-tokenizes and unchanged
configs reuse the existing data. The control panel's **Data** tab shows the decoded examples with
the trained-on tokens highlighted. Check it before a long run.

The tokenizer must be dolma2-based (every OLMo 2 / OLMo 3 tokenizer is). Its chat template
formats chat and prompt/response data, so use the matching SFT/Instruct tokenizer, e.g.
`allenai/OLMo-2-1124-7B-SFT` or `allenai/Olmo-3-7B-Instruct-SFT`, or set `chat_template: olmo3`.

## Sizing (measured on 2× H100 NVL, OLMo-2 7B, job 6701929)

* 4096-token sequences × 32 per step = 131k tokens/step → **17.8 s/step**, 3,673 tokens/s/GPU, 84 GB/GPU
* 1,000 steps ≈ 5 h; one Tulu 3 epoch (~650M tokens) ≈ 24.5 h
* a checkpoint (with optimizer state) is ~88 GB and takes ~13 min to write → use `checkpoint.every`
  about once an hour of training; `keep: 1` deletes the older ones when the run finishes
* the HF export is ~15 GB (bf16)
* 1B models fit on one GPU (`slurm.gpus: 1`, `constraint: null`)
* `optim_foreach: true` runs out of memory for 7B on 2 GPUs (job 6701776); leave it off

## Folders on /bigtemp (`$OLMO_ROOT`, default `/bigtemp/$USER/olmo`)

```
models/<key>/        prepared base models (model_and_optim/ + config.json)
data/<key>/          tokenized data, manifest.json (stats + samples), pack.json, tokenizer/
runs/<name>/         stepN/ checkpoints, hf-stepN/ export, metrics.jsonl, status.json, events.jsonl,
                     jobs.jsonl, train-<job>.log, logs/<slurm job logs>
runs/<name>-fitcheck/
```

/bigtemp is scratch: not backed up, and files unused for 150 days are deleted. Copy finished models
somewhere safe (`launch_from_mac.sh … download`, then UVA Box).

## Testing changes

`bash finetune/scripts/local_test.sh` runs the full pipeline offline on CPU (15 checks, ~5 min):
all four data layouts, packing, training, resume, fit check, STOP, watchdog, HF export and the
control-panel data. Run it after editing anything in `finetune/`; `submit.sh --test` runs it before
submitting.

## Upload and start from the web UI

After the updated code is on the cluster and its environment is ready, open the live panel:

```bash
bash finetune/scripts/launch_from_mac.sh dgg6b panel
```

Authenticate in your terminal, wait for the server's ready message, then open the address below.
Keep this terminal open while uploading or using the UI. The panel owns a separate SSH connection
so its tunnel closes when it exits; already-submitted Slurm jobs keep running after disconnection.
`panel` uses the code already on the cluster and does **not** synchronize uncommitted Mac changes.
If port 8765 is occupied, close the earlier tunnel or launch with `PORT=8766` and use that port in
the browser. The launcher fails on a port-binding error rather than starting an unreachable panel.

Choose **New fine-tune**, or open `http://localhost:8765/launch.html` through that tunnel.
The page must be served by the cluster panel; opening an HTML file directly cannot upload or
submit jobs. A panel running on a Mac without Slurm supports uploads for testing but disables Start.

1. **Upload model**: choose a Hugging Face OLMo 2/3 folder containing `config.json`,
   `.safetensors` weights (and their index for sharded models), `tokenizer.json`, and
   `tokenizer_config.json`. The folder can also include the tokenizer's `chat_template.jinja`.
   Hidden/cache files and unrelated model-folder files are excluded by the browser.
2. **Upload dataset** separately: choose one or more `.jsonl`, `.json`, `.csv`, `.parquet`, or
   `.arrow` files. Use one file type per upload. The format and column mapping are selected below.
3. Select both completed uploads, choose a **new run name**, and verify the model architecture,
   dataset format, and conversation template. Training settings and custom columns are expandable.
   This is full-model fine-tuning, not QLoRA. GPU memory requirements depend on the selected model.
4. Click **Start fine-tuning**. The server saves the YAML, runs the existing submitter, and displays
   the resulting job IDs and submission log. The mandatory five-step GPU fit-check gates training,
   and Hugging Face export stays enabled. CPU preparation validates conversion and dataset parsing
   before GPUs are used. A failed submission can have already queued earlier pipeline jobs; inspect
   the displayed IDs before starting another run.
5. Follow loss, throughput, memory, and export paths in **Training dashboard**. `ft.py list` also
   includes web-created configs. Full exports remain under `$OLMO_ROOT/runs/<run>/hf-stepN/`.

Browser uploads use bounded 8 MiB chunks. **Pause** stops after the current chunk. To resume after a
network interruption or page reload, select the same unchanged files and click Upload again in the
same browser. Retain browser local storage until the upload finishes. A completed upload is checked
and SHA-256 hashed; each upload has a unique ID and immutable directory. Re-uploading a changed model
or dataset creates a new version. The server never uses an incomplete upload to start a run.
If the connection drops while the server checks a completed upload, resuming waits for that same
upload's result instead of transferring a second copy.

**Remove** deletes an unused upload after confirmation. Inputs referenced by a web-created run are
retained, including after a failed submission, to preserve reproducibility. It does not unload a model
from an active GPU job or delete training outputs. Stop jobs using the existing workflow.

New folders:

```text
$OLMO_ROOT/uploads/<id>/asset.json       upload state, file list, sizes, content hashes
$OLMO_ROOT/uploads/<id>/files/          uploaded files
$OLMO_ROOT/web_runs/<name>/run.yaml     generated run config
$OLMO_ROOT/web_runs/<name>/submission.json
$OLMO_ROOT/web_runs/<name>/submit.log   submission results; worker.log captures worker errors
```

Repeated Start requests for the same run are idempotent. Existing run names cannot be reused for a
fresh web launch. Resume an existing training run with its saved YAML through the CLI. Automatic
wall-time resubmission is not implemented in this version; select a suitable time limit and checkpoint
interval. For files managed outside the web UI, local file metadata changes also invalidate model,
tokenizer, and dataset preparation caches. Those metadata checks are not cryptographic integrity checks.
Shared model preparation is locked across runs. A failed conversion or validation leaves a marker
that prevents the partial checkpoint from being treated as ready; the next preparation retries it.

Only one live panel may use a given `$OLMO_ROOT` at a time; a filesystem lock prevents concurrent upload mutations.
The server binds only to `127.0.0.1`. Use the SSH tunnel; do not publish it as a public web service.
Mutating API requests require a session token and same-origin checks. No GitHub/SSH credentials are
entered into the web UI.

Web/API regression tests (standard library + PyYAML, no torch or GPU required):

```bash
python3 -B -m unittest discover -s finetune/tests -v
node --test finetune/tests/test_launch_ui.cjs
```

These tests cover HTTP uploads, pause/resume offsets, validation failures, immutable versions,
cache invalidation, duplicate launches, removal protection, SSH arguments, model-preparation
failures/concurrency, and the actual detached worker and fit-check dependency chain with fake Slurm
commands. The Node tests exercise browser upload recovery with DOM/fetch test doubles. They do not
replace a manual browser upload, the full CPU training test, or a real GPU validation.
