"""
Training: build the OLMo-core trainer from a RunConfig and run it (launched by torchrun).

Generalizes the feasibility-test script (uva/sft_olmo2_7b_uva.py, job 6701929):
  * model architecture comes from the prepared base model's config.json (OLMo 2 or OLMo 3)
  * resumes from the run's own latest checkpoint, else loads the base model weights
  * per-parameter AdamW by default (foreach AdamW OOMs on 2x H100 for 7B)
  * writes <run_dir>/metrics.jsonl and <run_dir>/status.json for the control panel
  * a file named STOP in the run folder ends the run cleanly at the next step (`ft.py stop`)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyPackedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

from .config import RunConfig
from .data import read_manifest
from .models import build_model_config

log = logging.getLogger(__name__)
CPU_TEST = bool(os.environ.get("FT_CPU_TEST"))


@dataclass
class FTExperimentConfig(Config):
    run_name: str
    model: TransformerConfig
    dataset: NumpyPackedFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 33333
    ft: Dict[str, Any] = field(default_factory=dict)  # the run YAML, for the record


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class RunMonitorCallback(Callback):
    """Rank 0 writes metrics.jsonl / status.json in the run folder; all ranks honor a STOP file."""

    run_dir: str = ""
    job_id: str = ""
    _status: Dict[str, Any] = field(default_factory=dict)
    _t0: float = 0.0

    def _write_status(self, **kw):
        if get_rank() != 0:
            return
        self._status.update(kw, updated_at=_now())
        p = Path(self.run_dir) / "status.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._status, indent=1))
        os.replace(tmp, p)

    def _event(self, kind: str, **kw):
        if get_rank() != 0:
            return
        with open(Path(self.run_dir) / "events.jsonl", "a") as f:
            f.write(json.dumps({"time": _now(), "event": kind, "step": self.step, **kw}) + "\n")

    def pre_train(self):
        self._t0 = time.time()
        self._write_status(
            state="running",
            job_id=self.job_id,
            started_at=_now(),
            step=self.step,
            max_steps=self.trainer.max_steps,
            world_size=get_world_size(),
            host=os.uname().nodename,
        )
        self._event("start", max_steps=self.trainer.max_steps, job_id=self.job_id)

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if get_rank() != 0:
            return
        row = {"step": step, "time": round(time.time(), 1)}
        row.update({k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()})
        with open(Path(self.run_dir) / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        self._write_status(step=step, elapsed_sec=round(time.time() - self._t0))

    def post_step(self):
        stop = Path(self.run_dir) / "STOP"
        if get_rank() == 0 and stop.exists() and not self.trainer.is_canceled:
            self._event("stop_requested")
            self.trainer.cancel_run("STOP file found in the run folder")

    def post_checkpoint_saved(self, path):
        self._event("checkpoint", path=str(path))
        self._write_status(last_checkpoint=str(path), last_checkpoint_step=self.step)

    def post_train(self):
        state = "stopped" if self.trainer.is_canceled else "finished"
        self._write_status(state=state, finished_at=_now(), step=self.step)
        self._event(state)

    def on_error(self, exc: BaseException):
        self._write_status(
            state="failed", error=f"{type(exc).__name__}: {exc}"[:500], finished_at=_now()
        )
        self._event("error", error=f"{type(exc).__name__}: {exc}"[:500])


def data_paths(cfg: RunConfig) -> tuple[List[str], List[str]]:
    man = read_manifest(cfg.data_dir)
    if not man:
        raise FileNotFoundError(
            f"no prepared data in {cfg.data_dir}: run `ft.py prepare-data {cfg.config_path}`"
        )
    ids, masks = [], []
    for d in man["datasets"]:
        ids.append(str(cfg.data_dir / d["files"][0]))
        masks.append(str(cfg.data_dir / d["files"][1]))
    return ids, masks


def dataset_config(
    cfg: RunConfig, tokenizer: TokenizerConfig, need_data: bool = True
) -> NumpyPackedFSLDatasetConfig:
    if need_data:
        ids, masks = data_paths(cfg)
    else:  # dry run before the data exists
        ids, masks = [str(cfg.data_dir / "x.npy")], [str(cfg.data_dir / "y.npy")]
    return NumpyPackedFSLDatasetConfig(
        tokenizer=tokenizer,
        work_dir=str(cfg.pack_cache_dir),
        paths=ids,
        label_mask_paths=masks,
        # intra-document attention masking (no attention across packed examples) needs flash-attn
        generate_doc_lengths=cfg.train.attn_backend.startswith("flash"),
        # chat / prompt_response docs are <= seq_len already (prepare-data skips or truncates them);
        # plain-text docs longer than seq_len are split into pieces instead of being cut off
        long_doc_strategy=LongDocStrategy.fragment,
        sequence_length=cfg.train.seq_len,
    )


def build_experiment(
    cfg: RunConfig,
    run_name: Optional[str] = None,
    steps: Optional[int] = None,
    world_size: Optional[int] = None,
    overrides: Optional[List[str]] = None,
    need_data: bool = True,
) -> FTExperimentConfig:
    t = cfg.train
    run_name = run_name or cfg.name
    run_dir = cfg.run_dir(run_name)
    world_size = world_size or get_world_size()
    model, tokenizer = build_model_config(cfg)
    if not isinstance(tokenizer, TokenizerConfig):
        tokenizer = TokenizerConfig.dolma2()

    dataset = dataset_config(cfg, tokenizer, need_data)

    micro_tokens = t.microbatch_seqs * t.seq_len
    global_tokens = t.global_batch_seqs * t.seq_len
    if global_tokens % (micro_tokens * world_size):
        raise ValueError(
            f"global_batch_seqs={t.global_batch_seqs} not divisible by "
            f"microbatch_seqs*world_size={t.microbatch_seqs}*{world_size}"
        )

    if steps is not None:
        duration = Duration.steps(steps)
    elif t.steps is not None:
        duration = Duration.steps(t.steps)
    else:
        duration = Duration.epochs(int(t.epochs))

    ck = cfg.checkpoint
    ckpt_enabled = bool(ck.final or ck.every)
    ac = {
        "ffn": TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        ),
        "full": TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
        ),
        "none": None,
    }[t.activation_checkpointing]

    trainer = (
        TrainerConfig(
            save_folder=str(run_dir),
            load_strategy=LoadStrategy.never,  # resume / base-model load handled in train()
            save_overwrite=True,
            no_checkpoints=not ckpt_enabled,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=duration,
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback(
            "monitor",
            RunMonitorCallback(run_dir=str(run_dir), job_id=os.environ.get("SLURM_JOB_ID", "")),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                enabled=ckpt_enabled,
                save_interval=None,  # only the final checkpoint is permanent
                ephemeral_save_interval=ck.every or None,  # overwritten; for crash/timeout resume
                pre_train_checkpoint=False,
                max_checkpoints=ck.keep,
                save_async=False,
            ),
        )
    )
    if not CPU_TEST:
        trainer = trainer.with_callback("gpu_monitor", GPUMemoryMonitorCallback())

    exp = FTExperimentConfig(
        run_name=run_name,
        model=model,
        dataset=dataset,
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=global_tokens,
            seed=t.seed + 1,
            num_workers=0 if CPU_TEST else 4,
            work_dir=str(cfg.pack_cache_dir),
        ),
        train_module=TransformerTrainModuleConfig(
            rank_microbatch_size=micro_tokens,
            max_sequence_length=t.seq_len,
            z_loss_multiplier=None,
            compile_model=t.compile,
            optim=SkipStepAdamWConfig(
                lr=t.lr,
                weight_decay=t.weight_decay,
                betas=tuple(t.betas),  # type: ignore
                compile=False,
                foreach=t.optim_foreach,
            ),
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
            ),
            ac_config=ac,
            scheduler=LinearWithWarmup(warmup_fraction=t.warmup_fraction, alpha_f=0.0),
            max_grad_norm=t.max_grad_norm,
        ),
        trainer=trainer,
        init_seed=t.seed,
        ft=cfg.to_dict(),
    )
    if overrides:
        exp = exp.merge(overrides)
    return exp


def prune_checkpoints(run_dir: Path, keep: int) -> None:
    """After a successful run keep only the newest `keep` checkpoints (a 7B checkpoint with optimizer
    state is ~88 GB; the crash-resume ones are useless once the run has finished)."""
    import shutil

    from .export import checkpoints

    for old in checkpoints(run_dir)[: -max(keep, 1)]:
        log.info(f"Removing intermediate checkpoint {old}")
        shutil.rmtree(old, ignore_errors=True)


def pack(cfg: RunConfig) -> int:
    """Single-process packing of the tokenized data into seq_len instances (cached in <data_dir>/pack-cache), so
    the multi-GPU job never builds indices itself (a failure there left ranks hanging, job 6701234).
    """
    from .models import read_model_meta

    tok = read_model_meta(cfg.model_dir).get("dataset", {}).get("tokenizer")
    ds = dataset_config(
        cfg, TokenizerConfig.from_dict(tok) if tok else TokenizerConfig.dolma2()
    ).build()
    ds.prepare()
    n = len(ds)
    info = {
        "instances": n,
        "seq_len": cfg.train.seq_len,
        "tokens": n * cfg.train.seq_len,
        "steps_per_epoch": n // cfg.train.global_batch_seqs,
        "packed_at": _now(),
    }
    with open(cfg.data_dir / "pack.json", "w") as f:
        json.dump(info, f, indent=1)
    log.info(
        f"Packed: {n:,d} instances of {cfg.train.seq_len} tokens "
        f"= {info['steps_per_epoch']} steps/epoch at {cfg.train.global_batch_seqs} seqs/step"
    )
    if info["steps_per_epoch"] == 0:
        raise RuntimeError("less than one batch of data: add data or lower train.global_batch_seqs")
    return n


def train(
    cfg: RunConfig,
    run_name: Optional[str] = None,
    steps: Optional[int] = None,
    overrides: Optional[List[str]] = None,
    fit_check: bool = False,
) -> None:
    """fit_check: a few steps of the real model on the real GPUs, no checkpoints, in a separate
    <name>-fitcheck folder; catches OOM / NCCL / data problems before the long job."""
    overrides = list(overrides or [])
    if fit_check:
        run_name = f"{cfg.name}-fitcheck"
        steps = steps or cfg.slurm.fit_check_steps or 5
        overrides += [
            "--trainer.no_checkpoints=true",
            "--trainer.callbacks.checkpointer.enabled=false",
        ]
    from olmo_core.train import prepare_training_environment, teardown_training_environment

    timeout = timedelta(minutes=cfg.train.dist_timeout_min)
    if CPU_TEST:
        prepare_training_environment(backend="gloo", timeout=timeout)
    else:
        prepare_training_environment(timeout=timeout)
    try:
        exp = build_experiment(cfg, run_name=run_name, steps=steps, overrides=overrides)
        run_dir = Path(exp.trainer.save_folder)
        if get_rank() == 0:
            if fit_check:
                import shutil

                shutil.rmtree(run_dir, ignore_errors=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "STOP").unlink(missing_ok=True)
            with open(run_dir / "run_config.json", "w") as f:
                json.dump(
                    {
                        "config_path": cfg.config_path,
                        "run_name": exp.run_name,
                        "steps_override": steps,
                        "fit_check": fit_check,
                        "data_dir": str(cfg.data_dir),
                        "model_dir": str(cfg.model_dir),
                        **cfg.to_dict(),
                    },
                    f,
                    indent=1,
                )
            log.info(f"Experiment config:\n{exp}")
        seed_all(exp.init_seed)

        if os.environ.get("FT_TEST_HANG_SEC"):  # test hook for the watchdog
            log.warning("FT_TEST_HANG_SEC set: simulating a hang")
            time.sleep(float(os.environ["FT_TEST_HANG_SEC"]))

        model = exp.model.build(init_device="meta")
        train_module = exp.train_module.build(model)
        dataset = exp.dataset.build()
        data_loader = exp.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
        trainer = exp.trainer.build(train_module, data_loader)
        cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = exp.as_config_dict()

        ckpt_on = exp.trainer.callbacks["checkpointer"].enabled  # type: ignore[attr-defined]
        if ckpt_on and trainer.maybe_load_checkpoint(trainer.save_folder):
            log.info(
                f"RESUMED from checkpoint in '{trainer.save_folder}' at step {trainer.global_step}"
            )
        else:
            src = str(
                cfg.model_dir / "model_and_optim"
            )  # model-only load: no optimizer / trainer state
            log.info(f"Loading base model weights from '{src}'")
            trainer.load_checkpoint(src, load_trainer_state=False, load_optim_state=False)
        trainer.fit()
        if get_rank() == 0 and not trainer.is_canceled:
            prune_checkpoints(run_dir, cfg.checkpoint.keep)
    finally:
        teardown_training_environment()
