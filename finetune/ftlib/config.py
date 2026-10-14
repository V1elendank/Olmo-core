"""
Run configuration: one YAML file per fine-tuning run (see finetune/configs/_template.yaml).

The YAML is parsed into plain dataclasses, validated, and resolved into concrete paths under
$OLMO_ROOT (default /bigtemp/$USER/olmo):

    $OLMO_ROOT/models/<model.key>/      prepared base model (OLMo-core format, model only)
    $OLMO_ROOT/data/<data key>/         tokenized + label-masked training data
    $OLMO_ROOT/runs/<name>/             checkpoints, logs, metrics for this run
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

MODEL_SOURCES = ("hf", "hf_local", "olmo_core")
DATA_FORMATS = ("chat", "prompt_response", "text")
# Architectures OLMo-core's HF converter knows (src/examples/huggingface/convert_checkpoint_from_hf.py)
ARCHS = (
    "olmo2_190m",
    "olmo2_370m",
    "olmo2_600m",
    "olmo2_760m",
    "olmo2_1b",
    "olmo2_1b_v2",
    "olmo2_3b",
    "olmo2_7b",
    "olmo2_13b",
    "olmo2_32b",
    "olmo3_190m",
    "olmo3_7b",
    "olmo3_32b",
    "tiny",  # ~55M-param test model, CPU tests only
)


class ConfigError(ValueError):
    pass


def _slug(s: str) -> str:
    s = re.sub(r"^https?://", "", s.strip().rstrip("/"))
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[-80:]


def local_revision(source: str) -> str:
    """Fingerprint local file versions without rereading multi-GB weights per Slurm stage.

    Browser uploads have unique immutable paths and SHA-256 manifests. For external local
    paths, inode/size/mtime/ctime changes invalidate caches when a file is replaced or edited.
    This metadata fingerprint is not a cryptographic integrity check.
    """
    path = Path(source).expanduser()
    if not path.exists():
        return ""
    files = (
        [path]
        if path.is_file()
        else sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts)
        )
    )
    records = []
    for p in files:
        st = p.stat()
        records.append(
            (
                str(p.relative_to(path)) if p != path else p.name,
                st.st_ino,
                st.st_size,
                st.st_mtime_ns,
                st.st_ctime_ns,
            )
        )
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()[:16]


def config_paths() -> List[Path]:
    """Include browser-created runs alongside the checked-in YAML configs."""
    here = Path(__file__).resolve().parents[1]
    root = Path(os.environ.get("OLMO_ROOT") or f"/bigtemp/{os.environ.get('USER', 'user')}/olmo")
    return [
        p.resolve()
        for p in sorted((here / "configs").glob("*.yaml"))
        + sorted((root / "web_runs").glob("*/run.yaml"))
    ]


@dataclass
class ModelCfg:
    source: str  # hf | hf_local | olmo_core
    path: str  # HF id, local HF folder, or OLMo-core checkpoint dir / URL
    # needed for hf/hf_local; optional for olmo_core (read from config.json)
    arch: Optional[str] = None
    revision: str = "main"  # hf only
    validate: bool = True  # compare logits after HF -> OLMo-core conversion
    key: Optional[str] = None  # folder name under $OLMO_ROOT/models (derived if empty)


@dataclass
class DatasetCfg:
    name: str
    source: str  # HF dataset id or local .jsonl/.json/.parquet/.csv file (or folder)
    format: str = "chat"  # chat | prompt_response | text
    split: str = "train"
    subset: Optional[str] = None  # HF config name, e.g. "default"
    num_examples: Optional[int] = None  # None = all
    seed: int = 0
    # column names (auto-detected when left empty)
    messages_column: Optional[str] = None
    prompt_column: Optional[str] = None
    # optional extra context appended to the prompt (Alpaca "input")
    input_column: Optional[str] = None
    response_column: Optional[str] = None
    text_column: Optional[str] = None
    # chat / prompt_response: added when a conversation has none
    system_prompt: Optional[str] = None
    too_long: str = "skip"  # chat / prompt_response examples longer than seq_len: skip | truncate


@dataclass
class TrainCfg:
    seq_len: int = 4096
    global_batch_seqs: int = 32
    microbatch_seqs: int = 1
    steps: Optional[int] = None  # optimizer steps ...
    # ... or epochs (passes over the data); set exactly one
    epochs: Optional[int] = None
    lr: float = 2e-5
    warmup_fraction: float = 0.1
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    max_grad_norm: float = 1.0
    attn_backend: str = "torch"  # torch | flash_2 | flash_3
    activation_checkpointing: str = "ffn"  # ffn | full | none
    optim_foreach: bool = False  # faster AdamW, needs ~30-45 GB extra per GPU for 7B on 2 GPUs
    compile: bool = False
    seed: int = 33333
    dist_timeout_min: float = 10


@dataclass
class CheckpointCfg:
    every: int = 0  # save a checkpoint every N steps so a crashed/timed-out job resumes; 0 = off
    final: bool = True  # save a checkpoint at the end (needed for export)
    keep: int = (
        1  # checkpoints kept once the run has finished (7B: ~88 GB each with optimizer state)
    )


@dataclass
class SlurmCfg:
    partition: str = "gpu"
    gpus: int = 2
    constraint: Optional[str] = "h100_94gb"
    cpus: int = 16
    mem: int = 240000  # MiB
    time: str = "04:00:00"
    mail: Optional[str] = None  # must be @virginia.edu
    fit_check_steps: int = 5  # short run before the real one; 0 = skip
    stall_min: int = 12  # watchdog


@dataclass
class ExportCfg:
    hf: bool = True  # convert the final checkpoint to Hugging Face format after training
    dtype: str = "bfloat16"
    validate: bool = (
        False  # compare logits of the HF export vs the checkpoint (needs ~2x model RAM)
    )
    max_seq_len: Optional[int] = None  # HF max_position_embeddings; default from the tokenizer


@dataclass
class RunConfig:
    name: str
    model: ModelCfg
    tokenizer: str  # HF id or local folder; must be dolma2-compatible; provides the chat template
    data: List[DatasetCfg]
    train: TrainCfg = field(default_factory=TrainCfg)
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    slurm: SlurmCfg = field(default_factory=SlurmCfg)
    export: ExportCfg = field(default_factory=ExportCfg)
    # "olmo2", "olmo3" or a Jinja string; overrides the tokenizer's template
    chat_template: Optional[str] = None
    notes: str = ""
    config_path: Optional[str] = None

    # ------------------------------------------------------------------ resolved paths
    @property
    def root(self) -> Path:
        return Path(
            os.environ.get("OLMO_ROOT") or f"/bigtemp/{os.environ.get('USER', 'user')}/olmo"
        )

    @property
    def model_key(self) -> str:
        if self.model.source == "hf_local":
            base = self.model.key or _slug(self.model.path)
            return f"{base}-{self.model.arch}-{local_revision(self.model.path)}"
        if self.model.key:
            return self.model.key
        rev = "" if self.model.revision in ("main", None) else f"@{self.model.revision}"
        return _slug(self.model.path + rev)

    @property
    def model_dir(self) -> Path:
        """Where the prepared (model-only, OLMo-core format) base model lives."""
        if self.model.source == "olmo_core" and not self.model.path.startswith(
            ("http://", "https://")
        ):
            return Path(self.model.path)  # use local OLMo-core checkpoints in place
        return self.root / "models" / self.model_key

    @property
    def data_key(self) -> str:
        blob = json.dumps(
            {
                "tok": self.tokenizer,
                "tmpl": self.chat_template,
                "seq": self.train.seq_len,
                "data": [dataclasses.asdict(d) for d in self.data],
                "local_tokenizer_version": local_revision(self.tokenizer),
                "local_data_versions": [local_revision(d.source) for d in self.data],
            },
            sort_keys=True,
        )
        return f"{_slug(self.name)}-{hashlib.sha1(blob.encode()).hexdigest()[:8]}"

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / self.data_key

    def run_dir(self, run_name: Optional[str] = None) -> Path:
        return self.root / "runs" / (run_name or self.name)

    @property
    def pack_cache_dir(self) -> Path:
        """Packing / data-order indices; belong to this exact tokenized data, rebuilt with it."""
        return self.data_dir / "pack-cache"

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("config_path", None)
        return d


def _build(cls, raw: Any, where: str):
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}")
    return cls(**raw)


def load_config(path: str | os.PathLike) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    for req in ("name", "model", "tokenizer", "data"):
        if req not in raw:
            raise ConfigError(f"missing required key '{req}'")
    data = raw["data"]
    if isinstance(data, dict):
        data = [data]
    cfg = RunConfig(
        name=str(raw["name"]),
        model=_build(ModelCfg, raw["model"], "model"),
        tokenizer=str(raw["tokenizer"]),
        data=[_build(DatasetCfg, d, f"data[{i}]") for i, d in enumerate(data)],
        train=_build(TrainCfg, raw.get("train"), "train"),
        checkpoint=_build(CheckpointCfg, raw.get("checkpoint"), "checkpoint"),
        slurm=_build(SlurmCfg, raw.get("slurm"), "slurm"),
        export=_build(ExportCfg, raw.get("export"), "export"),
        chat_template=raw.get("chat_template"),
        notes=str(raw.get("notes") or ""),
        config_path=str(Path(path).resolve()),
    )
    extra = set(raw) - {f.name for f in dataclasses.fields(RunConfig)}
    if extra:
        raise ConfigError(f"unknown top-level key(s) {sorted(extra)}")
    validate(cfg)
    return cfg


def validate(cfg: RunConfig) -> None:
    errs: List[str] = []
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cfg.name):
        errs.append("name: use only letters, digits, '.', '_' and '-'")
    m = cfg.model
    if m.source not in MODEL_SOURCES:
        errs.append(f"model.source must be one of {MODEL_SOURCES}")
    if m.arch is not None and m.arch not in ARCHS:
        errs.append(f"model.arch '{m.arch}' unknown; choose from {ARCHS}")
    if m.source in ("hf", "hf_local") and not m.arch:
        errs.append("model.arch is required for hf / hf_local sources (e.g. olmo2_7b, olmo3_7b)")
    if (
        m.source == "hf_local"
        and not Path(m.path).is_dir()
        and not os.environ.get("FT_SKIP_PATH_CHECKS")
    ):
        errs.append(f"model.path '{m.path}' is not a folder on this machine")
    if not cfg.data:
        errs.append("data: at least one dataset is required")
    names = [d.name for d in cfg.data]
    if len(set(names)) != len(names):
        errs.append("data: dataset names must be unique")
    for i, d in enumerate(cfg.data):
        if d.format not in DATA_FORMATS:
            errs.append(f"data[{i}].format must be one of {DATA_FORMATS}")
        if d.too_long not in ("skip", "truncate"):
            errs.append(f"data[{i}].too_long must be skip or truncate")
        if d.num_examples is not None and d.num_examples <= 0:
            errs.append(f"data[{i}].num_examples must be positive or null")
    t = cfg.train
    if t.seq_len <= 0 or t.seq_len & (t.seq_len - 1):
        errs.append("train.seq_len must be a power of two (e.g. 2048, 4096, 8192)")
    if (t.steps is None) == (t.epochs is None):
        errs.append("train: set exactly one of steps / epochs")
    if t.microbatch_seqs <= 0 or cfg.slurm.gpus <= 0 or t.global_batch_seqs <= 0:
        errs.append("batch sizes and GPU count must be positive")
    elif t.global_batch_seqs % (t.microbatch_seqs * cfg.slurm.gpus):
        errs.append("train.global_batch_seqs must be divisible by microbatch_seqs * slurm.gpus")
    if t.attn_backend not in ("torch", "flash_2", "flash_3"):
        errs.append("train.attn_backend must be torch, flash_2 or flash_3")
    if t.activation_checkpointing not in ("ffn", "full", "none"):
        errs.append("train.activation_checkpointing must be ffn, full or none")
    if cfg.slurm.mail and not cfg.slurm.mail.endswith("@virginia.edu"):
        errs.append("slurm.mail must be a @virginia.edu address (cluster policy)")
    if errs:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(errs))


def shell_exports(cfg: RunConfig) -> str:
    """Lines for `eval "$(python finetune/ft.py paths CONFIG)"` in the Slurm/submit scripts."""
    s = cfg.slurm
    vals = {
        "FT_NAME": cfg.name,
        "FT_MODEL_DIR": str(cfg.model_dir),
        "FT_DATA_DIR": str(cfg.data_dir),
        "FT_RUN_DIR": str(cfg.run_dir()),
        "FT_PARTITION": s.partition,
        "FT_GPUS": str(s.gpus),
        "FT_CONSTRAINT": s.constraint or "",
        "FT_CPUS": str(s.cpus),
        "FT_MEM": str(s.mem),
        "FT_TIME": s.time,
        "FT_MAIL": s.mail or "",
        "FT_FIT_STEPS": str(s.fit_check_steps),
        "FT_STALL_MIN": str(s.stall_min),
        "FT_EXPORT_HF": "1" if cfg.export.hf else "0",
    }
    import shlex

    return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in vals.items())
