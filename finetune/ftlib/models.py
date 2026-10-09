"""
Base-model handling: get any supported starting model into a local, model-only OLMo-core
checkpoint, and rebuild its exact architecture for training.

Sources (model.source in the YAML):
  hf        Hugging Face Hub id (e.g. allenai/OLMo-2-1124-7B)  -> download -> convert
  hf_local  a Hugging Face-format folder you uploaded (config.json + *.safetensors) -> convert
  olmo_core an OLMo-core checkpoint dir (one of your earlier runs, e.g. runs/x/step30) used in
            place, or a URL (e.g. AI2's https://olmo-checkpoints.org/... stage checkpoints) from
            which only the model weights are copied
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from .config import RunConfig

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
HF_CONVERTER = REPO / "src" / "examples" / "huggingface" / "convert_checkpoint_from_hf.py"


def _arch_factory(arch: str):
    from olmo_core.nn.transformer import TransformerBlockType, TransformerConfig

    if arch == "tiny":
        return lambda vocab_size, **kw: TransformerConfig.llama_like(
            d_model=256,
            n_layers=2,
            n_heads=4,
            vocab_size=vocab_size,
            hidden_size_multiplier=1.5,
            block_name=TransformerBlockType.reordered_norm,
            qk_norm=True,
            layer_norm_eps=1e-6,
            **kw,
        )
    family, size = arch.split("_", 1)  # e.g. olmo2_7b -> olmo2_7B, olmo2_1b_v2 -> olmo2_1B_v2
    size = size.replace("b", "B", 1) if "b" in size else size.replace("m", "M", 1)
    return getattr(TransformerConfig, f"{family}_{size}")


def is_prepared(path: Path) -> bool:
    return (path / "model_and_optim" / ".metadata").is_file()


def read_model_meta(model_dir: Path) -> Dict[str, Any]:
    p = model_dir / "config.json"
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    return {}


def build_model_config(cfg: RunConfig):
    """TransformerConfig for training: taken from the prepared checkpoint's config.json when present
    (so RoPE scaling, sliding windows, etc. match exactly), else from model.arch."""
    from olmo_core.data import TokenizerConfig
    from olmo_core.nn.transformer import TransformerConfig

    meta = read_model_meta(cfg.model_dir)
    if "model" in meta:
        d = json.loads(json.dumps(meta["model"]))
        _set_attn_backend(d, cfg.train.attn_backend)
        model = TransformerConfig.from_dict(d)
    else:
        if not cfg.model.arch:
            raise RuntimeError(f"{cfg.model_dir}/config.json has no model config; set model.arch")
        from olmo_core.nn.attention import AttentionBackendName

        model = _arch_factory(cfg.model.arch)(
            vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
            attn_backend=AttentionBackendName(cfg.train.attn_backend),
        )
    tok = meta.get("dataset", {}).get("tokenizer")
    tokenizer = TokenizerConfig.from_dict(tok) if tok else TokenizerConfig.dolma2()
    return model, tokenizer


def _set_attn_backend(d: Any, backend: str) -> None:
    if isinstance(d, dict):
        for k, v in list(d.items()):
            if k == "backend" and isinstance(v, (str, type(None))):
                d[k] = backend
            else:
                _set_attn_backend(v, backend)
    elif isinstance(d, list):
        for v in d:
            _set_attn_backend(v, backend)


def prepare_model(cfg: RunConfig, force: bool = False) -> Path:
    m, out = cfg.model, cfg.model_dir
    if is_prepared(out) and not force:
        log.info(f"Model already prepared: {out}")
        return out
    t0 = time.time()
    if m.source == "olmo_core" and not m.path.startswith(("http://", "https://")):
        if not is_prepared(out):
            raise FileNotFoundError(
                f"{out}/model_and_optim/.metadata not found (not an OLMo-core checkpoint)"
            )
        return out
    out.mkdir(parents=True, exist_ok=True)
    if m.source == "hf":
        from huggingface_hub import snapshot_download

        hf_dir = cfg.root / "hf" / cfg.model_key
        log.info(f"Downloading {m.path}@{m.revision} -> {hf_dir}")
        snapshot_download(
            m.path,
            revision=m.revision,
            local_dir=str(hf_dir),
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        )
        _convert_hf(hf_dir, m.arch, out, m.validate)
    elif m.source == "hf_local":
        _convert_hf(Path(m.path), m.arch, out, m.validate)
    else:
        _fetch_olmo_core(m.path, out, m.arch)
    meta_path = out / "ft_source.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "source": m.source,
                "path": m.path,
                "revision": m.revision,
                "arch": m.arch,
                "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seconds": round(time.time() - t0),
            },
            f,
            indent=2,
        )
    log.info(f"Prepared {out} in {time.time() - t0:.0f}s")
    return out


def _convert_hf(hf_dir: Path, arch: str, out: Path, validate: bool) -> None:
    if arch == "tiny":
        raise ValueError("arch 'tiny' cannot be converted from HF")
    cmd = [
        sys.executable,
        str(HF_CONVERTER),
        "-i",
        str(hf_dir),
        "-m",
        arch,
        "-t",
        "dolma2",
        "-o",
        str(out),
    ]
    if not validate:
        cmd.append("--skip-validation")
    log.info("Converting HF -> OLMo-core: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)


def _fetch_olmo_core(src: str, out: Path, arch: str | None) -> None:
    """Copy only the model tensors (no optimizer state) of a remote OLMo-core checkpoint."""
    import torch
    from cached_path import cached_path

    from olmo_core.data import TokenizerConfig
    from olmo_core.distributed.checkpoint import (
        load_model_and_optim_state,
        save_model_and_optim_state,
    )
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig

    src = src.rstrip("/")
    meta: Dict[str, Any] = {}
    try:
        with cached_path(f"{src}/config.json").open() as f:
            meta = json.load(f)
        log.info("Read model config from source config.json")
    except Exception as e:  # noqa: BLE001
        log.warning(f"No usable {src}/config.json ({e}); falling back to model.arch")
    tok = (
        TokenizerConfig.from_dict(meta["dataset"]["tokenizer"])
        if meta.get("dataset", {}).get("tokenizer")
        else TokenizerConfig.dolma2()
    )
    if "model" in meta:
        d = json.loads(json.dumps(meta["model"]))
        _set_attn_backend(d, "torch")
        model_config = TransformerConfig.from_dict(d)
    elif arch:
        model_config = _arch_factory(arch)(
            vocab_size=tok.padded_vocab_size(), attn_backend=AttentionBackendName.torch
        )
    else:
        raise RuntimeError("source has no config.json; set model.arch")
    model = model_config.build(init_device="cpu")
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))  # anything not loaded stays NaN and is caught below
    log.info(
        f"Loading model weights ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params) "
        f"from {src}/model_and_optim (optimizer state skipped)"
    )
    load_model_and_optim_state(f"{src}/model_and_optim", model, thread_count=16)
    bad = [k for k, v in model.state_dict().items() if torch.isnan(v).any()]
    if bad:
        raise RuntimeError(f"{len(bad)} tensors not loaded, e.g. {bad[:3]}")
    save_model_and_optim_state(str(out / "model_and_optim"), model, save_overwrite=True)
    with open(out / "config.json", "w") as f:
        json.dump(
            {
                "model": model_config.as_config_dict(),
                "dataset": {"tokenizer": tok.as_config_dict()},
                "source": src,
            },
            f,
            indent=2,
        )


def save_test_checkpoint(arch: str, out: Path) -> None:
    """Random-init checkpoint in OLMo-core format (CPU tests)."""
    from olmo_core.data import TokenizerConfig
    from olmo_core.distributed.checkpoint import save_model_and_optim_state
    from olmo_core.nn.attention import AttentionBackendName

    tok = TokenizerConfig.dolma2()
    mc = _arch_factory(arch)(
        vocab_size=tok.padded_vocab_size(), attn_backend=AttentionBackendName.torch
    )
    model = mc.build(init_device="cpu")
    out.mkdir(parents=True, exist_ok=True)
    save_model_and_optim_state(str(out / "model_and_optim"), model, save_overwrite=True)
    with open(out / "config.json", "w") as f:
        json.dump({"model": mc.as_config_dict(), "dataset": {"tokenizer": tok.as_config_dict()}}, f)


if __name__ == "__main__":  # pragma: no cover
    print("use finetune/ft.py", file=sys.stderr)
    os._exit(2)
