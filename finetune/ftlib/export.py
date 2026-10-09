"""Export a run's checkpoint to Hugging Face format (bf16 safetensors + tokenizer with chat template)."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .config import RunConfig

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
TO_HF = REPO / "src" / "examples" / "huggingface" / "convert_checkpoint_to_hf.py"


def checkpoints(run_dir: Path) -> list[Path]:
    steps = [
        p
        for p in run_dir.glob("step*")
        if re.fullmatch(r"step\d+", p.name) and (p / "model_and_optim" / ".metadata").is_file()
    ]
    return sorted(steps, key=lambda p: int(p.name[4:]))


def default_context(ck: Path) -> int:
    """max_position_embeddings for the HF config: OLMo 2 = 4096, OLMo 3 = 8192, times any YaRN factor.
    (The HF -> OLMo-core converter refuses other values for OLMo 2, so re-importing stays possible.)
    """
    try:
        model = json.loads((ck / "config.json").read_text())["model"]
    except (OSError, ValueError, KeyError):
        return 4096
    rope = json.dumps(model)
    m = re.search(r'"old_context_len": (\d+)', rope)
    f = re.search(r'"factor": ([\d.]+)', rope)
    if m and f:
        return int(int(m.group(1)) * float(f.group(1)))
    return 8192 if '"sliding_window": {' in rope else 4096


def export_hf(
    cfg: RunConfig,
    run_name: Optional[str] = None,
    step: Optional[int] = None,
    out: Optional[Path] = None,
) -> Path:
    run_dir = cfg.run_dir(run_name)
    cks = checkpoints(run_dir)
    if step is not None:
        cks = [c for c in cks if c.name == f"step{step}"]
    if not cks:
        raise FileNotFoundError(f"no checkpoint {'step' + str(step) if step else ''} in {run_dir}")
    ck = cks[-1]
    out = out or run_dir / f"hf-{ck.name}"
    tok_dir = cfg.data_dir / "tokenizer"
    cmd = [
        sys.executable,
        str(TO_HF),
        "-i",
        str(ck),
        "-o",
        str(out),
        "--dtype",
        cfg.export.dtype,
        "-t",
        str(tok_dir) if tok_dir.is_dir() else cfg.tokenizer,
    ]
    cmd += ["-s", str(cfg.export.max_seq_len or default_context(ck))]
    if not cfg.export.validate:
        cmd.append("--skip-validation")
    log.info("Exporting: " + " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=REPO)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    info = {
        "checkpoint": str(ck),
        "hf_dir": str(out),
        "bytes": size,
        "seconds": round(time.time() - t0),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(run_dir / "export.json", "w") as f:
        json.dump(info, f, indent=1)
    log.info(f"HF model: {out} ({size / 1e9:.1f} GB)")
    return out
