#!/usr/bin/env python
"""
ft.py: one entry point for fine-tuning OLMo 2 / OLMo 3 models from a YAML run config.

    python finetune/ft.py new my-run                  # configs/my-run.yaml from the template
    python finetune/ft.py validate CONFIG             # check the YAML, show resolved paths/state
    python finetune/ft.py prepare CONFIG              # base model + data + packing (CPU)
        prepare-model / prepare-data / pack           #   ...or one step at a time
    python finetune/ft.py dry-run CONFIG              # print the full training config (no GPU)
    torchrun ... finetune/ft.py train CONFIG          # (run by slurm/train.sbatch)
    python finetune/ft.py export CONFIG               # last checkpoint -> Hugging Face format
    python finetune/ft.py stop CONFIG                 # ask a running job to stop after this step
    python finetune/ft.py status [--serve 8765]       # build / serve the control panel data
    python finetune/ft.py list                        # all configs and how far each has got

Normally you only run:   bash finetune/scripts/submit.sh finetune/configs/<run>.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ftlib.config import ConfigError, config_paths, load_config, shell_exports  # noqa: E402

log = logging.getLogger("ft")


def _setup_logging():
    if os.environ.get("RANK") is None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )


def cmd_new(a):
    dst = HERE / "configs" / f"{a.name}.yaml"
    if dst.exists():
        sys.exit(f"{dst} already exists")
    src = Path(a.template) if a.template else HERE / "configs" / "_template.yaml"
    text = src.read_text()
    import re

    text = re.sub(r"(?m)^name:.*$", f"name: {a.name}", text, count=1)
    dst.write_text(text)
    print(
        f"created {dst.relative_to(HERE.parent)}  (edit it, then: bash finetune/scripts/submit.sh {dst.relative_to(HERE.parent)})"
    )


def _state(cfg):
    from ftlib.data import read_manifest
    from ftlib.export import checkpoints
    from ftlib.models import is_prepared

    run = cfg.run_dir()
    st = {}
    if (run / "status.json").is_file():
        st = json.loads((run / "status.json").read_text())
    return {
        "model_ready": is_prepared(cfg.model_dir),
        "data_ready": bool(read_manifest(cfg.data_dir)),
        "packed": (cfg.data_dir / "pack.json").is_file(),
        "checkpoints": [c.name for c in checkpoints(run)] if run.is_dir() else [],
        "run_state": st.get("state", "not started"),
        "step": st.get("step"),
    }


def cmd_validate(a):
    cfg = load_config(a.config)
    s = _state(cfg)
    print(f"OK  {a.config}")
    print(f"  name        {cfg.name}")
    print(
        f"  model       {cfg.model.source}: {cfg.model.path}  (arch {cfg.model.arch or 'from config.json'})"
    )
    print(f"  model dir   {cfg.model_dir}   [{'ready' if s['model_ready'] else 'not prepared'}]")
    for d in cfg.data:
        n = d.num_examples or "all"
        print(f"  data        {d.name}: {d.source} ({d.format}, {n} examples)")
    print(
        f"  data dir    {cfg.data_dir}   [{'ready' if s['data_ready'] else 'not prepared'}"
        f"{', packed' if s['packed'] else ''}]"
    )
    print(
        f"  run dir     {cfg.run_dir()}   [{s['run_state']}"
        f"{', step ' + str(s['step']) if s['step'] else ''}{', ' + ','.join(s['checkpoints']) if s['checkpoints'] else ''}]"
    )
    t = cfg.train
    dur = f"{t.steps} steps" if t.steps else f"{t.epochs} epoch(s)"
    print(
        f"  training    {dur}, {t.global_batch_seqs} x {t.seq_len} tokens/step, lr {t.lr}, "
        f"{cfg.slurm.gpus} GPU(s) {cfg.slurm.constraint or ''}"
    )


def cmd_paths(a):
    print(shell_exports(load_config(a.config)))


def cmd_prepare_model(a):
    from ftlib.models import prepare_model

    prepare_model(load_config(a.config), force=a.force)


def cmd_prepare_data(a):
    from ftlib.data import prepare_data

    prepare_data(load_config(a.config), force=a.force)


def cmd_pack(a):
    from ftlib.train import pack

    pack(load_config(a.config))


def cmd_prepare(a):
    from ftlib.data import prepare_data
    from ftlib.models import prepare_model
    from ftlib.train import pack

    cfg = load_config(a.config)
    if not a.skip_model:
        prepare_model(cfg, force=a.force)
    prepare_data(cfg, force=a.force)
    pack(cfg)


def cmd_dry_run(a, extra):
    from ftlib.data import is_prepared
    from ftlib.train import build_experiment

    cfg = load_config(a.config)
    exp = build_experiment(
        cfg,
        run_name=a.run_name,
        steps=a.steps,
        world_size=a.world_size or cfg.slurm.gpus,
        overrides=extra,
        need_data=is_prepared(cfg),
    )
    print(exp)
    n = exp.model.num_params
    print(f"\nmodel params: {n / 1e9:.2f}B   world size: {a.world_size or cfg.slurm.gpus}")


def cmd_train(a, extra):
    from ftlib.train import train

    train(
        load_config(a.config),
        run_name=a.run_name,
        steps=a.steps,
        overrides=extra,
        fit_check=a.fit_check,
    )


def cmd_export(a):
    from ftlib.export import export_hf

    export_hf(
        load_config(a.config), run_name=a.run_name, step=a.step, out=Path(a.out) if a.out else None
    )


def cmd_stop(a):
    cfg = load_config(a.config)
    run = cfg.run_dir(a.run_name)
    if not run.is_dir():
        sys.exit(f"no run folder {run}")
    (run / "STOP").touch()
    print(
        f"requested stop: {run}/STOP (the job ends after its current step; use scancel to kill immediately)"
    )


def cmd_status(a):
    from ftlib.status import collect, serve, write

    if a.serve:
        serve(a.serve, interval=a.interval)
    else:
        out = write(collect(), Path(a.out) if a.out else None)
        print(f"wrote {out}")


def cmd_list(a):
    for p in config_paths():
        if p.name.startswith("_"):
            continue
        try:
            cfg = load_config(p)
            s = _state(cfg)
            flags = "".join(
                c if s[k] else "."
                for c, k in (("M", "model_ready"), ("D", "data_ready"), ("P", "packed"))
            )
            print(
                f"{flags}  {cfg.name:40s} {s['run_state']:12s} step {s['step'] or '-':>6}  {','.join(s['checkpoints'])}"
            )
        except ConfigError as e:
            print(f"!!!  {p.name:40s} {str(e).splitlines()[0]}")
    print("(M = model prepared, D = data tokenized, P = packed)")


def main():
    _setup_logging()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("new", help="create configs/<name>.yaml from the template")
    s.add_argument("name")
    s.add_argument("--template", help="copy this YAML instead of _template.yaml")
    for name in ("validate", "paths"):
        sub.add_parser(name).add_argument("config")
    for name in ("prepare-model", "prepare-data"):
        s = sub.add_parser(name)
        s.add_argument("config")
        s.add_argument("--force", action="store_true", help="redo even if already prepared")
    sub.add_parser("pack").add_argument("config")
    s = sub.add_parser("prepare", help="prepare-model + prepare-data + pack")
    s.add_argument("config")
    s.add_argument("--force", action="store_true")
    s.add_argument("--skip-model", action="store_true")
    for name in ("dry-run", "train"):
        s = sub.add_parser(name, help="extra --a.b=c args override the OLMo-core config")
        s.add_argument("config")
        s.add_argument("--run-name", help="run folder name (default: the config's name)")
        s.add_argument("--steps", type=int, help="override train.steps (fit checks)")
        if name == "dry-run":
            s.add_argument("--world-size", type=int)
        else:
            s.add_argument(
                "--fit-check",
                action="store_true",
                help="slurm.fit_check_steps steps into <name>-fitcheck, no checkpoints",
            )
    s = sub.add_parser("export")
    s.add_argument("config")
    s.add_argument("--run-name")
    s.add_argument("--step", type=int, help="default: latest checkpoint")
    s.add_argument("--out")
    s = sub.add_parser("stop")
    s.add_argument("config")
    s.add_argument("--run-name")
    s = sub.add_parser("status", help="write dashboard/data.js (or serve the dashboard)")
    s.add_argument("--out")
    s.add_argument(
        "--serve", type=int, metavar="PORT", help="serve the control panel on localhost:PORT"
    )
    s.add_argument("--interval", type=int, default=60, help="refresh seconds when serving")
    sub.add_parser("list")

    a, extra = p.parse_known_args()
    if extra and a.cmd not in ("dry-run", "train"):
        p.error(f"unrecognized arguments: {' '.join(extra)}")
    fn = {
        "new": cmd_new,
        "validate": cmd_validate,
        "paths": cmd_paths,
        "prepare-model": cmd_prepare_model,
        "prepare-data": cmd_prepare_data,
        "pack": cmd_pack,
        "prepare": cmd_prepare,
        "export": cmd_export,
        "stop": cmd_stop,
        "status": cmd_status,
        "list": cmd_list,
    }
    try:
        if a.cmd == "dry-run":
            cmd_dry_run(a, extra)
        elif a.cmd == "train":
            cmd_train(a, extra)
        else:
            fn[a.cmd](a)
    except ConfigError as e:
        sys.exit(f"CONFIG ERROR: {e}")


if __name__ == "__main__":
    main()
