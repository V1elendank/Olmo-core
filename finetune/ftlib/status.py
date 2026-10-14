"""
Collect everything the control panel shows into one JSON blob (finetune/dashboard/data.js):

  cluster   GPU nodes of the gpu partition: state, GPUs in use / total, features
  jobs      your Slurm queue + the last 7 days of job history (sacct)
  runs      every folder in $OLMO_ROOT/runs: status, metrics per step, events, checkpoints, export
  data      every prepared dataset: stats + decoded samples with the trainable spans marked
  models    every prepared base model
  configs   every finetune/configs/*.yaml (raw text + what is prepared for it)

Pure stdlib + PyYAML so it runs anywhere (portal, laptop) in a second or two.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parents[1]  # finetune/
DASH = HERE / "dashboard"
METRIC_KEEP = re.compile(
    r"^(train/|optim/LR|optim/total grad norm|throughput/device/(TPS|MFU|TFLOPs)|"
    r"gpu_memory/GPU (active|reserved) mem|system/)"
)
MAX_POINTS = 2500


def _user() -> str:
    return os.environ.get("USER") or getpass.getuser()


def _root() -> Path:
    return Path(os.environ.get("OLMO_ROOT") or f"/bigtemp/{_user()}/olmo")


def _run(cmd: List[str], timeout: int = 20) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    try:
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def _du(p: Path) -> int:
    total = 0
    for dirpath, _, files in os.walk(p):
        for fn in files:
            try:
                total += os.lstat(os.path.join(dirpath, fn)).st_size
            except OSError:
                pass
    return total


# ----------------------------------------------------------------------------------- slurm
def _gpu_count(tres: str) -> int:
    m = re.search(r"gres/gpu=(\d+)", tres or "")
    if m:
        return int(m.group(1))
    m = re.search(r"gpu(?::[\w-]+)?:(\d+)", tres or "")
    return int(m.group(1)) if m else 0


def cluster() -> Dict[str, Any]:
    out = _run(["scontrol", "show", "node", "-o"])
    if out is None:
        return {"available": False}
    nodes = []
    for line in out.splitlines():
        kv = dict(re.findall(r"(\w+)=(\S*)", line))
        if "gpu" not in kv.get("Partitions", "").split(","):
            continue
        total = _gpu_count(kv.get("Gres", "")) or _gpu_count(kv.get("CfgTRES", ""))
        if not total:
            continue
        used = _gpu_count(kv.get("AllocTRES", ""))
        feats = kv.get("AvailableFeatures", "")
        gpu_type = next(
            (
                f
                for f in feats.split(",")
                if re.search(
                    r"(a100|h100|h200|a40|a6000|l40|v100|rtx|titan|a16|2080|3090|4090)", f, re.I
                )
            ),
            "",
        )
        nodes.append(
            {
                "node": kv.get("NodeName"),
                "state": kv.get("State", "").lower(),
                "gpu_type": gpu_type,
                "gpus_total": total,
                "gpus_used": used,
                "features": feats,
                "cpus": f"{kv.get('CPUAlloc', '?')}/{kv.get('CPUTot', '?')}",
                "mem_gb": f"{int(kv.get('AllocMem', 0) or 0) // 1024}/{int(kv.get('RealMemory', 0) or 0) // 1024}",
            }
        )
    return {
        "available": True,
        "nodes": sorted(nodes, key=lambda n: (n["gpu_type"], n["node"] or "")),
    }


def jobs() -> Dict[str, Any]:
    q = _run(["squeue", "-u", _user(), "-h", "-o", "%i|%j|%T|%M|%l|%R|%b|%S|%P"])
    h = _run(
        [
            "sacct",
            "-u",
            _user(),
            "-S",
            "now-7days",
            "-X",
            "-n",
            "-P",
            "-o",
            "JobID,JobName%40,State,Elapsed,Start,End,ExitCode,NodeList,AllocTRES,Submit",
        ]
    )
    queue = []
    for line in (q or "").splitlines():
        f = line.split("|")
        if len(f) >= 9:
            queue.append(
                dict(
                    zip(
                        [
                            "id",
                            "name",
                            "state",
                            "elapsed",
                            "limit",
                            "reason",
                            "gres",
                            "start",
                            "partition",
                        ],
                        f,
                    )
                )
            )
    hist = []
    for line in (h or "").splitlines():
        f = line.split("|")
        if len(f) >= 10:
            d = dict(
                zip(
                    [
                        "id",
                        "name",
                        "state",
                        "elapsed",
                        "start",
                        "end",
                        "exit",
                        "nodes",
                        "tres",
                        "submit",
                    ],
                    f,
                )
            )
            d["gpus"] = _gpu_count(d.pop("tres"))
            hist.append(d)
    hist.sort(key=lambda d: d["submit"], reverse=True)
    return {"available": q is not None, "queue": queue, "history": hist[:80]}


# ----------------------------------------------------------------------------------- runs
_LOG_STEP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S*\s.*\[step=(\d+)/(\d+)")
_LOG_KV = re.compile(r"^\s+(\S.*?)=(\S+)\s*$")


def _metrics_from_log(log_path: Path) -> List[Dict[str, Any]]:
    """For runs from before metrics.jsonl existed (e.g. the uva/ feasibility runs)."""
    rows: Dict[int, Dict[str, Any]] = {}
    cur = None
    try:
        f = open(log_path, errors="replace")
    except OSError:
        return []
    with f:
        for line in f:
            m = _LOG_STEP.search(line)
            if m:
                step = int(m.group(2))
                cur = rows.setdefault(
                    step,
                    {
                        "step": step,
                        "time": time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")),
                        "_max": int(m.group(3)),
                    },
                )
                continue
            if cur is not None:
                m = _LOG_KV.match(line)
                if m and m.group(1) not in cur:
                    try:
                        cur[m.group(1)] = float(m.group(2).replace(",", ""))
                    except ValueError:
                        pass
                elif not line.startswith(" "):
                    cur = None
    return [rows[k] for k in sorted(rows)]


def _slim(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[int, Dict[str, Any]] = {}
    for r in rows:  # one step's metrics can arrive in several log calls
        if "step" in r:
            m = merged.setdefault(r["step"], {})
            for k, v in r.items():
                m.setdefault(k, v)
    rows = [merged[k] for k in sorted(merged)]
    if len(rows) > MAX_POINTS:
        stride = len(rows) / MAX_POINTS
        rows = [rows[int(i * stride)] for i in range(MAX_POINTS)] + [rows[-1]]
    keys = sorted({k for r in rows for k in r if METRIC_KEEP.match(k)})
    return {
        "steps": [r["step"] for r in rows],
        "time": [r.get("time") for r in rows],
        "series": {k: [r.get(k) for r in rows] for k in keys},
    }


def runs(root: Path) -> List[Dict[str, Any]]:
    rd = root / "runs"
    out = []
    if not rd.is_dir():
        return out
    for d in sorted(p for p in rd.iterdir() if p.is_dir()):
        if d.name in ("prep",) or d.name.endswith("dataset-cache"):
            continue
        status = _json(d / "status.json") or {}
        rows = _jsonl(d / "metrics.jsonl")
        legacy_log = rd / f"{d.name}.train.log"
        if not rows and legacy_log.is_file():
            rows = _metrics_from_log(legacy_log)
            if rows:
                status.setdefault("state", "finished (legacy log)")
                status.setdefault("max_steps", rows[-1].get("_max"))
                status.setdefault("step", rows[-1]["step"])
        cks = []
        for c in sorted(d.glob("step*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
            if (c / "model_and_optim" / ".metadata").is_file():
                cks.append({"name": c.name, "bytes": _du(c), "mtime": c.stat().st_mtime})
        hf = [
            {"name": h.name, "bytes": _du(h)}
            for h in sorted(d.glob("hf*"))
            if (h / "config.json").is_file()
        ]
        rc = _json(d / "run_config.json") or {}
        if not rows and not cks and not status and not rc:
            continue
        logs = [
            p
            for p in list(d.glob("*.log"))
            + list(d.glob("logs/*.out"))
            + list(rd.glob(f"{d.name}.*.log"))
            if p.is_file()
        ]
        logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        tail = ""
        if logs:
            try:
                with open(logs[0], "rb") as f:
                    f.seek(max(0, logs[0].stat().st_size - 8000))
                    tail = f.read().decode(errors="replace")
            except OSError:
                pass
        out.append(
            {
                "name": d.name,
                "dir": str(d),
                "status": status,
                "config": rc,
                "metrics": _slim(rows),
                "events": _jsonl(d / "events.jsonl")[-60:],
                "checkpoints": cks,
                "hf_exports": hf,
                "export": _json(d / "export.json"),
                "stop_requested": (d / "STOP").exists(),
                "log_tail": tail[-8000:],
                "log_file": str(logs[0]) if logs else None,
                "logs": [
                    {
                        "name": str(p.relative_to(rd)),
                        "bytes": p.stat().st_size,
                        "mtime": p.stat().st_mtime,
                    }
                    for p in logs[:20]
                ],
                "jobs": _jsonl(d / "jobs.jsonl")[-30:],
                "mtime": d.stat().st_mtime,
            }
        )
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def datasets(root: Path) -> List[Dict[str, Any]]:
    out = []
    for m in sorted((root / "data").glob("*/manifest.json")):
        man = _json(m)
        if not man:
            continue
        man["dir"] = str(m.parent)
        man["pack"] = _json(m.parent / "pack.json")
        man["bytes"] = sum(f.stat().st_size for f in m.parent.glob("*.npy"))
        out.append(man)
    return out


def models(root: Path) -> List[Dict[str, Any]]:
    from .models import is_prepared

    out = []
    for d in sorted((root / "models").glob("*")):
        if not is_prepared(d):
            continue
        meta = _json(d / "config.json") or {}
        m = meta.get("model", {})
        out.append(
            {
                "key": d.name,
                "dir": str(d),
                "source": _json(d / "ft_source.json"),
                "bytes": _du(d),
                "d_model": m.get("d_model"),
                "n_layers": m.get("block", {}).get("n_layers", m.get("n_layers")),
                "vocab_size": m.get("vocab_size"),
            }
        )
    return out


def configs() -> List[Dict[str, Any]]:
    import sys

    sys.path.insert(0, str(HERE))
    from ftlib.config import config_paths, load_config
    from ftlib.models import is_prepared

    out = []
    for p in config_paths():
        item: Dict[str, Any] = {"file": str(p), "path": str(p), "text": p.read_text()}
        try:
            cfg = load_config(p)
            item.update(
                name=cfg.name,
                model_dir=str(cfg.model_dir),
                data_dir=str(cfg.data_dir),
                run_dir=str(cfg.run_dir()),
                model_ready=is_prepared(cfg.model_dir),
                data_ready=(cfg.data_dir / "manifest.json").is_file(),
                packed=(cfg.data_dir / "pack.json").is_file(),
            )
        except Exception as e:  # noqa: BLE001
            item["error"] = str(e)
        out.append(item)
    return out


def collect() -> Dict[str, Any]:
    root = _root()
    t0 = time.time()
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_ts": time.time(),
        "user": _user(),
        "host": socket.gethostname(),
        "root": str(root),
        "disk": None,
        "cluster": cluster(),
        "jobs": jobs(),
        "runs": runs(root),
        "datasets": datasets(root),
        "models": models(root),
        "configs": configs(),
    }
    try:
        st = os.statvfs(root)
        data["disk"] = {
            "free_bytes": st.f_bavail * st.f_frsize,
            "total_bytes": st.f_blocks * st.f_frsize,
        }
    except OSError:
        pass
    data["collect_seconds"] = round(time.time() - t0, 2)
    return data


def write(data: Dict[str, Any], out: Optional[Path] = None) -> Path:
    out = out or DASH / "data.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(
        "window.FT_DATA = " + json.dumps(data, separators=(",", ":"), default=str) + ";\n"
    )
    os.replace(tmp, out)
    return out


def serve(port: int, interval: int = 60) -> None:
    """Serve finetune/dashboard on 127.0.0.1:PORT, refreshing data.js every `interval` seconds.
    From your laptop:  ssh -L PORT:localhost:PORT <you>@portal.cs.virginia.edu  then open
    http://localhost:PORT"""
    from .web import Application, make_server

    srv = make_server(Application(_root()), port, DASH)
    port = srv.server_port
    stopped = threading.Event()

    def loop():
        while not stopped.is_set():
            try:
                snapshot = collect()
                if not stopped.is_set():
                    write(snapshot)
            except Exception as e:  # noqa: BLE001
                log.warning(f"status refresh failed: {e}")
            stopped.wait(interval)

    # Slurm status commands can be slow; uploads must not wait for the first snapshot.
    threading.Thread(target=loop, daemon=True).start()
    print(
        f"control panel on http://localhost:{port}  (data refreshed every {interval}s; Ctrl-C to stop)"
    )
    print(f"from your laptop:  ssh -L {port}:localhost:{port} {_user()}@portal.cs.virginia.edu")
    print(f"upload models/datasets and start a run: http://localhost:{port}/launch.html")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        srv.server_close()
