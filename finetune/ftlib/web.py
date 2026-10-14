"""Loopback-only upload and launch API for the fine-tuning dashboard."""
from __future__ import annotations

import fcntl
import functools
import http.server
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml

from .config import ARCHS, load_config
from .store import CHUNK_BYTES, Store, atomic_json

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent


def _jobs(path: Path) -> list:
    """Read complete job records while the submitter may still be appending a line."""
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                result.append(item)
        except ValueError:
            continue
    return result


def _integer(body: dict, key: str, default: int, low: int, high: int) -> int:
    value = body.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} must be an integer between {low} and {high}")
    return value


def _worker(config: Path, record_path: Path) -> None:
    """Run the existing submitter, never bypassing its fit check."""
    record = json.loads(record_path.read_text())
    record["state"] = "submitting"
    atomic_json(record_path, record)
    try:
        with record_path.with_name("submit.log").open("w") as log:
            result = subprocess.run(
                ["bash", str(HERE / "scripts/submit.sh"), str(config)],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        record.update(
            state="submitted" if result.returncode == 0 else "failed", returncode=result.returncode
        )
        jobs_path = Path(record["run_dir"]) / "jobs.jsonl"
        record["jobs"] = _jobs(jobs_path)
        if result.returncode:
            record[
                "error"
            ] = "Submission failed. Check the log and any listed jobs before starting a new run."
    except Exception as e:
        record.update(state="failed", error=str(e))
    atomic_json(record_path, record)


class Application:
    """Build immutable run configs from registered local uploads."""

    def __init__(self, root: Path, spawn=None):
        self.root = Path(root).resolve()
        self.store = Store(self.root)
        self._lock_file = (self.root / ".webui.lock").open("a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise ValueError(
                "Another panel is using this storage root; connect to that panel or stop it first"
            ) from None
        self.token = secrets.token_urlsafe(32)
        self.spawn = spawn or self._spawn
        # An interrupted validation can safely be restarted from complete uploaded files.
        for asset in self.store.inventory():
            if asset["state"] == "checking":
                asset.update(
                    state="uploading",
                    error="Server restarted during validation; finish upload again",
                )
                atomic_json(self.store.directory(asset["id"]) / "asset.json", asset)

    def close(self):
        """Release the shared-filesystem panel lock on shutdown."""
        self._lock_file.close()

    def _spawn(self, config: Path, record_path: Path):
        env = os.environ.copy()
        env["OLMO_ROOT"] = str(self.root)
        env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
        with record_path.with_name("worker.log").open("ab") as log:
            return subprocess.Popen(
                [sys.executable, "-m", "ftlib.web", "--submit", str(config), str(record_path)],
                cwd=REPO,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def launches(self):
        """Return submission states, known job IDs, and bounded log tails."""
        result = []
        for path in sorted(
            (self.root / "web_runs").glob("*/submission.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            item = json.loads(path.read_text())
            log = path.with_name("submit.log")
            if log.exists():
                with log.open("rb") as f:
                    f.seek(max(0, log.stat().st_size - 12000))
                    item["log"] = f.read().decode("utf-8", errors="replace")
            # Partial submissions can have valid jobs even before the submitter exits.
            jobs = Path(item["run_dir"]) / "jobs.jsonl"
            if jobs.exists():
                item["jobs"] = _jobs(jobs)
            result.append(item)
        return result[:100]

    def start(self, body: dict) -> dict:
        """Validate selected uploads and launch the gated pipeline exactly once per run."""
        request_id = body.get("request_id", "")
        if not isinstance(request_id, str) or not re.fullmatch(r"[a-f0-9]{32}", request_id):
            raise ValueError("A valid launch request ID is required")
        name = body.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name):
            raise ValueError("Run name: 1–80 letters, numbers, dots, dashes, or underscores")
        folder = self.root / "web_runs" / name
        path = folder / "submission.json"
        with self.store.lock:
            if path.exists():
                previous = json.loads(path.read_text())
                if previous["request_id"] == request_id and previous["request"] == body:
                    return previous
                raise ValueError(
                    "That run name already exists. Choose a new name for a fresh fine-tune"
                )
            if (self.root / "runs" / name).exists():
                raise ValueError("That run already has results. Choose a new run name")
            if shutil.which("sbatch") is None:
                raise ValueError(
                    "Slurm is unavailable. Open this panel on the CS portal through the SSH tunnel"
                )
            model = self.store.get(body.get("model_id", ""))
            data = self.store.get(body.get("dataset_id", ""))
            if (
                model["state"] != "ready"
                or model["kind"] != "model"
                or data["state"] != "ready"
                or data["kind"] != "dataset"
            ):
                raise ValueError(
                    "Select one completed model upload and one completed dataset upload"
                )
            arch = body.get("arch")
            if (
                arch not in ARCHS
                or arch == "tiny"
                or not arch.startswith(model["model"]["family"] + "_")
            ):
                raise ValueError("Choose an architecture matching the uploaded OLMo model family")
            fmt = body.get("format", "chat")
            if fmt not in ("chat", "prompt_response", "text"):
                raise ValueError("Choose chat, prompt_response, or text dataset format")
            template = body.get("chat_template", "")
            if template not in ("", "olmo2", "olmo3"):
                raise ValueError("Invalid chat template selection")
            if fmt != "text" and not template and not model["model"]["has_chat_template"]:
                raise ValueError(
                    "This tokenizer has no chat template; select an OLMo chat template"
                )
            constraint = body.get("constraint", "h100_94gb")
            if constraint not in ("h100_94gb", "a100_80gb", "a100_40gb", "a40"):
                raise ValueError("Choose a supported GPU type")
            seq_len = _integer(body, "seq_len", 1024, 128, 65536)
            gpus = _integer(body, "gpus", 2, 1, 8)
            if gpus > (2 if constraint == "h100_94gb" else 4):
                raise ValueError(
                    "GPU count exceeds the supported single-node allocation for this GPU type"
                )
            batch = _integer(body, "global_batch_seqs", 4, 1, 4096)
            steps = _integer(body, "steps", 100, 1, 1000000)
            hours = _integer(body, "hours", 4, 1, 96)
            lr = body.get("lr", 2e-5)
            if type(lr) not in (int, float) or not math.isfinite(lr) or not 0 < lr <= 0.01:
                raise ValueError("Learning rate must be positive and at most 0.01")
            model_path = self.store.directory(model["id"]) / "files"
            data_path = self.store.directory(data["id"]) / "files"
            dataset = {"name": "uploaded", "source": str(data_path), "format": fmt}
            for field in ("messages_column", "prompt_column", "response_column", "text_column"):
                value = body.get(field, "")
                if not isinstance(value, str) or len(value) > 120:
                    raise ValueError(f"Invalid {field}")
                if value.strip():
                    dataset[field] = value.strip()
            config = {
                "name": name,
                "model": {"source": "hf_local", "path": str(model_path), "arch": arch},
                "tokenizer": str(model_path),
                "data": [dataset],
                "train": {
                    "steps": steps,
                    "seq_len": seq_len,
                    "global_batch_seqs": batch,
                    "microbatch_seqs": 1,
                    "lr": lr,
                    "optim_foreach": False,
                    "activation_checkpointing": "full",
                },
                "checkpoint": {"every": min(200, steps), "final": True, "keep": 1},
                "slurm": {
                    "gpus": gpus,
                    "constraint": constraint,
                    "time": f"{hours}:00:00",
                    "fit_check_steps": 5,
                },
                "export": {"hf": True},
                "notes": f'Web uploads: model {model["id"]} ({model["digest"]}), data {data["id"]} ({data["digest"]})',
            }
            if template:
                config["chat_template"] = template
            # Exclusive mkdir prevents another panel process from launching the same run.
            folder.mkdir(parents=True, exist_ok=False)
            config_path = folder / "run.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            try:
                load_config(config_path)
            except Exception:
                shutil.rmtree(folder)
                raise
            record = {
                "name": name,
                "request_id": request_id,
                "request": body,
                "assets": [model["id"], data["id"]],
                "state": "launching",
                "config": str(config_path),
                "run_dir": str(self.root / "runs" / name),
                "jobs": [],
            }
            atomic_json(path, record)
            try:
                process = self.spawn(config_path, path)
                # The worker owns later state transitions; do not overwrite its result.
                record["worker_pid"] = process.pid if process is not None else None
            except Exception as e:
                record.update(state="failed", error=f"Could not start submitter: {e}")
                atomic_json(path, record)
            return record


def make_server(app: Application, port: int = 8765, dashboard: Path | None = None):
    """Expose a same-origin, token-protected API on loopback only."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data):
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def allowed(self, mutate=False):
            port = self.server.server_port
            host = self.headers.get("Host", "")
            if host not in (f"localhost:{port}", f"127.0.0.1:{port}"):
                self.reply(403, {"error": "Use the localhost SSH tunnel address"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{host}":
                self.reply(403, {"error": "Cross-origin requests are not allowed"})
                return False
            if mutate and not secrets.compare_digest(self.headers.get("X-FT-Token", ""), app.token):
                self.reply(403, {"error": "Reload the panel to establish a new session"})
                return False
            return True

        def body(self, limit):
            n = int(self.headers.get("Content-Length", "0"))
            if not 0 < n <= limit or self.headers.get("Transfer-Encoding"):
                raise ValueError("Invalid request size")
            self.connection.settimeout(120)
            data = self.rfile.read(n)
            if len(data) != n:
                raise ValueError("Incomplete request body")
            return data

        def do_GET(self):
            if not self.allowed():
                return
            route = urlparse(self.path).path
            try:
                if route == "/api/session":
                    self.reply(
                        200,
                        {
                            "token": app.token,
                            "chunk_bytes": CHUNK_BYTES,
                            "slurm_available": shutil.which("sbatch") is not None,
                            "architectures": [a for a in ARCHS if a != "tiny"],
                        },
                    )
                elif route == "/api/assets":
                    self.reply(200, {"assets": app.store.inventory()})
                elif route == "/api/runs":
                    self.reply(200, {"runs": app.launches()})
                elif re.fullmatch(r"/api/uploads/[a-f0-9]{32}", route):
                    asset_id = route.rsplit("/", 1)[1]
                    if not (app.store.directory(asset_id) / "asset.json").is_file():
                        self.reply(404, {"error": "Upload not found"})
                        return
                    self.reply(
                        200,
                        {"asset": app.store.get(asset_id), "files": app.store.offsets(asset_id)},
                    )
                elif route.startswith("/api/"):
                    self.reply(404, {"error": "Unknown API route"})
                else:
                    super().do_GET()
            except (ValueError, KeyError, TypeError, OSError) as e:
                self.reply(400, {"error": str(e)})

        def do_POST(self):
            self.mutate("POST")

        def do_PUT(self):
            self.mutate("PUT")

        def do_DELETE(self):
            self.mutate("DELETE")

        def mutate(self, method):
            if not self.allowed(mutate=True):
                self.close_connection = True
                return
            route = urlparse(self.path).path
            try:
                parts = route.strip("/").split("/")
                if method == "PUT" and len(parts) == 5 and parts[:2] == ["api", "uploads"]:
                    offset = int(self.headers.get("X-Upload-Offset", "-1"))
                    end = (
                        app.store.chunk(parts[2], int(parts[4]), offset, self.body(CHUNK_BYTES))
                        if parts[3] == "files"
                        else None
                    )
                    if end is None:
                        raise ValueError("Unknown upload route")
                    self.reply(200, {"offset": end})
                elif method == "DELETE" and len(parts) == 3 and parts[:2] == ["api", "uploads"]:
                    app.store.remove(parts[2])
                    self.reply(200, {"removed": True})
                elif (
                    method == "POST"
                    and len(parts) == 4
                    and parts[:2] == ["api", "uploads"]
                    and parts[3] == "complete"
                ):
                    self.reply(200, {"asset": app.store.finalize(parts[2])})
                elif method == "POST" and route in ("/api/uploads", "/api/runs"):
                    body = json.loads(self.body(2 * 1024 * 1024))
                    if not isinstance(body, dict):
                        raise ValueError("Expected a JSON object")
                    if route == "/api/uploads":
                        self.reply(
                            201,
                            {
                                "asset": app.store.create(
                                    body.get("kind"), body.get("name"), body.get("files")
                                )
                            },
                        )
                    else:
                        self.reply(202, {"run": app.start(body)})
                else:
                    self.reply(404, {"error": "Unknown API route"})
            except (ValueError, KeyError, TypeError, OSError) as e:
                self.close_connection = True
                self.reply(400, {"error": str(e)})

    class Server(http.server.ThreadingHTTPServer):
        def server_close(self):
            super().server_close()
            app.close()

    handler = functools.partial(Handler, directory=str(dashboard or HERE / "dashboard"))
    try:
        return Server(("127.0.0.1", port), handler)
    except Exception:
        app.close()
        raise


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--submit":
        _worker(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit("Use: python finetune/ft.py status --serve 8765")
