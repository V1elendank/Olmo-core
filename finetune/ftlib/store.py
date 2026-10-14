"""Immutable, chunked browser uploads. No torch imports and no cluster jobs here."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import threading
import uuid
from pathlib import Path, PurePosixPath

CHUNK_BYTES = 8 * 1024 * 1024
DATA_EXTS = {".jsonl", ".json", ".csv", ".parquet", ".arrow"}
MODEL_EXTS = {".json", ".safetensors", ".model", ".txt", ".jinja"}


def atomic_json(path: Path, value) -> None:
    """Replace a JSON record atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def relative_path(value: str) -> str:
    """Reject traversal, hidden files, and platform-specific path escapes."""
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError("Invalid file path")
    p = PurePosixPath(value)
    if "\\" in value or "\x00" in value or p.is_absolute():
        raise ValueError("Use relative file paths")
    if any(x in ("", ".", "..") or x.startswith(".") for x in value.split("/")):
        raise ValueError("Hidden files and path traversal are not allowed")
    return str(p)


def _json_file(path: Path):
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"Missing or oversized JSON file: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def validate_model(folder: Path) -> dict:
    """Check HF OLMo files and shard references without loading model weights."""
    config = _json_file(folder / "config.json")
    family = config.get("model_type")
    if family not in ("olmo2", "olmo3"):
        raise ValueError("Upload a Hugging Face OLMo 2 or OLMo 3 model folder")
    _json_file(folder / "tokenizer.json")
    _json_file(folder / "tokenizer_config.json")
    weights = sorted(folder.glob("*.safetensors"))
    if not weights:
        raise ValueError("The model folder needs .safetensors weights")
    index = folder / "model.safetensors.index.json"
    if index.exists():
        mapping = _json_file(index).get("weight_map", {})
        if (
            not isinstance(mapping, dict)
            or not mapping
            or not all(isinstance(value, str) for value in mapping.values())
        ):
            raise ValueError("Model weight index must map tensor names to shard filenames")
        for filename in set(mapping.values()):
            if relative_path(filename) != Path(filename).name or not (folder / filename).is_file():
                raise ValueError(f"Missing or invalid model shard: {filename}")
    elif len(weights) != 1:
        raise ValueError("Sharded model weights require model.safetensors.index.json")
    for p in weights:
        with p.open("rb") as f:
            raw = f.read(8)
            n = struct.unpack("<Q", raw)[0] if len(raw) == 8 else 0
            if not 0 < n <= min(p.stat().st_size - 8, 64 * 1024 * 1024):
                raise ValueError(f"Invalid safetensors header: {p.name}")
            header = json.loads(f.read(n))
            if not isinstance(header, dict):
                raise ValueError(f"Invalid tensor header in {p.name}")
            tensors = [v for k, v in header.items() if k != "__metadata__"]
            if not tensors:
                raise ValueError(f"No tensors in {p.name}")
            for tensor in tensors:
                if not isinstance(tensor, dict):
                    raise ValueError(f"Invalid tensor entry in {p.name}")
                offsets = tensor.get("data_offsets", [])
                if (
                    len(offsets) != 2
                    or not 0 <= offsets[0] <= offsets[1] <= p.stat().st_size - 8 - n
                ):
                    raise ValueError(f"Invalid tensor offsets in {p.name}")
    return {
        "family": family,
        "hidden_size": config.get("hidden_size"),
        "layers": config.get("num_hidden_layers"),
        "has_chat_template": bool(
            _json_file(folder / "tokenizer_config.json").get("chat_template")
            or (folder / "chat_template.jinja").exists()
        ),
    }


class Store:
    """Each finalized upload has its own immutable directory and content digest."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.base = self.root / "uploads"
        self.base.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def directory(self, asset_id: str) -> Path:
        """Resolve a validated upload ID inside this store."""
        if not re.fullmatch(r"[a-f0-9]{32}", asset_id):
            raise ValueError("Invalid upload ID")
        return self.base / asset_id

    def get(self, asset_id: str) -> dict:
        """Read the upload's persisted state."""
        p = self.directory(asset_id) / "asset.json"
        if not p.is_file():
            raise ValueError("Upload not found")
        return json.loads(p.read_text())

    def inventory(self) -> list:
        """List all uploads, including unfinished transfers."""
        records = []
        for p in sorted(
            self.base.glob("*/asset.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                records.append(json.loads(p.read_text()))
            except (ValueError, OSError):
                continue
        return records

    def create(self, kind: str, name: str, files: list) -> dict:
        """Reserve a new immutable version and validate its declared file list."""
        if kind not in ("model", "dataset"):
            raise ValueError("Choose model or dataset")
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("Give the upload a name (up to 120 characters)")
        if not isinstance(files, list) or not 1 <= len(files) <= 10000:
            raise ValueError("Select between 1 and 10,000 files")
        entries = []
        seen = set()
        for item in files:
            path = relative_path(item["path"])
            size = item["size"]
            if type(size) is not int or not 0 < size <= 2 * 1024**4:
                raise ValueError("Files must be nonempty and at most 2 TB each")
            if path in seen or Path(path).suffix not in (
                MODEL_EXTS if kind == "model" else DATA_EXTS
            ):
                raise ValueError(f"Duplicate or unsupported file: {path}")
            seen.add(path)
            entries.append({"path": path, "size": size})
        if kind == "dataset" and len({Path(f["path"]).suffix for f in entries}) != 1:
            raise ValueError("Use one file type per dataset upload")
        total = sum(f["size"] for f in entries)
        if total > shutil.disk_usage(self.root).free - 1024**3:
            raise ValueError("Not enough free disk space for this upload plus 1 GB headroom")
        asset_id = uuid.uuid4().hex
        record = {
            "id": asset_id,
            "kind": kind,
            "name": name.strip(),
            "state": "uploading",
            "bytes": total,
            "files": entries,
        }
        folder = self.directory(asset_id)
        (folder / "files").mkdir(parents=True)
        atomic_json(folder / "asset.json", record)
        return record

    def offsets(self, asset_id: str) -> list:
        """Return actual stored byte counts so an interrupted upload can resume."""
        with self.lock:
            record = self.get(asset_id)
            return [
                dict(
                    f,
                    offset=(self.directory(asset_id) / "files" / f["path"]).stat().st_size
                    if (self.directory(asset_id) / "files" / f["path"]).exists()
                    else 0,
                )
                for f in record["files"]
            ]

    def chunk(self, asset_id: str, index: int, offset: int, content: bytes) -> int:
        """Append one bounded chunk at exactly the expected offset."""
        with self.lock:
            record = self.get(asset_id)
            if record["state"] != "uploading":
                raise ValueError("This upload is immutable; create a new upload version")
            if not 0 <= index < len(record["files"]) or not 0 < len(content) <= CHUNK_BYTES:
                raise ValueError("Invalid file or chunk size")
            entry = record["files"][index]
            p = self.directory(asset_id) / "files" / entry["path"]
            current = p.stat().st_size if p.exists() else 0
            if offset != current or current + len(content) > entry["size"]:
                raise ValueError("Upload offset mismatch; resume from the server offset")
            if len(content) > shutil.disk_usage(self.root).free - 1024**3:
                raise ValueError("Disk space is low; upload paused")
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("ab") as f:
                f.write(content)
            return current + len(content)

    def finalize(self, asset_id: str) -> dict:
        """Validate complete files and record their SHA-256 hashes before use."""
        with self.lock:
            record = self.get(asset_id)
            if record["state"] == "ready":
                return record
            if record["state"] != "uploading":
                raise ValueError("Upload is already being checked")
            if any(f["offset"] != f["size"] for f in self.offsets(asset_id)):
                raise ValueError("Upload is incomplete; resume the missing files")
            record["state"] = "checking"
            atomic_json(self.directory(asset_id) / "asset.json", record)
        try:
            folder = self.directory(asset_id) / "files"
            hashes = []
            for entry in record["files"]:
                digest = hashlib.sha256()
                with (folder / entry["path"]).open("rb") as f:
                    for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
                        digest.update(chunk)
                hashes.append({"path": entry["path"], "sha256": digest.hexdigest()})
            if record["kind"] == "model":
                record["model"] = validate_model(folder)
            record.update(
                state="ready",
                hashes=hashes,
                digest=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            )
            record.pop("error", None)
        except Exception as e:
            record.update(state="uploading", error=str(e))
            raise
        finally:
            with self.lock:
                atomic_json(self.directory(asset_id) / "asset.json", record)
        return record

    def remove(self, asset_id: str) -> None:
        """Delete only an upload with no recorded run references."""
        with self.lock:
            record = self.get(asset_id)
            if record["state"] == "checking":
                raise ValueError("Wait for upload validation to finish")
            for p in (self.root / "web_runs").glob("*/submission.json"):
                submission = json.loads(p.read_text())
                if asset_id in submission.get("assets", []):
                    raise ValueError(
                        "This upload is referenced by a run; it is retained for reproducibility"
                    )
            shutil.rmtree(self.directory(asset_id))
