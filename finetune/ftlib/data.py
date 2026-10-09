"""
Dataset preparation: turn any supported dataset into OLMo-core's tokenized SFT format.

For every dataset listed under `data:` in the run YAML this writes, into cfg.data_dir:

    <name>.token_ids_part_0000.npy   raw uint32 memmap of token ids, documents end with EOS (100257)
    <name>.labels_mask_0000.npy      raw bool memmap, True where the loss is computed
    manifest.json                    stats per dataset + a few decoded samples (shown in the dashboard)
    tokenizer/                       copy of the tokenizer (used again for the HF export)

Sources: a Hugging Face dataset id, or a local .jsonl / .json / .parquet / .csv / .arrow file or a
folder of them (your own data). Formats:

    chat             one column with a list of messages: [{"role": "user", "content": ...}, ...]
                     ShareGPT style [{"from": "human", "value": ...}] is accepted too.
                     Loss on assistant turns only.
    prompt_response  one prompt column + one response column (+ optional input/context column).
                     Rendered with the chat template; loss on the response only.
    text             plain text (continued pretraining / domain adaptation). Loss on every token;
                     documents longer than seq_len are split into seq_len pieces at packing time.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .config import DatasetCfg, RunConfig

log = logging.getLogger(__name__)

EOS_ID = 100257  # dolma2 <|endoftext|>: OLMo-core splits packed data into documents here
DOLMA2_VOCAB = 100278
LOCAL_EXTS = {
    ".jsonl": "json",
    ".json": "json",
    ".parquet": "parquet",
    ".csv": "csv",
    ".arrow": "arrow",
}
N_SAMPLES = 6  # decoded examples kept in the manifest for inspection

# Built-in chat templates, used when `chat_template:` is set to one of these names, or when the
# tokenizer has no template of its own (e.g. a base-model tokenizer).
CHAT_TEMPLATES = {
    # OLMo 3 Instruct / Think (ChatML)
    "olmo3": (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}"
        "{% if m['role'] == 'assistant' %}<|im_end|>{% else %}<|im_end|>\n{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    ),
    # OLMo 2 Instruct / Tulu 3
    "olmo2": (
        "{{ bos_token }}{% for m in messages %}"
        "{% if m['role'] == 'system' %}<|system|>\n{{ m['content'] }}\n"
        "{% elif m['role'] == 'user' %}<|user|>\n{{ m['content'] }}\n"
        "{% elif m['role'] == 'assistant' %}<|assistant|>\n{{ m['content'] }}{{ eos_token }}\n{% endif %}"
        "{% if loop.last and add_generation_prompt %}<|assistant|>\n{% endif %}{% endfor %}"
    ),
}

PROMPT_COLS = ("prompt", "instruction", "question", "query", "input_text")
INPUT_COLS = ("input", "context")
RESPONSE_COLS = ("response", "output", "answer", "completion", "chosen")
MESSAGE_COLS = ("messages", "conversations", "conversation", "chat", "dialog")
TEXT_COLS = ("text", "content", "document", "body")
ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "ipython": "tool",
}


# ----------------------------------------------------------------------------------- tokenizer
def load_tokenizer(cfg: RunConfig):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    if cfg.chat_template:
        tok.chat_template = CHAT_TEMPLATES.get(cfg.chat_template, cfg.chat_template)
    needs_template = any(d.format != "text" for d in cfg.data)
    if needs_template and not getattr(tok, "chat_template", None):
        log.warning("Tokenizer has no chat template; using the built-in 'olmo3' (ChatML) template")
        tok.chat_template = CHAT_TEMPLATES["olmo3"]
    check_tokenizer(tok)
    return tok


def check_tokenizer(tok) -> None:
    """OLMo 2 / OLMo 3 checkpoints use the dolma2 vocabulary; anything else would silently
    produce garbage, so fail early."""
    eos = tok.convert_ids_to_tokens(EOS_ID)
    if eos != "<|endoftext|>":
        raise ValueError(
            f"tokenizer is not dolma2-compatible: id {EOS_ID} is {eos!r}, expected '<|endoftext|>'"
        )
    if len(tok) > DOLMA2_VOCAB:
        raise ValueError(f"tokenizer has {len(tok)} tokens > dolma2 vocab {DOLMA2_VOCAB}")


def _ids(out: Any) -> List[int]:
    if isinstance(out, dict) or hasattr(out, "keys"):  # transformers versions differ
        out = out["input_ids"]
    return list(out)


def chat_ids(tok, messages, add_generation_prompt: bool) -> List[int]:
    return _ids(
        tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=False
        )
    )


# ----------------------------------------------------------------------------------- loading
def _local_files(src: str) -> Tuple[str, List[str]]:
    p = Path(src).expanduser()
    files = (
        [str(p)]
        if p.is_file()
        else sorted(
            f for ext in LOCAL_EXTS for f in glob.glob(str(p / "**" / f"*{ext}"), recursive=True)
        )
    )
    if not files:
        raise FileNotFoundError(f"no {sorted(LOCAL_EXTS)} files at {src}")
    kinds = {LOCAL_EXTS[Path(f).suffix] for f in files}
    if len(kinds) != 1:
        raise ValueError(f"{src}: mixed file types {kinds}; keep one type per dataset")
    return kinds.pop(), files


def is_local(src: str) -> bool:
    return src.startswith(("/", "./", "../", "~")) or Path(src).expanduser().exists()


def iter_rows(d: DatasetCfg) -> Iterator[Dict[str, Any]]:
    """Rows of the dataset, shuffled with d.seed when only a subset (num_examples) is used."""
    import datasets

    if is_local(d.source):
        kind, files = _local_files(d.source)
        ds = datasets.load_dataset(kind, data_files=files, split="train")
        log.info(f"[{d.name}] {len(ds):,d} rows from {len(files)} local {kind} file(s)")
        if d.num_examples:
            ds = ds.shuffle(seed=d.seed)
        yield from ds
    else:
        ds = datasets.load_dataset(d.source, d.subset, split=d.split, streaming=True)
        if d.num_examples:
            ds = ds.shuffle(seed=d.seed, buffer_size=20_000)
        yield from ds


def _pick(
    row: Dict[str, Any], explicit: Optional[str], candidates: Iterable[str], what: str
) -> str:
    if explicit:
        if explicit not in row:
            raise KeyError(f"{what} column '{explicit}' not in dataset columns {sorted(row)}")
        return explicit
    for c in candidates:
        if c in row:
            return c
    raise KeyError(
        f"could not find a {what} column among {sorted(row)}; set it in the YAML "
        f"(e.g. {what}_column: my_column)"
    )


# ----------------------------------------------------------------------------------- formats
def row_to_messages(
    row: Dict[str, Any], d: DatasetCfg, cols: Dict[str, str]
) -> List[Dict[str, str]]:
    if d.format == "chat":
        msgs = []
        for m in row[cols["messages"]] or []:
            role = m.get("role", m.get("from"))
            content = m.get("content", m.get("value"))
            if role is None or content is None:
                continue
            msgs.append(
                {
                    "role": ROLE_MAP.get(str(role).lower(), str(role).lower()),
                    "content": str(content),
                }
            )
    else:  # prompt_response
        prompt = str(row[cols["prompt"]] or "")
        if cols.get("input") and row.get(cols["input"]):
            prompt = f"{prompt}\n\n{row[cols['input']]}"
        msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(row[cols["response"]] or "")},
        ]
    if d.system_prompt and not any(m["role"] == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": d.system_prompt})
    return msgs


def encode_chat(tok, msgs: List[Dict[str, str]]) -> Tuple[List[int], np.ndarray, bool]:
    """Token ids + loss mask (assistant turns only). Returns aligned=False if the template does not
    tokenize prefixes consistently (mask may then be slightly off at turn boundaries)."""
    ids = chat_ids(tok, msgs, add_generation_prompt=False)
    if not ids or ids[-1] != EOS_ID:
        ids = ids + [EOS_ID]
    mask = np.zeros(len(ids), dtype=np.bool_)
    aligned = True
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prefix = chat_ids(tok, msgs[:i], add_generation_prompt=True) if i else []
        full = chat_ids(tok, msgs[: i + 1], add_generation_prompt=False)
        if ids[: len(prefix)] != prefix or ids[: len(full)] != full:
            aligned = False
        mask[len(prefix) : len(full)] = True
    mask[-1] = True  # learn to stop
    return ids, mask, aligned


def encode_text(tok, text: str) -> List[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"]) + [EOS_ID]


def _segments(tok, ids: np.ndarray, mask: np.ndarray, limit: int = 1500) -> List[Dict[str, Any]]:
    """Decoded text split into runs of trainable / not-trainable tokens (for the dashboard)."""
    ids, mask = ids[:limit], mask[:limit]
    out: List[Dict[str, Any]] = []
    start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or mask[i] != mask[start]:
            out.append({"t": tok.decode(ids[start:i].tolist()), "train": bool(mask[start])})
            start = i
    return out


# ----------------------------------------------------------------------------------- main entry
def _write(path: Path, arr: np.ndarray) -> None:
    with open(path, "ab") as f:  # raw bytes == the np.memmap layout OLMo-core reads
        f.write(arr.tobytes())


def prepare_one(tok, d: DatasetCfg, seq_len: int, out: Path) -> Dict[str, Any]:
    ids_path = out / f"{d.name}.token_ids_part_0000.npy"
    mask_path = out / f"{d.name}.labels_mask_0000.npy"
    for p in (ids_path, mask_path):
        p.unlink(missing_ok=True)

    stats: Counter = Counter()
    lengths: List[int] = []
    samples: List[Dict[str, Any]] = []
    buf_ids: List[np.ndarray] = []
    buf_mask: List[np.ndarray] = []
    buffered = 0
    cols: Dict[str, str] = {}
    t0 = time.time()

    def flush():
        nonlocal buffered
        if buf_ids:
            _write(ids_path, np.concatenate(buf_ids))
            _write(mask_path, np.concatenate(buf_mask))
            buf_ids.clear(), buf_mask.clear()
            buffered = 0

    def add(ids: List[int], mask: np.ndarray):
        nonlocal buffered
        a = np.asarray(ids, dtype=np.uint32)
        buf_ids.append(a)
        buf_mask.append(mask)
        buffered += a.size
        lengths.append(int(a.size))
        stats["tokens"] += int(a.size)
        stats["trainable_tokens"] += int(mask.sum())
        if len(samples) < N_SAMPLES:
            samples.append(
                {
                    "tokens": int(a.size),
                    "trainable": int(mask.sum()),
                    "segments": _segments(tok, a, mask),
                }
            )
        if buffered > 50_000_000:
            flush()

    for row in iter_rows(d):
        if d.num_examples and stats["examples"] >= d.num_examples:
            break
        stats["rows_read"] += 1
        if not cols:  # resolve column names from the first row
            if d.format == "chat":
                cols["messages"] = _pick(row, d.messages_column, MESSAGE_COLS, "messages")
            elif d.format == "prompt_response":
                cols["prompt"] = _pick(row, d.prompt_column, PROMPT_COLS, "prompt")
                cols["response"] = _pick(row, d.response_column, RESPONSE_COLS, "response")
                if d.input_column or any(c in row for c in INPUT_COLS):
                    cols["input"] = _pick(row, d.input_column, INPUT_COLS, "input")
            else:
                cols["text"] = _pick(row, d.text_column, TEXT_COLS, "text")
            log.info(f"[{d.name}] columns: {cols}")

        if d.format == "text":
            text = row[cols["text"]]
            if not text or not str(text).strip():
                stats["skipped_empty"] += 1
                continue
            ids = encode_text(tok, str(text))
            add(ids, np.ones(len(ids), dtype=np.bool_))  # packing splits docs > seq_len into pieces
            if len(ids) > seq_len:
                stats["longer_than_seq_len"] += 1
            stats["examples"] += 1
            continue

        msgs = row_to_messages(row, d, cols)
        if not any(m["role"] == "assistant" and m["content"].strip() for m in msgs):
            stats["skipped_no_response"] += 1
            continue
        ids, mask, aligned = encode_chat(tok, msgs)
        if not aligned:
            stats["mask_misaligned"] += 1
        if len(ids) > seq_len:
            if d.too_long == "skip":
                stats["skipped_too_long"] += 1
                continue
            ids, mask = ids[: seq_len - 1] + [EOS_ID], np.append(mask[: seq_len - 1], True)
            stats["truncated"] += 1
        if not mask[:-1].any():
            stats["skipped_no_response"] += 1
            continue
        add(ids, mask)
        stats["examples"] += 1
        if stats["examples"] % 5000 == 0:
            log.info(
                f"[{d.name}] {stats['examples']:,d} examples, {stats['tokens']:,d} tokens "
                f"({time.time() - t0:.0f}s)"
            )
    flush()
    if not stats["examples"]:
        raise RuntimeError(f"[{d.name}] no usable examples (stats: {dict(stats)})")
    if stats["mask_misaligned"]:
        log.warning(
            f"[{d.name}] {stats['mask_misaligned']} examples where the chat template does not "
            "tokenize prefixes consistently; check the samples in manifest.json"
        )

    lens = np.asarray(lengths)
    bins = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    bins = [b for b in bins if b < seq_len] + [seq_len + 1]  # last bin: longer than seq_len
    hist, _ = np.histogram(lens, bins=bins + [max(seq_len + 2, int(lens.max()) + 1)])
    entry = {
        "name": d.name,
        "source": d.source,
        "format": d.format,
        "split": d.split,
        "subset": d.subset,
        "columns": cols,
        **{k: int(v) for k, v in stats.items()},
        "trainable_fraction": round(stats["trainable_tokens"] / max(stats["tokens"], 1), 4),
        "length": {
            "mean": round(float(lens.mean()), 1),
            "median": int(np.median(lens)),
            "p95": int(np.percentile(lens, 95)),
            "max": int(lens.max()),
            "hist_edges": bins,
            "hist_counts": [int(x) for x in hist],
        },
        "files": [ids_path.name, mask_path.name],
        "seconds": round(time.time() - t0, 1),
        "samples": samples,
    }
    log.info(
        f"[{d.name}] done: {entry['examples']:,d} examples, {entry['tokens']:,d} tokens, "
        f"{entry['trainable_fraction']:.0%} trainable"
    )
    return entry


def is_prepared(cfg: RunConfig) -> bool:
    return (cfg.data_dir / "manifest.json").is_file()


def read_manifest(data_dir: Path) -> Dict[str, Any]:
    p = Path(data_dir) / "manifest.json"
    if not p.is_file():
        return {}
    with open(p) as f:
        return json.load(f)


def prepare_data(cfg: RunConfig, force: bool = False) -> Path:
    out = cfg.data_dir
    if is_prepared(cfg) and not force:
        log.info(f"Data already prepared: {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)
    (out / "pack.json").unlink(missing_ok=True)
    shutil.rmtree(cfg.pack_cache_dir, ignore_errors=True)  # stale packing indices
    tok = load_tokenizer(cfg)
    entries = [prepare_one(tok, d, cfg.train.seq_len, out) for d in cfg.data]
    tok.save_pretrained(str(out / "tokenizer"))
    manifest = {
        "data_key": cfg.data_key,
        "run_name": cfg.name,
        "tokenizer": cfg.tokenizer,
        "chat_template": cfg.chat_template,
        "seq_len": cfg.train.seq_len,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_examples": sum(e["examples"] for e in entries),
        "total_tokens": sum(e["tokens"] for e in entries),
        "trainable_tokens": sum(e["trainable_tokens"] for e in entries),
        "datasets": entries,
    }
    tmp = out / "manifest.json.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, out / "manifest.json")  # manifest marks the data as complete
    log.info(
        f"Data ready in {out}: {manifest['total_examples']:,d} examples, {manifest['total_tokens']:,d} tokens"
    )
    return out
