"""Offline fixtures for CPU tests: a dolma2-shaped tokenizer and small datasets in every format."""

from __future__ import annotations

import json
import random
from pathlib import Path

SPECIAL = {100257: "<|endoftext|>", 100264: "<|im_start|>", 100265: "<|im_end|>", 100277: "<|pad|>"}


def make_test_tokenizer(out: Path) -> Path:
    """Byte-level tokenizer with the dolma2 vocab size and special-token ids (no download needed).
    Each byte is one token; ids 256..100256 are never produced."""
    from tokenizers import Tokenizer, decoders, models
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    from .data import CHAT_TEMPLATES

    vocab = {c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}
    for i in range(len(vocab), 100257):
        vocab[f"<|unused_{i}|>"] = i
    for i in range(100257, 100278):
        vocab[SPECIAL.get(i, f"<|extra_{i}|>")] = i
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        pad_token="<|pad|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    fast.chat_template = CHAT_TEMPLATES["olmo3"]
    fast.save_pretrained(str(out))
    return out


WORDS = (
    "the model learns to answer questions about data science physics music history code "
    "gradient token checkpoint cluster virginia batch loss"
).split()


def _sent(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def make_test_datasets(out: Path, n: int = 300, seed: int = 0) -> dict:
    """chat.jsonl (OpenAI-style messages), sharegpt.json (from/value), alpaca.csv (instruction/input/
    output), text.jsonl (plain text, some longer than seq_len)."""
    import csv

    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chat.jsonl", "w") as f:
        for _ in range(n):
            msgs = []
            for _ in range(rng.randint(1, 2)):
                msgs += [
                    {"role": "user", "content": _sent(rng, rng.randint(3, 15))},
                    {"role": "assistant", "content": _sent(rng, rng.randint(5, 45))},
                ]
            f.write(json.dumps({"messages": msgs}) + "\n")
    rows = [
        {
            "conversations": [
                {"from": "human", "value": _sent(rng, 10)},
                {"from": "gpt", "value": _sent(rng, 30)},
            ]
        }
        for _ in range(n // 2)
    ]
    (out / "sharegpt.json").write_text(json.dumps(rows))
    with open(out / "alpaca.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instruction", "input", "output"])
        w.writeheader()
        for i in range(n):
            w.writerow(
                {
                    "instruction": _sent(rng, 8),
                    "input": _sent(rng, 6) if i % 2 else "",
                    "output": _sent(rng, rng.randint(5, 50)),
                }
            )
    with open(out / "text.jsonl", "w") as f:
        for _ in range(n // 3):
            f.write(
                json.dumps({"text": " ".join(_sent(rng, 12) for _ in range(rng.randint(2, 30)))})
                + "\n"
            )
    return {k: str(out / k) for k in ("chat.jsonl", "sharegpt.json", "alpaca.csv", "text.jsonl")}
