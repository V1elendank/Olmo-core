"""
Tokenize a small slice of an HF chat dataset into the OLMo-core SFT format:

    <out>/token_ids_part_0000.npy   raw uint32 memmap of token ids (docs separated by EOS)
    <out>/labels_mask_0000.npy      raw bool memmap, True where the loss should be computed
                                    (assistant turns only)

This mirrors open-instruct's ``convert_sft_data_for_olmocore.py`` but has no extra deps
beyond ``transformers`` + ``datasets``. CPU only.

    python uva/prep_sft_data.py --out /bigtemp/$USER/olmo/data/tulu3-sample --num_examples 5000
"""

import argparse
import json
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

EOS_ID = 100257  # dolma2 tokenizer <|endoftext|>


def chat_ids(tok, messages, add_generation_prompt):
    out = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=False
    )
    if isinstance(out, dict) or hasattr(out, "keys"):  # older/newer transformers differ
        out = out["input_ids"]
    return list(out)


def encode(tok, messages, max_len):
    ids = chat_ids(tok, messages, add_generation_prompt=False)
    if ids and ids[-1] != EOS_ID:
        ids = ids + [EOS_ID]
    mask = np.zeros(len(ids), dtype=np.bool_)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = len(chat_ids(tok, messages[:i], add_generation_prompt=True))
        end = len(chat_ids(tok, messages[: i + 1], add_generation_prompt=False))
        mask[start:end] = True
    mask[-1] = True  # always learn to emit EOS at the end of the conversation
    if len(ids) > max_len:
        return None, None
    return np.asarray(ids, dtype=np.uint32), mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    p.add_argument("--split", default="train")
    p.add_argument("--tokenizer", default="allenai/OLMo-2-1124-7B-SFT")
    p.add_argument("--num_examples", type=int, default=5000)
    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tok.eos_token_id == EOS_ID, f"unexpected eos id {tok.eos_token_id}"

    ds = load_dataset(args.dataset, split=args.split, streaming=True).shuffle(
        seed=0, buffer_size=20_000
    )

    all_ids, all_mask = [], []
    kept = skipped = 0
    for ex in ds:
        if kept >= args.num_examples:
            break
        msgs = [{"role": m["role"], "content": m["content"]} for m in ex["messages"]]
        if not any(m["role"] == "assistant" for m in msgs):
            skipped += 1
            continue
        ids, mask = encode(tok, msgs, args.max_len)
        if ids is None:
            skipped += 1
            continue
        all_ids.append(ids)
        all_mask.append(mask)
        kept += 1
        if kept % 1000 == 0:
            print(f"tokenized {kept} examples", flush=True)

    ids = np.concatenate(all_ids)
    mask = np.concatenate(all_mask)
    assert ids.shape == mask.shape

    # Same raw format np.memmap uses (what OLMo-core reads), despite the .npy extension.
    ids_mm = np.memmap(
        f"{args.out}/token_ids_part_0000.npy", dtype=np.uint32, mode="w+", shape=ids.shape
    )
    ids_mm[:] = ids
    ids_mm.flush()
    mask_mm = np.memmap(
        f"{args.out}/labels_mask_0000.npy", dtype=np.bool_, mode="w+", shape=mask.shape
    )
    mask_mm[:] = mask
    mask_mm.flush()

    stats = {
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "examples": kept,
        "skipped": skipped,
        "total_tokens": int(ids.size),
        "trainable_tokens": int(mask.sum()),
        "seqs_of_4096_after_packing_approx": int(ids.size // 4096),
    }
    with open(f"{args.out}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    # Save the tokenizer next to the data for later HF conversion / inference.
    tok.save_pretrained(f"{args.out}/tokenizer")


if __name__ == "__main__":
    main()
