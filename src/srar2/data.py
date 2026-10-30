from __future__ import annotations
from pathlib import Path
import numpy as np
from .io import digest, file_hash, read_rows, write_rows
from .records import audit_manifests
from .vendor.prompts import prompt_and_chars, overlap, anchors


def prepare_short(source, target, task, split, tokenizer):
    if task not in ("sst2", "boolq", "logiqa"):
        raise ValueError("LITM requires a separately audited document coordinate adapter")
    output = []
    for index, row in enumerate(read_rows(source)):
        if "raw_prompt" in row:
            # Import prepared historical rows without rebuilding their prompt.
            prompt = row["raw_prompt"]
            if row.get("add_special_tokens", False):
                raise ValueError("short-task manifest unexpectedly adds special tokens")
            encoded = tokenizer(prompt, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
            controls, qspan = row["controls"], row["query_token_span"]
            if row.get("prompt_tokens", len(encoded["input_ids"])) != len(encoded["input_ids"]):
                raise ValueError("historical token count mismatch")
            if "context_token_span" in row:
                expected, _ = anchors(*row["context_token_span"])
                if any(controls[k] != expected[k] for k in expected):
                    raise ValueError("historical anchors do not match frozen coordinate definition")
        else:
            prompt, context_chars, query_chars, _ = prompt_and_chars(task, row)
            encoded = tokenizer(prompt, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
            start, end = overlap(encoded["offset_mapping"], *context_chars)
            controls, _ = anchors(start, end)
            qspan = overlap(encoded["offset_mapping"], *query_chars)
        token_hash = digest(encoded["input_ids"])
        if row.get("input_ids_sha256", token_hash) != token_hash:
            raise ValueError("historical token IDs mismatch")
        # Rows are unique; group repeats are intentionally retained.
        uid = str(row.get("row_uid", row.get("row_id", row.get("sample_id", f"{split}:{index:08d}"))))
        if row.get("source_question_id") is not None:
            group = str(row["source_question_id"])
        elif "question" in row:
            group = digest(" ".join(row["question"].lower().split()))
        elif task == "sst2" and "sentence" in row:
            group = digest(" ".join(row["sentence"].lower().split()))
        elif "raw_prompt" in row:
            # Prepared Stage26 manifests omit question text. Recover it from
            # their audited token span; SST-2 groups by the sentence/context.
            span = row.get("context_token_span") if task == "sst2" else qspan
            if span is None:
                raise ValueError("prepared SST-2 requires context span for sentence grouping")
            offsets = [(a, b) for a, b in encoded["offset_mapping"][span[0]:span[1]] if b > a]
            if not offsets:
                raise ValueError("cannot recover source question text")
            text = prompt[min(a for a, _ in offsets):max(b for _, b in offsets)]
            group = digest(" ".join(text.lower().split()))
        else:
            raise ValueError("prepared row requires source_question_id for leakage audit")
        gold = row.get("gold", row.get("label", row.get("answer")))
        if gold is None:
            raise ValueError("missing gold; do not silently substitute")
        result = {"row_uid": uid, "source_question_id": group, "split": split, "raw_prompt": prompt,
                  "input_ids_sha256": token_hash, "prompt_tokens": len(encoded["input_ids"]),
                  "controls": {k: int(controls[k]) for k in ("RE", "RM", "RL")},
                  "query_token_span": list(qspan), "gold": int(gold), "source_sha256": file_hash(source)}
        result["source_row_index"] = index
        result["historical_sample_id"] = row.get("sample_id")
        result["historical_split"] = row.get("split")
        validate_input(result)
        output.append(result)
    audit_manifests({split: output})
    write_rows(target, output)
    return {"n": len(output), "path": str(target), "sha256": file_hash(target)}


def validate_input(row):
    length = row["prompt_tokens"]
    if length < 2 or not 0 <= row["query_token_span"][0] < row["query_token_span"][1] <= length:
        raise ValueError("invalid query span")
    if any(not 0 <= row["controls"][k] < length for k in ("RE", "RM", "RL")):
        raise ValueError("invalid token anchors")
    if not row.get("input_ids_sha256") or not row.get("row_uid"):
        raise ValueError("missing input hash or manifest UID")


def select_rows(path, split, max_samples, shard_index=0, num_shards=1):
    if not 0 <= shard_index < num_shards or max_samples < 0:
        raise ValueError("invalid shard/limit")
    rows = read_rows(path)
    if not rows:
        raise ValueError("empty manifest")
    audit_manifests({split: rows})
    for row in rows:
        validate_input(row)
    if max_samples:
        rows = rows[:max_samples]
    return [r for i, r in enumerate(rows) if i % num_shards == shard_index]
