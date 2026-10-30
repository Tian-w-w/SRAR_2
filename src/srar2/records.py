from __future__ import annotations
import numpy as np
from .io import digest

CACHE_FIELDS = ("row_uid", "input_ids_sha256", "model_sha256", "tokenizer_sha256", "operator_sha256",
                "protocol_sha256", "feature_schema_sha256", "anchor_token_index", "gamma", "generation", "scorer_version")


def cache_key(metadata):
    if any(k not in metadata for k in CACHE_FIELDS):
        raise ValueError("incomplete cache provenance")
    return digest({k: metadata[k] for k in CACHE_FIELDS})


def audit_manifests(splits):
    row_seen, group_split, prompt_split = set(), {}, {}
    counts = {}
    for split, rows in splits.items():
        counts[split] = len(rows)
        for row in rows:
            uid = row["row_uid"]
            group = row["source_question_id"]
            prompt = digest(row["raw_prompt"])
            if not uid or not group or (split, uid) in row_seen:
                raise ValueError("empty identifier or duplicate manifest row")
            if row.get("split") != split:
                raise ValueError("manifest split mismatch")
            row_seen.add((split, uid))
            for key, registry in ((group, group_split), (prompt, prompt_split)):
                if key in registry and registry[key] != split:
                    raise ValueError("cross-split question/prompt leakage")
                registry[key] = split
    return {"counts": counts, "row_count": len(row_seen), "groups": len(group_split), "cross_split_overlap": 0}


def validate_record(record, space, allow_missing=False):
    if record["action_schema_sha256"] != space.sha256:
        raise ValueError("action schema/grid/order mismatch")
    if record["initial"]["gamma"] != space.gamma_values[0] or record["initial"]["anchor_token_index"] != 0:
        raise ValueError("initial reference changed")
    expected = set(space.order)
    actions = record["actions"]
    if len(actions) != len(expected) or {a["action_id"] for a in actions} != expected:
        raise ValueError("missing/duplicate action mapping")
    utilities = np.zeros((3, space.m), np.float32)
    mask = np.zeros_like(utilities, bool)
    initial = float(record["initial"]["utility"])
    if not np.isfinite(initial):
        raise ValueError("non-finite initial utility")
    equivalents = {}
    for a in actions:
        i, j = space.indices(a["action_id"])
        if (a["anchor_id"], a["coefficient_id"], a["gamma"]) != (i, j, space.gamma_values[j]):
            raise ValueError("action metadata mismatch")
        if a["anchor_token_index"] != record["anchor_token_indices"][("RE", "RM", "RL")[i]]:
            raise ValueError("anchor mismatch")
        if a["generation_status"] != "complete":
            if not allow_missing:
                raise ValueError("incomplete action")
            continue
        u = float(a["utility"])
        if not np.isfinite(u) or not np.isclose(a["advantage"], u - initial):
            raise ValueError("invalid advantage")
        key = ("identity",) if a["gamma"] == 1.0 else (a["anchor_token_index"], a["gamma"])
        value = (a["prediction_raw"], u)
        if key in equivalents and equivalents[key] != value:
            raise ValueError("conflicting equivalent actions")
        equivalents[key] = value
        utilities[i, j], mask[i, j] = u, True
    return utilities - initial, utilities, mask
