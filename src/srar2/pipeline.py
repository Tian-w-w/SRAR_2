from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from .actions import ANCHORS, ActionSpace
from .config import binding
from .features import FEATURE_SCHEMA_SHA256
from .io import atomic_bytes, digest, file_hash, read_json, write_json
from .records import cache_key, validate_record
from .vendor.scoring import extract_label_from_response, extract_answer_from_response


def score(task, prediction, gold):
    if task == "boolq":
        parsed = extract_label_from_response(prediction, [" False", " True"])
    elif task == "sst2":
        parsed = extract_label_from_response(prediction, [" negative", " positive"])
    elif task == "logiqa":
        parsed = extract_answer_from_response(prediction)
    else:
        raise ValueError("task scorer not audited: " + task)
    return parsed, float(parsed == int(gold))


def input_only(row):
    return {k: row[k] for k in ("raw_prompt", "prompt_tokens", "input_ids_sha256", "controls", "query_token_span", "row_uid")}


def record_path(output, row):
    return Path(output) / "records" / row["split"] / (digest(row["row_uid"]) + ".json")


def generate_rows(config, rows, manifest_path, output, backend, resume=False, shard_index=0, num_shards=1):
    space = ActionSpace(config["gamma_values"])
    output = Path(output)
    generation = {"do_sample": False, "num_beams": 1, "max_new_tokens": config["max_new_tokens"], "use_cache": True,
                  "generation_config_sha256": backend.assets["files"].get("generation_config.json")}
    identity = backend.identity
    run = {"identity": identity, "manifest_sha256": file_hash(manifest_path)}
    split = rows[0]["split"] if rows else "empty"
    provenance = output / ("run_identity_" + split + ".json")
    if provenance.exists() and read_json(provenance) != run:
        raise ValueError("run identity changed; select a NEW output directory")
    write_json(provenance, run)
    stats = {"rows": len(rows), "cache_hits": 0, "generations": 0, "failures": 0, "split": split,
             "shard_index": shard_index, "num_shards": num_shards}
    budget_path = output / f"budget_{split}_{shard_index}_of_{num_shards}.json"
    for row in rows:
        if row["split"] not in ("train", "dev", "validation"):
            raise ValueError("counterfactual generation disallows Formal before freeze")
        target = record_path(output, row)
        if target.exists():
            if not resume:
                raise FileExistsError("record exists; use --resume or a new run")
            record = read_json(target)
            validate_record(record, space)
            if record["manifest_sha256"] != file_hash(manifest_path) or record["runtime_identity"] != identity:
                raise ValueError("stale record")
            if file_hash(output / record["feature_file"]) != record["feature_sha256"]:
                raise ValueError("feature artifact changed")
            stats["cache_hits"] += 1 + 3 * space.m
            continue
        base = {"row_uid": row["row_uid"], "input_ids_sha256": row["input_ids_sha256"],
                **{k: identity[k] for k in ("model_sha256", "tokenizer_sha256", "protocol_sha256")},
                "operator_sha256": config["operator_sha256"], "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                "generation": generation, "scorer_version": config["scorer_version"]}
        def get(anchor, gamma, capture=False):
            metadata = {**base, "anchor_token_index": anchor, "gamma": gamma}
            key = cache_key(metadata)
            cache = output / "cache" / (key + ".json")
            if cache.exists() and not capture:
                saved = read_json(cache)
                if saved["metadata"] != metadata:
                    raise ValueError("cache metadata mismatch")
                stats["cache_hits"] += 1
                return saved["generation"], None
            value, features = backend.generate(input_only(row), anchor, gamma, capture)
            stats["generations"] += 1
            write_json(cache, {"metadata": metadata, "generation": value})
            return value, features
        try:
            initial, features = get(0, config["gamma0"], True)
            initial["prediction_parsed"], initial["utility"] = score(config["task"], initial["prediction_raw"], row["gold"])
            feature_file = f"features/{row['split']}/{digest(row['row_uid'])}.npz"
            atomic_bytes(output / feature_file, lambda f: np.savez(f, vectors=features[0], scalars=features[1]))
            actions = []
            for aid in space.order:
                i, j = space.indices(aid)
                generated, _ = get(row["controls"][ANCHORS[i]], space.gamma_values[j])
                parsed, utility = score(config["task"], generated["prediction_raw"], row["gold"])
                actions.append({**generated, "action_id": aid, "anchor_id": i, "coefficient_id": j,
                                "prediction_parsed": parsed, "utility": utility, "advantage": utility - initial["utility"],
                                "generation_status": "complete"})
            record = {"schema_version": "srar2-v1", "dataset": config["task"], "split": row["split"],
                      "row_uid": row["row_uid"], "source_question_id": row["source_question_id"],
                      "manifest_sha256": file_hash(manifest_path), "input_ids_sha256": row["input_ids_sha256"],
                      **binding(config), "runtime_identity": identity, "gamma_values": list(space.gamma_values),
                      "anchor_token_indices": row["controls"], "initial": initial, "actions": actions,
                      "feature_file": feature_file, "feature_sha256": file_hash(output / feature_file)}
            validate_record(record, space)
            write_json(target, record)
        except Exception as exc:
            stats["failures"] += 1
            write_json(output / "failures" / row["split"] / (digest(row["row_uid"]) + ".json"),
                       {"row_uid": row["row_uid"], "error_type": type(exc).__name__, "error": str(exc), "time": time.time()})
            write_json(budget_path, stats)
            raise
        write_json(budget_path, stats)
    write_json(budget_path, stats)
    return stats
