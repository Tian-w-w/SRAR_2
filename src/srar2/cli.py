from __future__ import annotations
import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
import numpy as np
import torch
from .actions import ActionSpace
from .config import binding, check_binding, load_config
from .data import prepare_short, select_rows
from .io import digest, file_hash, read_json, read_rows, write_json, write_rows
from .records import audit_manifests

ROOT = Path(__file__).resolve().parents[2]


def output_path(args, default):
    root = Path(args.project_root).resolve()
    value = Path(args.output) if args.output else root / default
    path = value.resolve()
    if root != path and root not in path.parents:
        raise ValueError("output must stay in the independent --project-root directory")
    return path


def manifests(args, c):
    result = {}
    for split in ("train", "dev"):
        value = getattr(args, split + "_manifest", None) or c["manifests"].get(split)
        if value:
            result[split] = Path(value)
    return result


def require_task(c):
    if c["task"] not in ("boolq", "sst2", "logiqa") or not c["scorer_version"]:
        raise ValueError("LITM runtime adapter/input parity remains unresolved; config is audit-only")


def get_backend(args, c):
    require_task(c)
    if not (args.model_path or c.get("model_path")):
        raise ValueError("provide local Qwen2-7B Base --model-path")
    from .runtime import QwenBackend
    return QwenBackend(c, args.model_path or c["model_path"], args.device)


def dataset(args, c, split):
    from .training import RecordDataset
    paths = manifests(args, c)
    if split not in paths:
        raise ValueError("provide --" + split + "-manifest")
    if args.max_samples and len(read_rows(paths[split])) > args.max_samples:
        raise ValueError("manifest exceeds --max-samples; use a separate smoke manifest or explicitly --max-samples 0 for full data")
    return RecordDataset(c, args.run_dir, split, paths[split])


def audit(args, c):
    paths = manifests(args, c)
    missing = []
    found = {}
    for split in ("train", "dev"):
        path = paths.get(split)
        if path is None or not path.is_file():
            missing.append(split + "_manifest")
        else:
            found[split] = read_rows(path)
    data_audit = audit_manifests(found) if found else {"status": "assets_missing", "counts": {}}
    model_path = args.model_path or c.get("model_path")
    if not model_path or not Path(model_path).is_dir():
        missing.append("local_Qwen2_7B_Base_weights_and_tokenizer")
    elif args.hash_model:
        from .runtime import asset_identity
        data_audit["model_assets"] = asset_identity(model_path)
    if not importlib.util.find_spec("transformers"):
        missing.append("transformers==4.53.2")
    if not torch.cuda.is_available():
        missing.append("CUDA_GPU_for_real_BF16_smoke")
    if c["task"] == "litm":
        missing.extend(["audited_LITM_coordinates_and_scorer", "recomputed_LITM_features_using_gamma0_0.95"])
    result = {"status": "ready" if not missing else "blocked_for_real_experiment", "task": c["task"],
              "environment": {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
                              "numpy": np.__version__, "cuda_available": torch.cuda.is_available()},
              **binding(c), "action_schema": ActionSpace(c["gamma_values"]).schema,
              "missing": missing, "data_audit": data_audit,
              "old_outputs_reused": False, "real_experiment_completed": False}
    path = output_path(args, "reports/environment_audit.json")
    write_json(path, result)
    write_json(path.parent / "data_manifest_audit.json", data_audit)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent SRAR_2 pipeline; original SRAR directories are read-only references")
    parser.add_argument("command", choices=["audit", "prepare-manifest", "smoke", "generate-counterfactuals", "extract-features",
                                          "build-labels", "train", "calibrate", "freeze", "infer", "evaluate", "analyze"])
    parser.add_argument("--config", default=str(ROOT / "configs/qwen_boolq.yaml"))
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest")
    parser.add_argument("--train-manifest")
    parser.add_argument("--dev-manifest")
    parser.add_argument("--input-rows")
    parser.add_argument("--predictions")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "validation", "formal"])
    parser.add_argument("--output")
    parser.add_argument("--run-dir", default=str(ROOT / "runs/boolq_smoke"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--reference-checkpoint")
    parser.add_argument("--fixed-checkpoints", nargs="*")
    parser.add_argument("--calibration")
    parser.add_argument("--frozen")
    parser.add_argument("--kind", choices=["joint", "fixed"], default="joint")
    parser.add_argument("--coefficient", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hash-model", action="store_true")
    args = parser.parse_args(argv)
    c = load_config(args.config)
    space = ActionSpace(c["gamma_values"])
    if args.command == "audit":
        result = audit(args, c)
    elif args.command in ("smoke", "generate-counterfactuals", "extract-features"):
        if args.split == "formal":
            raise ValueError("Formal counterfactual tables are disallowed in this pre-experiment pipeline")
        rows = select_rows(args.manifest, args.split, args.max_samples, args.shard_index, args.num_shards) if args.manifest else []
        if not rows and not args.dry_run:
            raise ValueError("provide a nonempty --manifest")
        plan = {"command": args.command, "selected_rows": len(rows), "max_samples": args.max_samples,
                "generations_without_cache": len(rows) * (1 + 3 * space.m if args.command != "smoke" else 15),
                "counterfactual_actions_per_row": 3 * space.m,
                "verified_cache_hits": 0, "pending_actions": len(rows) * 3 * space.m,
                "warning": "Cache verification requires matching model hashes; dry-run never assumes old outputs reusable",
                "initial_features_from_single_pass": True, "real_experiment_completed": False}
        if args.dry_run:
            result = plan
            write_json(output_path(args, "reports/" + args.command + "_dry_run.json"), result)
        else:
            backend = get_backend(args, c)
            if args.command == "smoke":
                from .smoke import real_smoke
                result = real_smoke(c, rows, backend)
                write_json(output_path(args, "reports/real_smoke.json"), result)
            else:
                from .pipeline import generate_rows
                result = generate_rows(c, rows, args.manifest, output_path(args, "runs/boolq_smoke"), backend, args.resume,
                                       args.shard_index, args.num_shards)
    elif args.command == "prepare-manifest":
        if args.dry_run:
            result = {"input_rows": args.input_rows, "split": args.split, "model_path": args.model_path, "loads_weights": False}
        else:
            require_task(c)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
            result = prepare_short(args.input_rows, output_path(args, f"data/{args.split}.jsonl"), c["task"], args.split, tokenizer)
    elif args.command in ("build-labels", "train", "calibrate", "analyze"):
        if args.dry_run:
            result = {"command": args.command, "run_dir": args.run_dir, "manifest_paths": {k: str(v) for k, v in manifests(args, c).items()},
                      "loads_backbone": False, "training_kind": args.kind, "coefficient": args.coefficient}
        elif args.command == "build-labels":
            data = dataset(args, c, args.split)
            result = {"split": args.split, "n": len(data), "complete_actions": len(data) * 3 * space.m,
                      "oracle": __import__("srar2.metrics", fromlist=["oracle"]).oracle(data.initial, data.utilities)}
            write_json(output_path(args, "reports/labels_" + args.split + ".json"), result)
        elif args.command == "train":
            from .training import train
            result = train(c, dataset(args, c, "train"), dataset(args, c, "dev"),
                           output_path(args, "runs/joint.pt"), args.device, args.kind, args.coefficient)
        elif args.command == "calibrate":
            from .training import calibrate_checkpoint
            result = calibrate_checkpoint(c, dataset(args, c, "dev"), args.checkpoint, output_path(args, "runs/calibration.json"))
        else:
            from .training import analyze
            result = analyze(c, dataset(args, c, "dev"), args.checkpoint, args.reference_checkpoint, args.fixed_checkpoints)
            write_json(output_path(args, "reports/validation_comparison.json"), result)
    elif args.command == "freeze":
        from .training import load_checkpoint
        _, saved = load_checkpoint(args.checkpoint, c)
        cal = read_json(args.calibration)
        check_binding(cal, c)
        if cal["checkpoint_sha256"] != file_hash(args.checkpoint) or cal["dev_manifest_sha256"] != saved["dev_manifest_sha256"]:
            raise ValueError("calibration/checkpoint mismatch")
        if cal["split"] not in ("dev", "validation") or saved["kind"] != "joint":
            raise ValueError("freeze requires calibrated joint model")
        result = {**binding(c), "checkpoint_sha256": file_hash(args.checkpoint), "calibration_sha256": file_hash(args.calibration),
                  "tau": cal["selected"]["tau"], "gamma0": c["gamma0"], "action_schema": space.schema,
                  "runtime_identity": saved["runtime_identity"], "formal_used": False,
                  "train_manifest_sha256": saved["train_manifest_sha256"], "dev_manifest_sha256": saved["dev_manifest_sha256"]}
        write_json(output_path(args, "runs/frozen.json"), result)
    elif args.command == "infer":
        from .training import load_checkpoint
        from .inference import infer_srar2
        from .pipeline import input_only
        frozen = read_json(args.frozen)
        check_binding(frozen, c)
        if digest(frozen["action_schema"]) != space.sha256 or frozen["checkpoint_sha256"] != file_hash(args.checkpoint):
            raise ValueError("frozen action schema/checkpoint changed")
        rows = select_rows(args.manifest, args.split, args.max_samples, args.shard_index, args.num_shards)
        if args.dry_run:
            result = {"rows": len(rows), "min_generations": len(rows), "max_generations": 2 * len(rows), "uses_gold": False}
        else:
            backend = get_backend(args, c)
            if backend.identity != frozen["runtime_identity"]:
                raise ValueError("inference runtime differs from frozen training data")
            model, saved = load_checkpoint(args.checkpoint, c)
            output = output_path(args, "runs/predictions.jsonl")
            predictions = read_rows(output) if output.exists() and args.resume else []
            if output.exists() and not args.resume:
                raise FileExistsError("predictions exist; use --resume or new output")
            done = {r["row_uid"] for r in predictions}
            for r in predictions:
                if r["frozen_sha256"] != file_hash(args.frozen) or r["manifest_sha256"] != file_hash(args.manifest):
                    raise ValueError("stale prediction resume")
            for row in rows:
                if row["row_uid"] in done:
                    continue
                final, decision = infer_srar2(input_only(row), backend, model, frozen, space, saved["scalar_mean"], saved["scalar_std"])
                predictions.append({"row_uid": row["row_uid"], "source_question_id": row["source_question_id"],
                    "split": args.split, "manifest_sha256": file_hash(args.manifest), "frozen_sha256": file_hash(args.frozen),
                    "input_ids_sha256": row["input_ids_sha256"], "output": final, "decision": decision, "runtime_identity": backend.identity})
                write_rows(output, predictions)
            result = {"predictions": len(predictions), "path": str(output)}
    else:  # scoring is explicitly downstream of online inference
        from .pipeline import score
        from .metrics import transitions, primary
        rows, predictions = read_rows(args.manifest), read_rows(args.predictions)
        truth = {r["row_uid"]: r for r in rows}
        if len(truth) != len(rows) or len(predictions) != len(rows) or {r["row_uid"] for r in predictions} != set(truth):
            raise ValueError("evaluation requires complete paired manifest rows")
        initial, final, triggered = [], [], []
        for p in predictions:
            r = truth[p["row_uid"]]
            if p["manifest_sha256"] != file_hash(args.manifest) or p["input_ids_sha256"] != r["input_ids_sha256"]:
                raise ValueError("evaluation pairing/hash mismatch")
            initial.append(score(c["task"], p["decision"]["initial"]["prediction_raw"], r["gold"])[1])
            final.append(score(c["task"], p["output"]["prediction_raw"], r["gold"])[1])
            triggered.append(p["decision"]["triggered"])
        result = {"metric": primary(final), "initial_metric": primary(initial), "transitions": transitions(initial, final, triggered)}
        write_json(output_path(args, "reports/evaluation.json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
