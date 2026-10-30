"""Sequential GPU pre-experiment using existing SRAR manifests. Never reads Formal.

python tools/run_boolq_smoke.py --legacy-root /root/zerotuning_adaptive --model-path /path/Qwen2-7B
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-root", type=Path, required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--output", type=Path, default=Path("runs/boolq_smoke"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = (root / args.output).resolve()
    if root not in out.parents or args.samples < 1:
        raise ValueError("output must be inside this new repo and samples must be positive")
    legacy = args.legacy_root / "runs/stage26q2_short_prepare/manifests_qwen2_base"
    raw = {"train": legacy / "boolq_train_effective.jsonl", "dev": legacy / "boolq_validation.jsonl"}
    commands = []
    for split, source in raw.items():
        commands.append(["prepare-manifest", "--input-rows", str(source), "--split", split,
                         "--model-path", args.model_path, "--output", str(out / f"{split}_full.jsonl")])
    # Subsets preserve first rows in the original ordered membership; no split creation.
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    def run(command):
        cmd = [sys.executable, "-m", "srar2", *command]
        print("COMMAND", cmd, flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=root, env=env, check=True)
    for cmd in commands:
        run(cmd)
    if not args.dry_run:
        sys.path.insert(0, str(root / "src"))
        from srar2.io import read_rows, write_rows
        from srar2.records import audit_manifests
        full = {s: read_rows(out / f"{s}_full.jsonl") for s in raw}
        audit_manifests(full)
        for split in raw:
            write_rows(out / f"{split}.jsonl", full[split][:args.samples])
    common = ["--run-dir", str(out), "--train-manifest", str(out / "train.jsonl"), "--dev-manifest", str(out / "dev.jsonl"),
              "--max-samples", str(args.samples)]
    run(["audit", *common, "--model-path", args.model_path, "--hash-model", "--output", str(out / "asset_audit.json")])
    run(["smoke", "--manifest", str(out / "dev.jsonl"), "--model-path", args.model_path,
         "--max-samples", "2", "--output", str(out / "real_smoke.json")])
    for split in raw:
        run(["generate-counterfactuals", "--manifest", str(out / f"{split}.jsonl"), "--split", split,
             "--model-path", args.model_path, "--max-samples", str(args.samples), "--output", str(out)] + (["--resume"] if args.resume else []))
    for split in raw:
        run(["build-labels", *common, "--split", split, "--output", str(out / f"labels_{split}.json")])
    # Labels/oracle are written BEFORE training; this script never tunes the grid.
    for kind, coefficient, name in [("joint", 0, "joint"), ("fixed", 0, "base"), ("fixed", 1, "low"), ("fixed", 2, "high")]:
        cp = out / (name + ".pt")
        if not (args.resume and cp.exists()):
            run(["train", *common, "--kind", kind, "--coefficient", str(coefficient), "--output", str(cp), "--device", "cpu"])
    run(["calibrate", *common, "--checkpoint", str(out / "joint.pt"), "--output", str(out / "calibration.json")])
    run(["analyze", *common, "--checkpoint", str(out / "joint.pt"), "--reference-checkpoint", str(out / "base.pt"),
         "--fixed-checkpoints", *[str(out / (x + ".pt")) for x in ("base", "low", "high")],
         "--output", str(out / "validation_comparison.json")])
    print("Smoke pipeline complete" if not args.dry_run else "Dry-run only; no model/data read or generation performed")


if __name__ == "__main__":
    main()
