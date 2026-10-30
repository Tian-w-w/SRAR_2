from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .actions import ActionSpace
from .calibration import policy, calibrate
from .config import binding, check_binding
from .features import validate_features
from .io import atomic_bytes, digest, file_hash, read_json, write_json
from .losses import joint_loss, sample_weights
from .metrics import oracle, primary, transitions
from .model import Router
from .records import audit_manifests, validate_record


class RecordDataset(Dataset):
    def __init__(self, config, run_dir, split, expected_manifest=None):
        self.config, self.root, self.split = config, Path(run_dir), split
        self.space = ActionSpace(config["gamma_values"])
        files = sorted((self.root / "records" / split).glob("*.json"))
        self.records = [read_json(p) for p in files]
        if not self.records:
            raise ValueError("no complete records for " + split)
        if expected_manifest is None:
            raise ValueError("explicit manifest required; incomplete shards cannot be silently trained")
        from .io import read_rows
        manifest = read_rows(expected_manifest)
        audit_manifests({split: manifest})
        if {r["row_uid"] for r in self.records} != {r["row_uid"] for r in manifest} or len(self.records) != len(manifest):
            raise ValueError("record coverage does not match entire manifest; create a separate smoke manifest")
        self.manifest_sha256 = file_hash(expected_manifest)
        self.manifest_rows = manifest
        self.vectors, self.scalars, self.advantages, self.utilities, self.equivalence = [], [], [], [], []
        identity = self.records[0]["runtime_identity"]
        for r in self.records:
            check_binding(r, config)
            if r["split"] != split or r["manifest_sha256"] != self.manifest_sha256 or r["runtime_identity"] != identity:
                raise ValueError("mixed dataset provenance")
            a, u, mask = validate_record(r, self.space)
            feature = self.root / r["feature_file"]
            if file_hash(feature) != r["feature_sha256"]:
                raise ValueError("feature checksum mismatch")
            with np.load(feature, allow_pickle=False) as f:
                v, s = f["vectors"].astype(np.float32), f["scalars"].astype(np.float32)
            validate_features(v, s)
            self.vectors.append(v); self.scalars.append(s); self.advantages.append(a); self.utilities.append(u)
            self.equivalence.append(self.space.equivalent_ids(r["anchor_token_indices"]))
        self.scalars = torch.tensor(np.stack(self.scalars))
        self.advantages = torch.tensor(np.stack(self.advantages))
        self.utilities = np.stack(self.utilities)
        self.initial = np.asarray([r["initial"]["utility"] for r in self.records])
        self.runtime_identity = identity

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return torch.from_numpy(self.vectors[i]), self.scalars[i], self.advantages[i], torch.from_numpy(self.equivalence[i])


def load_checkpoint(path, c):
    # Checkpoints are own local artifacts, never download/unpickle untrusted files.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    check_binding(saved, c)
    if digest(saved["action_schema"]) != ActionSpace(c["gamma_values"]).sha256:
        raise ValueError("checkpoint action order/grid altered")
    model = Router(m=3 if saved["kind"] == "joint" else 1, dropout=c["dropout"], joint=saved["kind"] == "joint")
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), saved


def predict(model, dataset, mean, std, device="cpu"):
    model.to(device).eval()
    output = []
    with torch.inference_mode():
        for v, s, _, _ in DataLoader(dataset, batch_size=64):
            output.append(model(v.to(device), (s.to(device) - mean.to(device)) / std.to(device)).cpu().numpy())
    scores = np.concatenate(output)
    if not model.joint:
        expanded = np.zeros((len(scores), 3, dataset.space.m), dtype=scores.dtype)
        expanded[:] = scores
        scores = expanded
    return scores


def train(c, train_data, dev_data, output, device="cpu", kind="joint", coefficient=0):
    if train_data.split != "train" or dev_data.split not in ("dev", "validation"):
        raise ValueError("train/dev roles required")
    audit_manifests({train_data.split: train_data.manifest_rows, dev_data.split: dev_data.manifest_rows})
    if train_data.runtime_identity != dev_data.runtime_identity:
        raise ValueError("train/dev runtime mismatch")
    if kind not in ("joint", "fixed") or not 0 <= coefficient < train_data.space.m:
        raise ValueError("invalid training kind/coefficient")
    seed = int(c["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = Router(m=train_data.space.m if kind == "joint" else 1, dropout=c["dropout"], joint=kind == "joint").to(device)
    mean = train_data.scalars.mean((0, 1))[None, None]
    std = train_data.scalars.std((0, 1), unbiased=False).clamp_min(1e-6)[None, None]
    advantages = train_data.advantages if kind == "joint" else train_data.advantages[:, :, coefficient:coefficient + 1]
    weights = sample_weights(advantages)
    sampler = WeightedRandomSampler(weights, len(train_data), replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(train_data, batch_size=c["batch_size"], sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    allowed = None if kind == "joint" else [train_data.space.action_id(i, coefficient) for i in range(3)]
    best, stale, history = -float("inf"), 0, []
    output = Path(output)
    if output.exists():
        raise FileExistsError("training checkpoint exists; use a new output name")
    for epoch in range(1, c["max_epochs"] + 1):
        model.train(); losses = []
        for v, s, a, eq in loader:
            if kind != "joint":
                a, eq = a[:, :, coefficient:coefficient + 1], eq[:, :, coefficient:coefficient + 1]
            scores = model(v.to(device), (s.to(device) - mean.to(device)) / std.to(device))
            loss = joint_loss(scores, a.to(device), equivalence=eq.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        scores = predict(model, dev_data, mean, std, device)
        final, _, _, _ = policy(scores, dev_data.initial, dev_data.utilities, dev_data.space, 0., allowed)
        gain = primary(final) - primary(dev_data.initial)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_tau0_gain": gain})
        if gain > best:
            best, stale = gain, 0
            saved = {**binding(c), "action_schema": dev_data.space.schema, "kind": kind, "coefficient": coefficient,
                     "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     "scalar_mean": mean, "scalar_std": std, "epoch": epoch, "seed": seed,
                     "train_manifest_sha256": train_data.manifest_sha256, "dev_manifest_sha256": dev_data.manifest_sha256,
                     "runtime_identity": train_data.runtime_identity, "validation_tau0_gain": gain,
                     "parameter_count": sum(p.numel() for p in model.parameters()), "initialized_from_scratch": True}
            atomic_bytes(output, lambda f: torch.save(saved, f))
        else:
            stale += 1
        write_json(output.with_suffix(".history.json"), {"epochs": history, "selection": "dev tau=0, earliest epoch on tie"})
        if stale >= c["patience"]:
            break
    return {"checkpoint": str(output), "checkpoint_sha256": file_hash(output), "best_dev_gain": best, "epochs": len(history)}


def calibrate_checkpoint(c, data, checkpoint, output):
    model, saved = load_checkpoint(checkpoint, c)
    if data.runtime_identity != saved["runtime_identity"] or data.manifest_sha256 != saved["dev_manifest_sha256"]:
        raise ValueError("calibration must use bound validation set/runtime")
    scores = predict(model, data, saved["scalar_mean"], saved["scalar_std"])
    allowed = None if saved["kind"] == "joint" else [data.space.action_id(i, saved["coefficient"]) for i in range(3)]
    result = calibrate(scores, data.initial, data.utilities, data.space, data.split, allowed)
    result.update({**binding(c), "checkpoint_sha256": file_hash(checkpoint), "dev_manifest_sha256": data.manifest_sha256,
                   "runtime_identity": saved["runtime_identity"], "allowed": allowed})
    write_json(output, result)
    return result["selected"]


def analyze(c, data, joint_checkpoint, reference_checkpoint=None, fixed_checkpoints=None):
    if data.split not in ("dev", "validation"):
        raise ValueError("pre-experiment analysis requires non-Formal validation")
    model, saved = load_checkpoint(joint_checkpoint, c)
    if saved["kind"] != "joint":
        raise ValueError("joint checkpoint required")
    scores = predict(model, data, saved["scalar_mean"], saved["scalar_std"])
    report = {"split": data.split, "oracle": oracle(data.initial, data.utilities), "policies": {},
              "warning": "Validation diagnostics only; no formal superiority claim", "initial_metric": primary(data.initial)}
    def add(name, s, allowed=None, sequential=None):
        cal = calibrate(s, data.initial, data.utilities, data.space, data.split, allowed=allowed, sequential_scores=sequential)
        tau = cal["selected"]["tau"]
        final, t, choice, _ = policy(s, data.initial, data.utilities, data.space, tau, allowed, sequential)
        item = {**cal["selected"], "average_passes": 1 + float(t.mean()),
                "transitions": transitions(data.initial, final, t),
                "selected_action_counts": {aid: int(((choice == i) & t).sum()) for i, aid in enumerate(data.space.order)}}
        report["policies"][name] = item
        return item
    add("J", scores)
    add("J-BASE-mask", scores, [a + ":BASE" for a in ("RE", "RM", "RL")])
    final, t, _, _ = policy(scores, data.initial, data.utilities, data.space, np.finfo(np.float64).min)
    report["policies"]["J-Always"] = {"metric": primary(final), "trigger_count": int(t.sum())}
    gamma_only = [add("B4-" + a, scores, [a + ":" + g for g in ("BASE", "LOW", "HIGH")[:data.space.m]]) for a in ("RE", "RM", "RL")]
    report["policies"]["B4-selected"] = max(gamma_only, key=lambda r: (r["metric"], -r["trigger_count"], r["tau"]))
    if reference_checkpoint:
        ref, rs = load_checkpoint(reference_checkpoint, c)
        if rs["kind"] != "fixed" or rs["coefficient"] != 0:
            raise ValueError("B2 requires independently trained fixed BASE model")
        ref_scores = predict(ref, data, rs["scalar_mean"], rs["scalar_std"])
        add("B2-SRAR-ref", ref_scores, [a + ":BASE" for a in ("RE", "RM", "RL")])
        add("B5-Sequential", scores, sequential=ref_scores)
    if fixed_checkpoints:
        if len(fixed_checkpoints) != data.space.m:
            raise ValueError("B3 requires one independently trained model per coefficient")
        fixed = []
        for j, cp in enumerate(fixed_checkpoints):
            fm, fs = load_checkpoint(cp, c)
            if fs["kind"] != "fixed" or fs["coefficient"] != j:
                raise ValueError("fixed checkpoint order mismatch")
            fixed.append(add("B3-" + str(j), predict(fm, data, fs["scalar_mean"], fs["scalar_std"]),
                             [data.space.action_id(i, j) for i in range(3)]))
        report["policies"]["B3-selected"] = max(fixed, key=lambda r: (r["metric"], -r["trigger_count"], r["tau"]))
    return report
