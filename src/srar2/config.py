from __future__ import annotations
from pathlib import Path
import yaml
from .actions import ActionSpace
from .features import FEATURE_SCHEMA_SHA256
from .io import digest, file_hash


def load_config(path):
    c = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = ("task", "backbone_id", "dtype", "attention_backend", "operator", "operator_sha256", "initial_anchor",
                "gamma0", "gamma_values", "selected_heads", "target_layers", "observed_layers", "hidden_size",
                "max_new_tokens", "add_special_tokens", "scorer_version", "seed", "max_epochs", "patience", "batch_size")
    if any(k not in c for k in required):
        raise ValueError("incomplete configuration")
    space = ActionSpace(c["gamma_values"])
    if space.gamma_values[0] != c["gamma0"] or c["initial_anchor"] != 0:
        raise ValueError("initial/BASE mismatch")
    if c["backbone_id"] != "Qwen/Qwen2-7B" or c["dtype"] != "bfloat16" or c["attention_backend"] != "sdpa":
        raise ValueError("v1 requires Qwen2-7B Base BF16/native SDPA")
    if c["target_layers"] != list(range(28)) or c["observed_layers"] != [14, 27] or c["hidden_size"] != 3584:
        raise ValueError("unsupported layer/feature schema")
    if len(c["selected_heads"]) != 12 or len(set(c["selected_heads"])) != 12 or any(h < 0 or h >= 28 for h in c["selected_heads"]):
        raise ValueError("invalid frozen heads")
    operator = Path(__file__).parent / "vendor/native_operator.py"
    if file_hash(operator) != c["operator_sha256"]:
        raise ValueError("operator changed; audit required")
    if c.get("strict_gate") is not True or c["max_new_tokens"] <= 0:
        raise ValueError("invalid gate/generation configuration")
    if c.get("loss_weights") != {"regression": 1., "positive": .5, "ranking": .2}:
        raise ValueError("v1 fixes loss weights")
    return c


def code_identity():
    base = Path(__file__).parent
    return digest({str(p.relative_to(base)).replace("\\", "/"): file_hash(p) for p in sorted(base.rglob("*.py"))})


def protocol(c):
    # Location-independent identity; actual asset content is added by runtime.
    return {k: v for k, v in c.items() if k not in ("model_path", "manifests", "experiment", "runtime_status")}


def binding(c):
    return {"protocol_sha256": digest(protocol(c)), "action_schema_sha256": ActionSpace(c["gamma_values"]).sha256,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256, "code_sha256": code_identity()}


def check_binding(saved, c):
    for k, v in binding(c).items():
        if saved.get(k) != v:
            raise ValueError("checkpoint/cache binding mismatch: " + k)
