"""One-time read-only legacy capture. Run from any directory; writes only srar2."""
import ast
import json
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

selected = ["SRAR_2.md", "stage26q2_short_base_srar.py", "stage26q2_short_prepare.py",
            "srar_paper_export/scripts/qwen2_zerotuning_sdpa_native_semantics_v2.py",
            "srar_paper_export/scripts/stage26q2_litm_srar_features.py",
            "srar_share_package_20261020/README.md", "srar_share_package_20261020/RELEASE_SCOPE.md",
            "srar_share_package_20261020/training_calibration/TRAINING_CALIBRATION_CHAIN.md",
            "srar_share_package_20261020/main_experiments/MAIN_RESULTS_CHAIN.md",
            "srar_share_package_20261020/analyses/ANALYSES_CHAIN.md"]
res = "srar_share_package_20261020/resources/zerotuning_adaptive/runs/"
for task in ("sst2", "boolq", "logiqa"):
    selected.append(res + f"stage26q2_short_base_srar/{task}/selector/frozen_config.json")
for task in ("litm", "babilong", "ruler"):
    selected.append(res + f"stage26q2_{task}_srar/frozen_config_parent.json")
parser = "srar_share_package_20261020/main_experiments/source/ZeroTuning_official/core algorithms/eval_classification_final.py"
selected.append(parser)
vendor = HERE / "src/srar2/vendor"
vendor.mkdir(parents=True, exist_ok=True)
(vendor / "__init__.py").write_text('"""Frozen original operator and AST-extracted scoring helpers. See provenance."""\n', encoding="utf-8")
operator = ROOT / selected[3]
(vendor / "native_operator.py").write_bytes(operator.read_bytes())
nodes = ast.parse((ROOT / parser).read_text(encoding="utf-8")).body
names = ("extract_label_from_response", "extract_answer_from_response")
(vendor / "scoring.py").write_text("import re\n\n" + "\n\n".join(ast.unparse(n) for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names) + "\n", encoding="utf-8")
# Capture the exact short-task prompt constructor and coordinate function, without old path/import side effects.
nodes = ast.parse((ROOT / "stage26q2_short_prepare.py").read_text(encoding="utf-8")).body
(vendor / "prompts.py").write_text("from __future__ import annotations\n\n" + "\n\n".join(ast.unparse(n) for n in nodes if isinstance(n, ast.FunctionDef) and n.name in ("prompt_and_chars", "anchors", "overlap")) + "\n", encoding="utf-8")
nodes = ast.parse((ROOT / "stage26q2_short_base_srar.py").read_text(encoding="utf-8")).body
header = '''from __future__ import annotations
import math, time
import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from . import native_operator as op
from . import scoring
LAYERS=(14,27)
HIDDEN=3584
ALT=("RE","RM","RL")
MAX_NEW={"sst2":1,"boolq":8,"logiqa":8}
MODEL_ID="Qwen/Qwen2-7B"
def _official_eval(): return scoring
def official_utility(task,text,references):
    value = parse_prediction(task,text)
    return float(value == int(references[0]))
'''
names = ("parse_prediction", "attention_summaries", "Observer", "make_features", "generate_one")
(vendor / "legacy_short_reference.py").write_text(header + "\n\n".join(ast.unparse(n) for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names) + "\n", encoding="utf-8")
manifest = {"sources": [{"path": p, "sha256": sha(ROOT / p), "bytes": (ROOT / p).stat().st_size} for p in selected],
            "copies": {p.name: sha(p) for p in vendor.glob("*.py")},
            "selection": "Root short pipeline plus paper-export native operator; sharing frozen runtime values only; no old outputs reused"}
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports/source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
for task in ("boolq", "sst2", "logiqa", "litm"):
    path = res + (f"stage26q2_short_base_srar/{task}/selector/frozen_config.json" if task != "litm" else "stage26q2_litm_srar/frozen_config_parent.json")
    frozen = json.loads((ROOT / path).read_text(encoding="utf-8"))
    gamma = frozen.get("gamma", frozen.get("best_gamma"))
    config = dict(experiment=f"srar2_qwen_{task}_v1", task=task, backbone_id="Qwen/Qwen2-7B", model_path=None,
                  dtype="bfloat16", attention_backend="sdpa", operator="K_AFTER_REPEAT_KV_NATIVE_SDPA_SEMANTICS",
                  operator_sha256=sha(operator), initial_anchor=0, gamma0=gamma,
                  gamma_values=[gamma, round(gamma * .8, 8), round(gamma * 1.2, 8)],
                  selected_heads=frozen["selected_heads"], target_layers=list(range(28)), observed_layers=[14, 27],
                  hidden_size=3584, coefficient_embedding_dim=16, seed=101, max_epochs=50, patience=7,
                  learning_rate=.0001, weight_decay=.01, batch_size=42, dropout=.2,
                  loss_weights=dict(regression=1., positive=.5, ranking=.2), strict_gate=True,
                  add_special_tokens=task == "litm", max_new_tokens={"sst2": 1, "boolq": 8, "logiqa": 8, "litm": 100}[task],
                  scorer_version="zerotuning-extractors-ast-v1" if task != "litm" else None,
                  manifests={"train": None, "dev": None},
                  runtime_status="ready_for_asset_audit" if task != "litm" else "blocked_pending_litm_adapter_and_input_parity",
                  reference_config_sha256=sha(ROOT / path))
    (HERE / "configs").mkdir(exist_ok=True)
    # JSON is a strict YAML subset and keeps values/order explicit.
    (HERE / f"configs/qwen_{task}.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print("Captured", len(selected), "legacy source/config hashes; original files unchanged")
