from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from srar2.actions import ActionSpace
from srar2.config import binding, load_config
from srar2.data import prepare_short
from srar2.io import digest, read_rows, write_rows
from srar2.pipeline import generate_rows
from srar2.training import RecordDataset, train, load_checkpoint, calibrate_checkpoint, analyze

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    """Synthetic fixture, never a real experiment result."""
    def __init__(self, config):
        self.assets = {"files": {}}
        self.identity = {**binding(config), "model_sha256": "synthetic-model", "tokenizer_sha256": "synthetic-tokenizer"}
        self.calls = 0
    def generate(self, x, anchor, gamma, capture=False):
        assert "gold" not in x
        self.calls += 1
        text = "True" if anchor == 2 and gamma > 1 else "False"
        result = {"prediction_raw": text, "generated_token_ids": [1 if text == "True" else 0],
                  "anchor_token_index": anchor, "gamma": gamma}
        feature = (np.zeros((10, 3584), np.float16), np.zeros((3, 16), np.float32)) if capture else None
        return result, feature


def test_synthetic_end_to_end_training_calibration_and_resume(tmp_path):
    c = load_config(ROOT / "configs/qwen_boolq.yaml")
    c["max_epochs"], c["patience"], c["batch_size"] = 2, 1, 2
    backend = FakeBackend(c)
    manifests = {}
    for split in ("train", "dev"):
        rows = [{"row_uid": split + str(i), "source_question_id": split + "q" + str(i), "split": split,
                 "raw_prompt": split + "prompt" + str(i), "prompt_tokens": 5, "input_ids_sha256": "synthetic-ids",
                 "controls": {"RE": 1, "RM": 2, "RL": 3}, "query_token_span": [3, 5], "gold": i % 2} for i in range(4)]
        path = tmp_path / (split + ".jsonl"); write_rows(path, rows); manifests[split] = path
        stats = generate_rows(c, rows, path, tmp_path, backend)
        assert stats["generations"] == 40 and stats["failures"] == 0
        previous = backend.calls
        stats = generate_rows(c, rows, path, tmp_path, backend, resume=True)
        assert stats["generations"] == 0 and stats["cache_hits"] == 40 and previous == backend.calls
    train_data = RecordDataset(c, tmp_path, "train", manifests["train"])
    dev_data = RecordDataset(c, tmp_path, "dev", manifests["dev"])
    cp = tmp_path / "joint.pt"
    outcome = train(c, train_data, dev_data, cp)
    assert outcome["epochs"] <= 2 and cp.is_file()
    model, saved = load_checkpoint(cp, c)
    assert saved["initialized_from_scratch"] and saved["parameter_count"] == 1635169
    calibrated = calibrate_checkpoint(c, dev_data, cp, tmp_path / "cal.json")
    assert np.isfinite(calibrated["tau"])
    report = analyze(c, dev_data, cp)
    assert "J-BASE-mask" in report["policies"] and report["oracle"]["additional_oracle_rows"] == 2
    for j in range(3):
        train(c, train_data, dev_data, tmp_path / f"fixed{j}.pt", kind="fixed", coefficient=j)
    comparison = analyze(c, dev_data, cp, tmp_path / "fixed0.pt", [tmp_path / f"fixed{j}.pt" for j in range(3)])
    assert "B2-SRAR-ref" in comparison["policies"] and "B3-selected" in comparison["policies"] and "B5-Sequential" in comparison["policies"]
    missing = read_rows(manifests["dev"])[:-1]
    write_rows(manifests["dev"], missing)
    with pytest.raises(ValueError, match="coverage"):
        RecordDataset(c, tmp_path, "dev", manifests["dev"])


def test_import_original_prepared_manifest_without_resplitting(tmp_path):
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}
    # Exact original field names, no source_question_id; question recovered by span.
    old = {"sample_id": "boolq_validation_0007", "split": "validation", "raw_prompt": "abcdefghi?",
           "prompt_tokens": 10, "context_token_span": [0, 6], "query_token_span": [6, 9],
           "controls": {"RE": 1, "RM": 3, "RL": 5}, "gold": 1, "add_special_tokens": False}
    source, output = tmp_path / "original.jsonl", tmp_path / "imported.jsonl"
    write_rows(source, [old])
    prepare_short(source, output, "boolq", "dev", Tokenizer())
    row = read_rows(output)[0]
    assert row["row_uid"] == old["sample_id"] and row["historical_split"] == "validation"
    assert row["source_question_id"] == digest("ghi") and row["controls"] == old["controls"]
    assert read_rows(source) == [old]
