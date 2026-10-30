from __future__ import annotations
import numpy as np
import torch
from .actions import ActionSpace
from .inference import infer_srar2
from .io import digest
from .model import Router
from .pipeline import input_only


def real_smoke(c, rows, backend):
    from .vendor import legacy_short_reference as legacy
    space = ActionSpace(c["gamma_values"])
    report = {"status": "running", "identity": backend.identity, "rows": [], "formal_read": False}
    frozen = {"selected_heads": c["selected_heads"], "gamma": c["gamma0"], "target_layers": c["target_layers"], "operator": c["operator"]}
    selector = Router(m=space.m).eval()
    # Forced scores through a zero final layer, while exercising the real selector API.
    with torch.no_grad():
        selector.scorer[-1].weight.zero_(); selector.scorer[-1].bias.zero_()
    for row in rows:
        if row["split"] not in ("train", "dev", "validation"):
            raise ValueError("smoke must not use Formal")
        x = input_only(row)
        y0, feature = backend.generate(x, 0, c["gamma0"], True)
        # AST-identical old methods, with imports redirected to our frozen operator copy.
        old_row = {**row, "sample_id": row["row_uid"], "gold_answers": [row["gold"]]}
        old0, old_feature = legacy.generate_one(backend.model, backend.tokenizer, c["task"], old_row, "initial", frozen, True)
        if y0["generated_token_ids"] != old0["generated_token_ids"]:
            raise AssertionError("initial legacy token parity failed")
        for new, old in zip(feature, old_feature):
            if not np.array_equal(new, old):
                raise AssertionError("initial legacy feature parity failed")
        base_outputs = {}
        for anchor in ("RE", "RM", "RL"):
            new, _ = backend.generate(x, row["controls"][anchor], c["gamma0"])
            old, _ = legacy.generate_one(backend.model, backend.tokenizer, c["task"], old_row, anchor, frozen)
            if new["generated_token_ids"] != old["generated_token_ids"]:
                raise AssertionError("BASE legacy candidate token parity failed")
            base_outputs[anchor] = new["generated_token_ids"]
        # An intervening different action must not affect the next initial capture.
        backend.generate(x, row["controls"]["RL"], space.gamma_values[-1])
        original_grid = backend.config["gamma_values"]
        try:
            backend.config["gamma_values"] = [c["gamma0"]]
            again, again_features = backend.generate(x, 0, c["gamma0"], True)
        finally:
            backend.config["gamma_values"] = original_grid
        if again["generated_token_ids"] != y0["generated_token_ids"] or any(not np.array_equal(a, b) for a, b in zip(feature, again_features)):
            raise AssertionError("gamma/observer state leak")
        before = backend.calls
        _, retain = infer_srar2(x, backend, selector, {"gamma0": c["gamma0"], "tau": 1.}, space,
                               torch.zeros(1, 1, 16), torch.ones(1, 1, 16))
        if backend.calls - before != 1 or retain["passes"] != 1:
            raise AssertionError("forced retain must use one generation")
        before = backend.calls
        final, reroute = infer_srar2(x, backend, selector, {"gamma0": c["gamma0"], "tau": -1.}, space,
                                    torch.zeros(1, 1, 16), torch.ones(1, 1, 16))
        if backend.calls - before != 2 or final["generated_token_ids"] != base_outputs["RE"]:
            raise AssertionError("online/offline selected-action parity failed")
        # Exercise actual GQA/SDPA branch with tensor capture on one selected layer.
        op = backend.op
        op.configure(backend.model, row["controls"]["RE"], c["selected_heads"], c["gamma_values"][-1], c["target_layers"], capture_layer=14)
        enc = backend.tokenizer(row["raw_prompt"], return_tensors="pt", add_special_tokens=c["add_special_tokens"])
        try:
            with torch.inference_mode():
                backend.model(**{k: v.to(backend.device) for k, v in enc.items()}, use_cache=False)
            trace = backend.model.model.layers[14].self_attn._d1r_capture_result
            if not trace["cache_unchanged_by_scale"] or set(trace["changed_heads"]) != set(c["selected_heads"]) or trace["changed_tokens"] != [row["controls"]["RE"]]:
                raise AssertionError("expanded-K selectivity/cache invariant failed")
        finally:
            op.disable(backend.model)
        report["rows"].append({"row_uid": row["row_uid"], "input_ids_sha256": row["input_ids_sha256"],
            "legacy_initial_and_base_parity": True, "feature_parity": True, "no_state_leak": True,
            "forced_retain_passes": retain["passes"], "forced_reroute_passes": reroute["passes"],
            "online_offline_parity": True, "operator_trace": trace,
            "initial_feature_hash": digest([f.tolist() for f in feature])})
    report["status"] = "passed"
    return report
