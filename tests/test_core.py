import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from srar2.actions import ActionSpace
from srar2.calibration import calibrate, policy
from srar2.config import binding, check_binding, load_config
from srar2.features import attention_summaries
from srar2.inference import infer_srar2
from srar2.io import digest
from srar2.losses import joint_loss, rank_loss, sample_weights
from srar2.metrics import oracle, primary, transitions
from srar2.model import Router
from srar2.operator_adapter import expanded_scaled_key
from srar2.records import audit_manifests, cache_key, validate_record

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
SPACE = ActionSpace((.95, .76, 1.14))


def test_router_geometry_parameters_gradients():
    torch.manual_seed(101)
    router = Router()
    v, s = torch.randn(2, 10, 3584), torch.randn(2, 3, 16)
    output = router(v, s)
    assert output.shape == (2, 3, 3)
    assert sum(p.numel() for p in router.parameters()) == 1635169
    legacy = Router(m=1, joint=False)
    assert sum(p.numel() for p in legacy.parameters()) == 1631025
    joint_loss(output, torch.randn_like(output)).backward()
    for projection in router.projections:
        assert torch.isfinite(projection.weight.grad).all() and projection.weight.grad.abs().sum() > 0
    assert (router.coefficient_embedding.weight.grad.abs().sum(1) > 0).all()


@pytest.mark.parametrize("m", [1, 3])
def test_legacy_compatibility_and_tie(m):
    old = Router(m=1, joint=False).eval()
    new = Router(m=m).eval()
    new.load_legacy(old.state_dict())
    v, s = torch.randn(3, 10, 3584), torch.randn(3, 3, 16)
    with torch.no_grad():
        assert torch.allclose(new(v, s), old(v, s).expand(-1, -1, m), atol=2e-7, rtol=2e-6)
    sp = ActionSpace((.95,) if m == 1 else (.95, .76, 1.14))
    choice, value = sp.best(np.zeros((3, m)))
    assert sp.order[int(choice)] == "RE:BASE"
    assert value == 0


def test_action_roundtrip_and_full_ties():
    x = np.arange(18).reshape(2, 3, 3)
    assert np.array_equal(SPACE.unflatten(SPACE.flatten(x)), x)
    assert SPACE.order == ("RE:BASE", "RM:BASE", "RL:BASE", "RE:LOW", "RE:HIGH", "RM:LOW", "RM:HIGH", "RL:LOW", "RL:HIGH")
    for n, aid in enumerate(SPACE.order):
        i, j = SPACE.indices(aid)
        assert SPACE.action_id(i, j) == aid
    x = np.zeros((3, 3)); x[2, 0] = 1; x[0, 1] = 1
    assert SPACE.order[int(SPACE.best(x)[0])] == "RL:BASE"


@pytest.mark.parametrize("grid", [(0, .8, 1.2), (.95, .95, 1.14), (float("nan"),), (.9, .8)])
def test_reject_bad_grids(grid):
    with pytest.raises(ValueError):
        ActionSpace(grid)


@pytest.mark.parametrize("change", ["gamma_values", "gamma0", "selected_heads", "operator_sha256", "action_order"])
def test_protocol_checkpoint_binding(change):
    c = load_config(ROOT / "configs/qwen_boolq.yaml")
    saved = binding(c)
    changed = copy.deepcopy(c)
    if change == "gamma_values":
        changed[change] = [.95, 1.14, .76]
    elif change == "gamma0":
        changed[change] = .9
    elif change == "selected_heads":
        changed[change] = list(reversed(changed[change]))
    else:
        changed[change] = "changed"
    with pytest.raises(ValueError):
        check_binding(saved, changed)
    assert ActionSpace((.95, .76, 1.14), "new-schema").sha256 != SPACE.sha256


def test_rank_direction_ties_mask_equivalence_and_empty_gradient():
    a = torch.tensor([[[0., 1., -1.]]])
    correct, wrong = a.clone(), -a.clone()
    assert rank_loss(correct, a) < rank_loss(wrong, a)
    s = torch.randn(2, 3, 3, requires_grad=True)
    loss = rank_loss(s, torch.zeros_like(s))
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(s.grad).all()
    eq = torch.zeros_like(a, dtype=torch.long)
    assert rank_loss(wrong, a, equivalence=eq).item() == 0
    mask = torch.zeros_like(s, dtype=torch.bool)
    assert joint_loss(s, torch.full_like(s, float("nan")), mask).item() == 0
    a = torch.tensor([[[1., float("nan")]]])
    assert torch.isfinite(joint_loss(torch.zeros_like(a), a, torch.tensor([[[True, False]]])))


def test_sample_weight_single_group_and_balanced():
    assert torch.allclose(sample_weights(torch.zeros(3, 3, 3)), torch.ones(3, dtype=torch.double) / 3)
    a = torch.zeros(4, 3, 3); a[0, 1, 2] = 1
    w = sample_weights(a)
    assert w[0] == .5 and torch.allclose(w[1:].sum(), torch.tensor(.5, dtype=torch.double))


def test_strict_calibration_equal_scores_and_endpoints():
    s = np.zeros((3, 3, 3)); s[:, 0, 0] = [1., 1., 2.]
    initial = np.array([0., 1., 0.]); u = np.zeros_like(s); u[:, 0, 0] = [1., 0., 1.]
    cal = calibrate(s, initial, u, SPACE, "dev")
    assert cal["selected"]["tau"] == 1.
    assert cal["selected"]["trigger_count"] == 1
    assert cal["frontier"][0]["trigger_count"] == 3
    assert cal["frontier"][-1]["trigger_count"] == 0
    unchanged = calibrate(np.zeros_like(s), np.zeros(3), np.zeros_like(s), SPACE, "validation")
    assert unchanged["selected"]["trigger_count"] == 0
    _, t, _, _ = policy(s, initial, u, SPACE, 2.)
    assert not t.any()
    float32 = calibrate(np.zeros((2, 3, 3), dtype=np.float32), np.zeros(2), np.ones((2, 3, 3)), SPACE, "dev")
    assert float32["frontier"][0]["trigger_count"] == 2


@pytest.mark.parametrize("split", ["train", "formal", "test"])
def test_calibration_rejects_nonvalidation(split):
    with pytest.raises(ValueError):
        calibrate(np.zeros((1, 3, 3)), np.zeros(1), np.zeros((1, 3, 3)), SPACE, split)


def test_macro_not_micro_and_rescue_harm_closure():
    assert primary([1, 1, 0], ["a", "a", "b"]) == .5
    assert primary([1, 1, 0]) == 2 / 3
    with pytest.raises(ValueError):
        primary([1, 0], ["a", "b"], ["a", "b", "c"])
    t = transitions([0, 1, 1, 0], [1, 0, 1, 0], [True, True, False, False])
    assert t["rescue"] == t["harm"] == 1 and t["accuracy_delta"] == 0
    assert transitions([0], [0], [False])["rescue_triggered"] is None
    with pytest.raises(ValueError):
        transitions([0], [1], [False])


def test_litm_group_rows_not_overwritten_and_leakage():
    rows = [{"row_uid": str(i), "source_question_id": "q1", "raw_prompt": "p" + str(i), "split": "train"} for i in range(3)]
    audit = audit_manifests({"train": rows})
    assert audit["row_count"] == 3 and audit["groups"] == 1
    with pytest.raises(ValueError, match="leakage"):
        audit_manifests({"train": rows, "dev": [{"row_uid": "other", "source_question_id": "q1", "raw_prompt": "other", "split": "dev"}]})
    with pytest.raises(ValueError):
        audit_manifests({"train": rows + [rows[0]]})


@pytest.mark.parametrize("field", ["gamma", "operator_sha256", "input_ids_sha256", "scorer_version", "model_sha256", "generation"])
def test_cache_invalidates_each_protocol_component(field):
    from srar2.records import CACHE_FIELDS
    metadata = {k: "original" for k in CACHE_FIELDS}
    other = {**metadata, field: "changed"}
    assert cache_key(metadata) != cache_key(other)
    del metadata["gamma"]
    with pytest.raises(ValueError):
        cache_key(metadata)


def test_record_matrix_advantage_equivalence_missing():
    space = ActionSpace((1., .8, 1.2))
    anchors = {"RE": 1, "RM": 1, "RL": 2}
    r = {"action_schema_sha256": space.sha256, "initial": {"gamma": 1., "anchor_token_index": 0, "utility": 0.},
         "anchor_token_indices": anchors, "actions": []}
    for aid in space.order:
        i, j = space.indices(aid)
        r["actions"].append({"action_id": aid, "anchor_id": i, "coefficient_id": j, "gamma": space.gamma_values[j],
            "anchor_token_index": anchors[("RE", "RM", "RL")[i]], "generation_status": "complete",
            "utility": 1., "advantage": 1., "prediction_raw": "same"})
    a, u, mask = validate_record(r, space)
    assert a.shape == (3, 3) and mask.all()
    broken = copy.deepcopy(r); broken["actions"][0]["advantage"] = -1
    with pytest.raises(ValueError): validate_record(broken, space)
    broken = copy.deepcopy(r); broken["actions"][1]["prediction_raw"] = "conflict"
    with pytest.raises(ValueError, match="conflicting"): validate_record(broken, space)
    r["actions"][0]["generation_status"] = "pending"
    with pytest.raises(ValueError): validate_record(r, space)
    assert validate_record(r, space, True)[2].sum() == 8


def test_native_operator_tensor_branch_and_state_reset():
    # Execute the frozen production function bodies with a repeat_kv shim;
    # actual SDPA runs on CPU, without needing Transformers installed.
    path = ROOT / "src/srar2/vendor/native_operator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    class Attention(torch.nn.Module):
        def __init__(self, i):
            super().__init__(); self.layer_idx = i; self.num_key_value_groups = 2; self.is_causal = True
    def repeat(k, groups):
        b, h, s, d = k.shape
        return k[:, :, None].expand(b, h, groups, s, d).reshape(b, h * groups, s, d)
    ns = {"torch": torch, "F": F, "hashlib": __import__("hashlib"), "Iterable": __import__("typing").Iterable,
          "Optional": __import__("typing").Optional, "BACKEND": "test", "ALL_ATTENTION_FUNCTIONS": {},
          "modeling_qwen2": SimpleNamespace(repeat_kv=repeat, Qwen2Attention=Attention)}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), "exec"), ns)
    model = torch.nn.Sequential(Attention(0), Attention(1)); model.config = SimpleNamespace()
    q, k, v = torch.randn(1, 4, 5, 8), torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)
    before = k.clone()
    ns["configure"](model, 2, [1], .76, [0], capture_layer=0)
    out, _ = ns["attention"](model[0], q, k, v, None)
    expected = F.scaled_dot_product_attention(q, expanded_scaled_key(k, 4, [1], 2, .76), repeat(v, 2), is_causal=True)
    assert torch.equal(out, expected.transpose(1, 2).contiguous())
    trace = model[0]._d1r_capture_result
    assert trace["changed_heads"] == [1] and trace["changed_tokens"] == [2] and trace["cache_unchanged_by_scale"]
    assert torch.equal(k, before) and not model[1]._d1r_enabled
    ns["configure"](model, 0, [3], 1., [1])
    assert not model[0]._d1r_enabled and model[1]._d1r_control == 0 and model[1]._d1r_gamma == 1.
    identity, _ = ns["attention"](model[1], q, k, v, None)
    native = F.scaled_dot_product_attention(q, repeat(k, 2), repeat(v, 2), is_causal=True).transpose(1, 2).contiguous()
    assert torch.equal(identity, native)
    ns["disable"](model)
    assert not any(m._d1r_enabled for m in model)


def test_observer_actual_gamma_and_causal_span():
    q, k = torch.randn(1, 4, 5, 8), torch.randn(1, 2, 5, 8)
    first = attention_summaries(q, k, 0, [1, 3], .95, [2, 5], [1, 2, 4])
    other = attention_summaries(q, k, 0, [1, 3], .9, [2, 5], [1, 2, 4])
    assert first.shape == (3, 5) and torch.isfinite(first).all() and not torch.equal(first, other)


def test_online_generation_budget_no_gold_and_no_grid_features():
    signature = inspect.signature(infer_srar2)
    assert not any(x in signature.parameters for x in ("gold", "reference", "utility"))
    class Backend:
        def __init__(self): self.calls = []
        def generate(self, x, anchor, gamma, capture=False):
            assert not any(k in x for k in ("gold", "reference", "utility"))
            self.calls.append((anchor, gamma, capture))
            return {"prediction_raw": str(anchor)}, (np.zeros((10, 3584), np.float32), np.zeros((3, 16), np.float32)) if capture else None
    router = Router().eval()
    with torch.no_grad(): router.scorer[-1].weight.zero_(); router.scorer[-1].bias.zero_()
    b = Backend(); x = {"controls": {"RE": 2, "RM": 3, "RL": 4}}
    _, r = infer_srar2(x, b, router, {"gamma0": .95, "tau": 0.}, SPACE, torch.zeros(1, 1, 16), torch.ones(1, 1, 16))
    assert r["passes"] == 1 and b.calls == [(0, .95, True)]
    _, r = infer_srar2(x, b, router, {"gamma0": .95, "tau": -.1}, SPACE, torch.zeros(1, 1, 16), torch.ones(1, 1, 16))
    assert r["passes"] == 2 and b.calls[-2:] == [(0, .95, True), (2, .95, False)]


def test_oracle_containment():
    initial = [0, 1]
    u = np.zeros((2, 3, 3)); u[0, 2, 1] = 1
    report = oracle(initial, u)
    assert report["joint_oracle"] == 1 and report["base_oracle"] == .5


def test_sequential_uses_only_first_stage_anchor():
    s = np.zeros((1, 3, 3)); s[0, 2, 2] = 10.; s[0, 0, 1] = 2.
    ref = np.zeros_like(s); ref[0, 0, 0] = 1.
    _, _, choice, _ = policy(s, [0.], np.zeros_like(s), SPACE, 0., sequential_scores=ref)
    assert SPACE.order[int(choice[0])] == "RE:LOW"
