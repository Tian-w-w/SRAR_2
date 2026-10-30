from __future__ import annotations
import numpy as np


def primary(values, groups=None, expected_groups=None):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("empty/non-finite metric input")
    if groups is None:
        return float(values.mean())
    groups = np.asarray(groups)
    if len(groups) != len(values):
        raise ValueError("group length mismatch")
    present = set(groups.tolist())
    if expected_groups is not None and present != set(expected_groups):
        raise ValueError("missing or unexpected macro groups")
    return float(np.mean([values[groups == g].mean() for g in sorted(present)]))


def transitions(initial, final, triggered):
    initial, final = np.asarray(initial), np.asarray(final)
    t = np.asarray(triggered, dtype=bool)
    if not (initial.shape == final.shape == t.shape) or initial.ndim != 1 or not len(t):
        raise ValueError("unpaired records")
    if not (np.isin(initial, [0, 1]).all() and np.isin(final, [0, 1]).all()):
        raise ValueError("binary transitions require binary utilities")
    if np.any(initial[~t] != final[~t]):
        raise ValueError("retained outputs changed")
    rescue = int(((initial == 0) & (final == 1)).sum())
    harm = int(((initial == 1) & (final == 0)).sum())
    count = int(t.sum())
    delta = float((final - initial).mean())
    if not np.isclose(delta, (rescue - harm) / len(t)):
        raise ValueError("rescue/harm closure failed")
    return {"n": len(t), "trigger_count": count, "rescue": rescue, "harm": harm,
            "both_correct": int(((initial == 1) & (final == 1)).sum()),
            "both_wrong": int(((initial == 0) & (final == 0)).sum()),
            "rescue_all": rescue / len(t), "harm_all": harm / len(t),
            "rescue_triggered": rescue / count if count else None,
            "harm_triggered": harm / count if count else None, "accuracy_delta": delta}


def oracle(initial, utilities):
    initial, utilities = np.asarray(initial), np.asarray(utilities)
    base = np.maximum(initial, utilities[:, :, 0].max(1))
    expanded = np.maximum(initial, utilities.max((1, 2)))
    if np.any(expanded < base):
        raise ValueError("oracle containment failure")
    return {"n": len(initial), "base_oracle": float(base.mean()), "joint_oracle": float(expanded.mean()),
            "additional_oracle_rows": int((expanded > base).sum())}
