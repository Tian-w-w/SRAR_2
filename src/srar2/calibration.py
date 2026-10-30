from __future__ import annotations
import numpy as np
from .metrics import primary


def policy(scores, initial, utilities, space, tau, allowed=None, sequential_scores=None):
    scores = np.asarray(scores)
    if sequential_scores is not None:
        first, _ = space.best(sequential_scores, [a + ":BASE" for a in ("RE", "RM", "RL")])
        anchor = np.asarray([space.indices(space.order[n])[0] for n in first])
        scores = scores.copy()
        # Keep every coefficient at the first-stage anchor; no gold used.
        flat = space.flatten(scores)
        valid = np.array([[space.indices(a)[0] == i for a in space.order] for i in anchor])
        # best() rejects infinity, so use finite minimum, then argmax directly.
        flat = np.where(valid, flat, np.finfo(np.float64).min)
        choice = flat.argmax(1)
        best = flat[np.arange(len(flat)), choice]
    else:
        choice, best = space.best(scores, allowed)
    selected = space.flatten(np.asarray(utilities))[np.arange(len(choice)), choice]
    # Compare in float64 so the nextafter all-trigger endpoint cannot round
    # back onto a float32 maximum score under NumPy scalar promotion rules.
    triggered = np.asarray(best, dtype=np.float64) > float(tau)
    final = np.where(triggered, selected, initial)
    return final, triggered, choice, best


def calibrate(scores, initial, utilities, space, split, allowed=None, groups=None, expected_groups=None,
              sequential_scores=None):
    if split not in ("dev", "validation"):
        raise ValueError("calibration requires validation, never Formal/train")
    _, _, _, best = policy(scores, initial, utilities, space, 0.0, allowed, sequential_scores)
    low = float(np.nextafter(np.float64(best.min()), -np.inf))
    if not np.isfinite(low):
        raise ValueError("cannot represent all-trigger endpoint")
    frontier = []
    for tau in [low, *[float(x) for x in np.unique(best)]]:
        final, t, _, _ = policy(scores, initial, utilities, space, tau, allowed, sequential_scores)
        frontier.append({"tau": tau, "metric": primary(final, groups, expected_groups),
                         "trigger_count": int(t.sum()), "trigger_rate": float(t.mean())})
    selected = max(frontier, key=lambda r: (r["metric"], -r["trigger_count"], r["tau"]))
    return {"split": split, "selected": selected, "frontier": frontier,
            "tie_rule": "metric desc, trigger_count asc, tau desc", "strict_gate": True}
