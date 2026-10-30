from __future__ import annotations
import math
import time
import numpy as np
import torch
from .features import validate_features


def infer_srar2(x, backend, selector, frozen, space, mean, std):
    """No reference, gold, candidate text, or utility accepted by this API."""
    if not math.isfinite(float(frozen["tau"])):
        raise ValueError("tau must be finite")
    started = time.perf_counter()
    y0, features = backend.generate(x, anchor=0, gamma=frozen["gamma0"], capture=True)
    initial_done = time.perf_counter()
    vectors, scalars = features
    validate_features(vectors, scalars)
    selector.eval()
    device = next(selector.parameters()).device
    with torch.inference_mode():
        scores = selector(torch.as_tensor(vectors.astype(np.float32), device=device)[None],
                          (torch.as_tensor(scalars, device=device)[None] - mean.to(device)) / std.to(device)).cpu().numpy()[0]
    choice, best = space.best(scores)
    action_id = space.order[int(choice)]
    i, j = space.indices(action_id)
    triggered = bool(best > frozen["tau"])
    selector_done = time.perf_counter()
    if triggered:
        final, _ = backend.generate(x, anchor=int(x["controls"][("RE", "RM", "RL")[i]]),
                                    gamma=space.gamma_values[j], capture=False)
    else:
        final = y0
    ended = time.perf_counter()
    return final, {"initial": y0, "scores": scores.tolist(), "action_order": list(space.order),
                   "selected_action": action_id, "selected_gamma": space.gamma_values[j],
                   "tau": frozen["tau"], "triggered": triggered, "passes": 1 + int(triggered),
                   "timing_seconds": {"initial_with_features": initial_done - started,
                                      "selector_and_decision": selector_done - initial_done,
                                      "reroute_generation": ended - selector_done, "total": ended - started}}
