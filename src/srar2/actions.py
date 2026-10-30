from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .io import digest

ANCHORS = ("RE", "RM", "RL")
COEFFICIENTS = ("BASE", "LOW", "HIGH")


@dataclass(frozen=True)
class ActionSpace:
    gamma_values: tuple
    schema_version: str = "srar2-v1"

    def __post_init__(self):
        object.__setattr__(self, "gamma_values", tuple(float(x) for x in self.gamma_values))
        if len(self.gamma_values) not in (1, 3):
            raise ValueError("v1 supports M=1 or M=3")
        if not all(math.isfinite(x) and x > 0 for x in self.gamma_values):
            raise ValueError("gamma must be finite and positive")
        if len(set(self.gamma_values)) != len(self.gamma_values):
            raise ValueError("duplicate coefficient values")

    @property
    def m(self):
        return len(self.gamma_values)

    @property
    def order(self):
        # All BASE actions first; then each region's LOW/HIGH.
        return tuple([f"{a}:BASE" for a in ANCHORS] +
                     [f"{a}:{c}" for a in ANCHORS for c in COEFFICIENTS[1:self.m]])

    def indices(self, action_id):
        a, c = action_id.split(":")
        i, j = ANCHORS.index(a), COEFFICIENTS.index(c)
        if j >= self.m:
            raise ValueError("coefficient not in grid")
        return i, j

    def action_id(self, i, j):
        if not (0 <= i < 3 and 0 <= j < self.m):
            raise ValueError("action indices out of bounds")
        return f"{ANCHORS[i]}:{COEFFICIENTS[j]}"

    def flatten(self, scores):
        if tuple(scores.shape[-2:]) != (3, self.m):
            raise ValueError("expected [...,3,M]")
        return np.stack([scores[..., self.indices(a)[0], self.indices(a)[1]] for a in self.order], -1)

    def unflatten(self, flat):
        flat = np.asarray(flat)
        if flat.shape[-1] != 3 * self.m:
            raise ValueError("wrong action count")
        out = np.empty((*flat.shape[:-1], 3, self.m), dtype=flat.dtype)
        for n, a in enumerate(self.order):
            i, j = self.indices(a)
            out[..., i, j] = flat[..., n]
        return out

    def best(self, scores, allowed=None):
        flat = self.flatten(np.asarray(scores))
        if not np.isfinite(flat).all():
            raise ValueError("non-finite prediction")
        allowed = set(self.order if allowed is None else allowed)
        if not allowed or not allowed.issubset(self.order):
            raise ValueError("invalid allowed actions")
        flat = np.where(np.array([a in allowed for a in self.order]), flat, -np.inf)
        choice = flat.argmax(-1)
        return choice, np.take_along_axis(flat, np.expand_dims(choice, -1), -1).squeeze(-1)

    def equivalent_ids(self, anchors):
        keys, result = {}, np.empty((3, self.m), dtype=np.int64)
        for i, anchor in enumerate(ANCHORS):
            for j, gamma in enumerate(self.gamma_values):
                key = ("identity",) if gamma == 1.0 else (int(anchors[anchor]), gamma)
                if key not in keys:
                    keys[key] = len(keys)
                result[i, j] = keys[key]
        return result

    @property
    def schema(self):
        return {"schema_version": self.schema_version, "gamma_values": list(self.gamma_values),
                "anchors": list(ANCHORS), "coefficient_ids": list(COEFFICIENTS[:self.m]),
                "action_order": list(self.order), "tensor_shape": [3, self.m]}

    @property
    def sha256(self):
        return digest(self.schema)
