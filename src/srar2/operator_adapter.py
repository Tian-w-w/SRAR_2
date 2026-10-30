"""Tensor-level invariant used by tests; runtime uses the vendored native operator."""
from __future__ import annotations
import math


def expanded_scaled_key(key, query_heads, selected_heads, anchor, gamma):
    if query_heads % key.shape[1] or not 0 <= anchor < key.shape[-2]:
        raise ValueError("invalid GQA geometry/anchor")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("invalid gamma")
    if len(set(selected_heads)) != len(selected_heads) or any(h < 0 or h >= query_heads for h in selected_heads):
        raise ValueError("invalid query heads")
    b, kv, length, dim = key.shape
    groups = query_heads // kv
    expanded = key[:, :, None].expand(b, kv, groups, length, dim).reshape(b, query_heads, length, dim).clone()
    expanded[:, list(selected_heads), anchor, :] *= gamma
    return expanded
