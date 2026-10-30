from __future__ import annotations
import torch
import torch.nn.functional as F


def rank_loss(scores, advantages, mask=None, equivalence=None):
    s, a = scores.flatten(1), advantages.flatten(1)
    valid = torch.ones_like(a, dtype=torch.bool) if mask is None else mask.flatten(1).bool()
    eq = None if equivalence is None else equivalence.flatten(1)
    values = []
    for left in range(s.shape[1]):
        for right in range(left + 1, s.shape[1]):
            v = valid[:, left] & valid[:, right] & (a[:, left] != a[:, right])
            if eq is not None:
                v = v & (eq[:, left] != eq[:, right])
            if v.any():
                sign = (a[v, left] - a[v, right]).sign()
                values.append(F.softplus(-(s[v, left] - s[v, right]) * sign).mean())
    return torch.stack(values).mean() if values else scores.sum() * 0


def joint_loss(scores, advantages, mask=None, equivalence=None):
    if scores.shape != advantages.shape:
        raise ValueError("scores/advantages shape mismatch")
    valid = torch.ones_like(scores, dtype=torch.bool) if mask is None else mask.bool()
    if not torch.isfinite(scores).all() or not torch.isfinite(advantages[valid]).all():
        raise ValueError("non-finite loss input")
    if not valid.any():
        return scores.sum() * 0
    regression = F.smooth_l1_loss(scores[valid], advantages[valid])
    positive = F.binary_cross_entropy_with_logits(scores[valid], (advantages[valid] > 0).float())
    return regression + 0.5 * positive + 0.2 * rank_loss(scores, advantages, valid, equivalence)


def sample_weights(advantages):
    positive = (advantages.flatten(1) > 0).any(1)
    n = int(positive.sum())
    if n == 0 or n == len(positive):
        return torch.ones(len(positive), dtype=torch.double) / len(positive)
    return torch.where(positive, 0.5 / n, 0.5 / (len(positive) - n)).double()
