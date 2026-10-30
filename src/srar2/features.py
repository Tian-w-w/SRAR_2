from __future__ import annotations
import math
import numpy as np
import torch
from .operator_adapter import expanded_scaled_key
from .io import digest

LAYERS = (14, 27)
HIDDEN = 3584
ALT = ("RE", "RM", "RL")
FEATURE_SCHEMA = {
    "version": "qwen2-short-initial-capture-v1",
    "vectors": ["q14", "q27", "initial14", "initial27", "RE14", "RM14", "RL14", "RE27", "RM27", "RL27"],
    "scalar_shape": [3, 16], "observed_layers": [14, 27], "hidden_size": 3584,
    "statistics": ["query_mean", "query_std_unbiased", "query_max", "query_last_mean", "all_mean"],
    "attention": "causal prefill all query heads; initial K scaled with actual gamma0; chunks=32",
    "uncertainty": ["top1", "top2", "margin", "entropy"],
    "length_features": ["length/4096", "anchor/max(1,length-1)"],
}
FEATURE_SCHEMA_SHA256 = digest(FEATURE_SCHEMA)


def attention_summaries(query, key, control, heads, gamma, qspan, candidates):
    expanded = expanded_scaled_key(key, query.shape[1], heads, control, gamma)
    q, k = query[0].float(), expanded[0].float()
    head_count, seq_len, dim = q.shape
    q0, q1 = map(int, qspan)
    if not 0 <= q0 < q1 <= seq_len or head_count * (q1 - q0) < 2:
        raise ValueError("invalid query span for unbiased std")
    result = []
    for candidate in candidates:
        blocks = []
        for start in range(0, seq_len, 32):
            end = min(seq_len, start + 32)
            scores = torch.matmul(q[:, start:end], k.transpose(-2, -1)) / math.sqrt(dim)
            rr = torch.arange(start, end, device=q.device)[:, None]
            cc = torch.arange(seq_len, device=q.device)[None, :]
            scores = scores.masked_fill(cc > rr, float("-inf"))
            probability = torch.exp(scores[:, :, candidate] - torch.logsumexp(scores, dim=-1))
            probability[:, torch.arange(start, end, device=q.device) < candidate] = 0
            blocks.append(probability.cpu())
        probabilities = torch.cat(blocks, dim=1)
        qvalues = probabilities[:, q0:q1]
        result.append(torch.stack((qvalues.mean(), qvalues.std(), qvalues.max(), probabilities[:, q1 - 1].mean(), probabilities.mean())))
    return torch.stack(result)


class Observer:
    def __init__(self, row, frozen):
        self.row, self.frozen, self.data = row, frozen, {}

    def wrap(self, original):
        def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
            layer = int(module.layer_idx)
            if layer in LAYERS and layer not in self.data and query.shape[-2] > 1:
                self.data[layer] = attention_summaries(query, key, 0, self.frozen["selected_heads"],
                    self.frozen["gamma0"], self.row["query_token_span"], [int(self.row["controls"][n]) for n in ALT])
            return original(module, query, key, value, attention_mask, *args, **kwargs)
        return wrapped


def make_features(row, generated, observer):
    if set(observer.data) != set(LAYERS):
        raise RuntimeError("incomplete observer capture")
    hidden = generated.hidden_states[0]
    mid, last = hidden[15][0], hidden[28][0]
    q0, q1 = row["query_token_span"]
    candidates = [row["controls"][n] for n in ALT]
    vectors = torch.stack([mid[q0:q1].float().mean(0), last[q0:q1].float().mean(0), mid[0].float(), last[0].float(),
                          *[mid[x].float() for x in candidates], *[last[x].float() for x in candidates]]).cpu().numpy().astype(np.float16)
    probabilities = torch.softmax(generated.scores[0][0].float(), -1)
    top = torch.topk(probabilities, 2).values
    uncertainty = [float(top[0]), float(top[1]), float(top[0] - top[1]),
                   float(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum())]
    attention = torch.cat([observer.data[14], observer.data[27]], 1).numpy()
    scalars = np.asarray([[*attention[i], *uncertainty, row["prompt_tokens"] / 4096.,
                           candidates[i] / max(1, row["prompt_tokens"] - 1)] for i in range(3)], np.float32)
    validate_features(vectors, scalars)
    return vectors, scalars


def validate_features(vectors, scalars):
    if vectors.shape != (10, HIDDEN) or scalars.shape != (3, 16):
        raise ValueError("feature shape mismatch")
    if not (np.isfinite(vectors).all() and np.isfinite(scalars).all()):
        raise ValueError("non-finite features")
