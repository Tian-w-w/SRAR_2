from __future__ import annotations
import torch
from torch import nn


class Router(nn.Module):
    def __init__(self, m=3, hidden_size=3584, dropout=0.2, joint=True):
        super().__init__()
        self.m, self.hidden_size, self.joint = m, hidden_size, joint
        self.projections = nn.ModuleList([nn.Linear(hidden_size, 64) for _ in range(6)])
        self.candidate_embedding = nn.Embedding(3, 16)
        if joint:
            self.coefficient_embedding = nn.Embedding(m, 16)
        self.scorer = nn.Sequential(nn.Linear(944 if joint else 928, 256), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, vectors, scalars):
        if vectors.ndim != 3 or tuple(vectors.shape[1:]) != (10, self.hidden_size):
            raise ValueError("vectors must have shape [B,10,H]")
        if tuple(scalars.shape) != (vectors.shape[0], 3, 16):
            raise ValueError("scalars must have shape [B,3,16]")
        q = torch.cat([self.projections[0](vectors[:, 0]), self.projections[1](vectors[:, 1])], -1)
        b = torch.cat([self.projections[2](vectors[:, 2]), self.projections[3](vectors[:, 3])], -1)
        ci = torch.cat([self.projections[4](vectors[:, 4:7]), self.projections[5](vectors[:, 7:10])], -1)
        q, b = q[:, None].expand(-1, 3, -1), b[:, None].expand(-1, 3, -1)
        ids = torch.arange(3, device=vectors.device)[None].expand(vectors.shape[0], -1)
        z = torch.cat([q, ci, b, q * ci, q * b, ci - b, q - ci, scalars, self.candidate_embedding(ids)], -1)
        if not self.joint:
            return self.scorer(z).squeeze(-1).unsqueeze(-1)
        z = z[:, :, None].expand(-1, -1, self.m, -1)
        e = self.coefficient_embedding(torch.arange(self.m, device=z.device))[None, None].expand(z.shape[0], 3, -1, -1)
        return self.scorer(torch.cat([z, e], -1)).squeeze(-1)

    def load_legacy(self, state):
        """Smoke compatibility only. New coefficient input columns start at zero."""
        adapted = {k: v.clone() for k, v in state.items()}
        weight = adapted["scorer.0.weight"]
        if weight.shape[1] != 928:
            raise ValueError("not an original 928-column Router")
        if self.joint:
            adapted["scorer.0.weight"] = torch.cat([weight, weight.new_zeros(weight.shape[0], 16)], -1)
            adapted["coefficient_embedding.weight"] = self.coefficient_embedding.weight.detach().clone()
        self.load_state_dict(adapted, strict=True)
