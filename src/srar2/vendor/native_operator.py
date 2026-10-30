"""Qwen2 ZeroTuning operator that inherits installed native SDPA semantics.

The body mirrors transformers 4.53.2 sdpa_attention_forward; the sole semantic
delta is a clone-and-scale of expanded K after repeat_kv. This file deliberately
does not short-circuit gamma=1 when the forced intervention branch is enabled.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import torch
from torch.nn import functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2 import modeling_qwen2


BACKEND = "stage25q2_d1r_qwen2_native_sdpa_v2"


def _digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _changed_heads(before: torch.Tensor, after: torch.Tensor, tol: float = 1e-3) -> list[int]:
    delta = (after.float() - before.float()).abs()
    return torch.where(delta.amax(dim=(-2, -1))[0] > tol)[0].detach().cpu().tolist()


def _changed_tokens(before: torch.Tensor, after: torch.Tensor, tol: float = 1e-3) -> list[int]:
    delta = (after.float() - before.float()).abs()
    return torch.where(delta.amax(dim=(0, 1, 3)) > tol)[0].detach().cpu().tolist()


def _ratios(before: torch.Tensor, after: torch.Tensor, positions: list[int]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for position in positions:
        prior = torch.linalg.vector_norm(before[0, :, position, :].float(), dim=-1)
        current = torch.linalg.vector_norm(after[0, :, position, :].float(), dim=-1)
        result[str(position)] = (current / prior.clamp_min(1e-12)).detach().cpu().tolist()
    return result


def _heads(module, count: int) -> tuple[int, ...]:
    heads = tuple(int(value) for value in getattr(module, "_d1r_heads", ()))
    if len(heads) != len(set(heads)):
        raise ValueError("duplicate selected query heads")
    if any(head < 0 or head >= count for head in heads):
        raise IndexError(f"query head outside [0,{count})")
    return heads


def attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Installed native SDPA body with only the marked K intervention delta."""
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
        # Match native behavior: SDPA does not return attention weights here.
        pass

    if hasattr(module, "num_key_value_groups"):
        key = modeling_qwen2.repeat_kv(key, module.num_key_value_groups)
        value = modeling_qwen2.repeat_kv(value, module.num_key_value_groups)

    active = bool(getattr(module, "_d1r_enabled", False))
    selected = _heads(module, int(key.shape[1])) if active else ()
    key_before = key
    cache_before_hash = _digest(key) if active and bool(getattr(module, "_d1r_capture", False)) else None

    # The only intervention delta: modify a fresh expanded-K tensor, never the
    # pre-repeat KV cache tensor supplied by Qwen2 attention/cache handling.
    if active:
        key = key.clone()
        if selected:
            control = int(getattr(module, "_d1r_control", 0))
            if control < 0 or control >= key.shape[-2]:
                raise IndexError(f"control position {control} outside K length {key.shape[-2]}")
            key[:, selected, control, :] *= float(getattr(module, "_d1r_gamma", 1.0))
    cache_after_hash = _digest(key_before) if cache_before_hash is not None else None

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # These lines mirror installed transformers/integrations/sdpa_attention.py.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    if active and bool(getattr(module, "_d1r_capture", False)):
        seq = int(key.shape[-2])
        positions = sorted(set([0, min(1, seq - 1), seq // 2, seq - 1]))
        module._d1r_capture_result = {
            "layer": int(module.layer_idx),
            "selected_heads": list(selected),
            "gamma": float(getattr(module, "_d1r_gamma", 1.0)),
            "control_position": int(getattr(module, "_d1r_control", 0)),
            "positions": positions,
            "changed_heads": _changed_heads(key_before, key),
            "changed_tokens": _changed_tokens(key_before, key),
            "key_exact_identity": bool(torch.equal(key_before, key)),
            "ratios_by_head": _ratios(key_before, key, positions),
            "cache_before_hash": cache_before_hash,
            "cache_after_hash": cache_after_hash,
            "cache_unchanged_by_scale": cache_before_hash == cache_after_hash,
        }

    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    return output.transpose(1, 2).contiguous(), None


def configure(
    model,
    control_position: int,
    target_heads: Iterable[int],
    gamma: float,
    target_layers: Optional[Iterable[int]] = None,
    capture_layer: Optional[int] = None,
) -> None:
    ALL_ATTENTION_FUNCTIONS[BACKEND] = attention
    model.config._attn_implementation = BACKEND
    layers = None if target_layers is None else {int(value) for value in target_layers}
    heads = tuple(int(value) for value in target_heads)
    for module in model.modules():
        if isinstance(module, modeling_qwen2.Qwen2Attention):
            module._d1r_enabled = layers is None or int(module.layer_idx) in layers
            module._d1r_heads = heads
            module._d1r_gamma = float(gamma)
            module._d1r_control = int(control_position)
            module._d1r_capture = capture_layer is not None and int(module.layer_idx) == int(capture_layer)
            module._d1r_capture_result = None


def disable(model) -> None:
    for module in model.modules():
        if isinstance(module, modeling_qwen2.Qwen2Attention):
            module._d1r_enabled = False
            module._d1r_capture = False
            module._d1r_capture_result = None
    model.config._attn_implementation = "sdpa"
