from __future__ import annotations
import math, time
import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from . import native_operator as op
from . import scoring
LAYERS=(14,27)
HIDDEN=3584
ALT=("RE","RM","RL")
MAX_NEW={"sst2":1,"boolq":8,"logiqa":8}
MODEL_ID="Qwen/Qwen2-7B"
def _official_eval(): return scoring
def official_utility(task,text,references):
    value = parse_prediction(task,text)
    return float(value == int(references[0]))
def parse_prediction(task: str, text: str) -> int | None:
    evaluator = _official_eval()
    if task == 'sst2':
        value = evaluator.extract_label_from_response(text, [' negative', ' positive'])
        return int(value) if value in (0, 1) else None
    if task == 'boolq':
        value = evaluator.extract_label_from_response(text, [' False', ' True'])
        return int(value) if value in (0, 1) else None
    value = evaluator.extract_answer_from_response(text)
    return int(value) if value in (0, 1, 2, 3) else None

def attention_summaries(query, key, control: int, heads: list[int], gamma: float, qspan: list[int], candidates: list[int]) -> torch.Tensor:
    expanded = modeling_qwen2.repeat_kv(key, query.shape[1] // key.shape[1]).clone()
    expanded[:, heads, control, :] *= gamma
    q = query[0].float()
    k = expanded[0].float()
    (head_count, seq_len, dim) = q.shape
    (q0, q1) = map(int, qspan)
    result = []
    for candidate in candidates:
        blocks = []
        for start in range(0, seq_len, 32):
            end = min(seq_len, start + 32)
            scores = torch.matmul(q[:, start:end], k.transpose(-2, -1)) / math.sqrt(dim)
            rr = torch.arange(start, end, device=q.device)[:, None]
            cc = torch.arange(seq_len, device=q.device)[None, :]
            scores = scores.masked_fill(cc > rr, float('-inf'))
            denominator = torch.logsumexp(scores, dim=-1)
            probability = torch.exp(scores[:, :, candidate] - denominator)
            probability[:, torch.arange(start, end, device=q.device) < candidate] = 0
            blocks.append(probability.cpu())
        probabilities = torch.cat(blocks, dim=1)
        qvalues = probabilities[:, q0:q1]
        result.append(torch.stack((qvalues.mean(), qvalues.std(), qvalues.max(), probabilities[:, q1 - 1].mean(), probabilities.mean())))
    return torch.stack(result)

class Observer:

    def __init__(self, row: dict, frozen: dict):
        self.row = row
        self.frozen = frozen
        self.data: dict[int, torch.Tensor] = {}

    def wrap(self, original):

        def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
            layer = int(module.layer_idx)
            if layer in LAYERS and layer not in self.data and (query.shape[-2] > 1):
                self.data[layer] = attention_summaries(query, key, 0, self.frozen['selected_heads'], self.frozen['gamma'], self.row['query_token_span'], [int(self.row['controls'][name]) for name in ALT])
            return original(module, query, key, value, attention_mask, *args, **kwargs)
        return wrapped

def make_features(row: dict, generated, observer: Observer) -> tuple[np.ndarray, np.ndarray]:
    if set(observer.data) != set(LAYERS):
        raise RuntimeError(f'attention capture incomplete: {observer.data.keys()}')
    hidden = generated.hidden_states[0]
    (mid, last) = (hidden[LAYERS[0] + 1][0], hidden[LAYERS[1] + 1][0])
    (q0, q1) = map(int, row['query_token_span'])
    candidates = [int(row['controls'][name]) for name in ALT]
    vectors = torch.stack([mid[q0:q1].float().mean(0), last[q0:q1].float().mean(0), mid[0].float(), last[0].float(), *[mid[index].float() for index in candidates], *[last[index].float() for index in candidates]]).cpu().numpy().astype(np.float16)
    logits = generated.scores[0][0].float()
    probabilities = torch.softmax(logits, dim=-1)
    top = torch.topk(probabilities, 2).values
    uncertainty = np.asarray([float(top[0]), float(top[1]), float(top[0] - top[1]), float(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum())], dtype=np.float32)
    attention = torch.cat([observer.data[LAYERS[0]], observer.data[LAYERS[1]]], dim=1).numpy().astype(np.float32)
    scalars = np.empty((3, 16), dtype=np.float32)
    for (index, name) in enumerate(ALT):
        scalars[index] = np.asarray([*attention[index], *uncertainty, row['prompt_tokens'] / 4096.0, row['controls'][name] / max(1, row['prompt_tokens'] - 1)], dtype=np.float32)
    if vectors.shape != (10, HIDDEN) or scalars.shape != (3, 16):
        raise RuntimeError('feature shape mismatch')
    if not (np.isfinite(vectors).all() and np.isfinite(scalars).all()):
        raise RuntimeError('non-finite feature')
    return (vectors, scalars)

def generate_one(model, tokenizer, task: str, row: dict, anchor: str, frozen: dict, capture: bool=False) -> tuple[dict, tuple[np.ndarray, np.ndarray] | None]:
    vanilla = anchor == 'vanilla'
    control = None if vanilla else 0 if anchor == 'initial' else int(row['controls'][anchor])
    encoded = tokenizer(row['raw_prompt'], return_tensors='pt', add_special_tokens=False, truncation=False)
    if int(encoded['input_ids'].shape[1]) != int(row['prompt_tokens']):
        raise RuntimeError('Base prompt token count changed')
    payload = {key: value.to('cuda:0') for (key, value) in encoded.items()}
    if vanilla:
        op.disable(model)
    else:
        op.configure(model, control, frozen['selected_heads'], frozen['gamma'], frozen['target_layers'])
    original = op.attention
    observer = Observer(row, frozen) if capture else None
    if observer is not None:
        ALL_ATTENTION_FUNCTIONS[op.BACKEND] = observer.wrap(original)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**payload, do_sample=False, num_beams=1, max_new_tokens=MAX_NEW[task], use_cache=True, pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True, output_scores=capture, output_hidden_states=capture)
    torch.cuda.synchronize()
    if observer is not None:
        ALL_ATTENTION_FUNCTIONS[op.BACKEND] = original
    token_ids = generated.sequences[0, payload['input_ids'].shape[1]:].detach().cpu().tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    parsed = parse_prediction(task, text)
    utility = official_utility(task, text, row['gold_answers'])
    result = {'sample_id': row['sample_id'], 'anchor': anchor, 'control_token_index': control, 'generated_token_ids': token_ids, 'raw_generation': text, 'parsed_prediction': parsed, 'gold': int(row['gold']), 'score': float(utility), 'correct': bool(utility), 'generated_token_count': len(token_ids), 'invalid_output': parsed is None, 'first_token_eos': bool(token_ids and token_ids[0] == 151643), 'runtime_error': None, 'scorer_error': False, 'oom': False, 'latency_seconds': time.perf_counter() - started, 'model_id': MODEL_ID, 'model_variant': 'BASE', 'operator': None if vanilla else frozen['operator'], 'selected_heads': None if vanilla else frozen['selected_heads'], 'gamma': None if vanilla else frozen['gamma']}
    features = make_features(row, generated, observer) if observer is not None else None
    return (result, features)
