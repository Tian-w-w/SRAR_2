"""Single-sample Qwen2 backend. No dataset scoring or candidate utilities here."""
from __future__ import annotations
import importlib.metadata
import time
from pathlib import Path
import torch
from .config import binding
from .data import validate_input
from .features import Observer, make_features
from .io import digest, file_hash


def asset_identity(model_path):
    base = Path(model_path)
    weights = sorted(base.glob("*.safetensors")) or sorted(base.glob("pytorch_model*.bin"))
    token_files = [base / x for x in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json") if (base / x).is_file()]
    if not (base / "config.json").is_file() or not weights or not token_files:
        raise ValueError("missing local model/config/tokenizer assets")
    hashes = {p.name: file_hash(p) for p in [base / "config.json", *weights, *token_files]}
    if (base / "generation_config.json").is_file():
        hashes["generation_config.json"] = file_hash(base / "generation_config.json")
    return {"model_sha256": digest({p: h for p, h in hashes.items() if p not in {x.name for x in token_files}}),
            "tokenizer_sha256": digest({p.name: hashes[p.name] for p in token_files}), "files": hashes}


class QwenBackend:
    def __init__(self, config, model_path, device="cuda:0"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from .vendor import native_operator as op
        if importlib.metadata.version("transformers") != "4.53.2":
            raise ValueError("native operator is pinned to transformers 4.53.2")
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("real BF16 experiment requires a CUDA GPU; use CPU tests here")
        self.config, self.device, self.op, self.registry = config, device, op, ALL_ATTENTION_FUNCTIONS
        self.assets = asset_identity(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                    torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
        mc = self.model.config
        if (mc.model_type, mc.hidden_size, mc.num_hidden_layers, mc.num_attention_heads, mc.num_key_value_heads) != ("qwen2", 3584, 28, 28, 4):
            raise ValueError("unexpected Qwen2-7B architecture")
        origin = str(getattr(mc, "_name_or_path", "")).lower()
        if "instruct" in str(model_path).lower() or "instruct" in origin:
            raise ValueError("Instruct checkpoint is not the Base protocol")
        self.model.requires_grad_(False)
        torch.manual_seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.calls = 0

    @property
    def identity(self):
        return {**binding(self.config), **{k: v for k, v in self.assets.items() if k != "files"},
                "torch": str(torch.__version__), "transformers": importlib.metadata.version("transformers"),
                "cuda_runtime": str(torch.version.cuda), "gpu": torch.cuda.get_device_name(self.device),
                "allow_tf32": False, "deterministic_algorithms": "warn_only"}

    def generate(self, x, anchor, gamma, capture=False):
        # Only public input fields are read. Every generate starts a new cache.
        validate_input(x)
        c = self.config
        encoded = self.tokenizer(x["raw_prompt"], return_tensors="pt", add_special_tokens=c["add_special_tokens"], truncation=False)
        ids = encoded["input_ids"][0].tolist()
        if len(ids) != x["prompt_tokens"] or digest(ids) != x["input_ids_sha256"]:
            raise ValueError("token IDs differ from audited manifest")
        payload = {k: v.to(self.device) for k, v in encoded.items()}
        observer = Observer(x, c) if capture else None
        if capture and (anchor != 0 or gamma != c["gamma0"]):
            raise ValueError("features may only observe fixed initial pass")
        original = self.op.attention
        started = time.perf_counter()
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            if anchor is None:
                self.op.disable(self.model)
            else:
                self.op.configure(self.model, int(anchor), c["selected_heads"], float(gamma), c["target_layers"])
            if observer is not None:
                self.registry[self.op.BACKEND] = observer.wrap(original)
            self.calls += 1
            with torch.inference_mode():
                generated = self.model.generate(**payload, do_sample=False, num_beams=1, max_new_tokens=c["max_new_tokens"],
                    use_cache=True, pad_token_id=self.tokenizer.pad_token_id, return_dict_in_generate=True,
                    output_scores=capture, output_hidden_states=capture)
            torch.cuda.synchronize(self.device)
            features = make_features(x, generated, observer) if capture else None
            out_ids = generated.sequences[0, len(ids):].cpu().tolist()
            result = {"prediction_raw": self.tokenizer.decode(out_ids, skip_special_tokens=True).strip(),
                      "generated_token_ids": out_ids, "anchor_token_index": anchor, "gamma": gamma,
                      "latency_seconds": time.perf_counter() - started,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                      "fresh_cache": True, "capture": capture}
            return result, features
        finally:
            self.registry[self.op.BACKEND] = original
            self.op.disable(self.model)
