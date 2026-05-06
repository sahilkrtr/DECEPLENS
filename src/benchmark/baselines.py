from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm

from ..utils.config import Config
from ..utils.hf_loader import load_causal_lm
from .extract import TrajectoryRecord, _build_chat, _pool_response_tokens


def base_view(records: List[TrajectoryRecord]) -> List[TrajectoryRecord]:
    return records


def magnitude_view(records: List[TrajectoryRecord]) -> List[TrajectoryRecord]:
    out = []
    for r in records:
        mags = np.linalg.norm(r.delta_h, axis=1, keepdims=True)
        out.append(TrajectoryRecord(
            model_name=r.model_name, language=r.language, domain=r.domain,
            t=r.t, c=r.c, s=r.s, tau=r.tau,
            delta_h=mags.astype(np.float32),
        ))
    return out


def causal_patching_view(
    records: List[Dict],
    model_name: str,
    cfg: Config,
    patch_layer: int | None = None,
) -> List[TrajectoryRecord]:
    tokenizer, model, device = load_causal_lm(
        model_name, dtype=cfg.trajectory.dtype, output_hidden_states=True,
    )
    pooling = cfg.trajectory.pooling
    n_layers = getattr(model.config, "num_hidden_layers", None) or len(model.transformer.h)
    patch = patch_layer if patch_layer is not None else n_layers // 2

    blocks = _decoder_blocks(model)
    target = blocks[min(patch, len(blocks) - 1)]
    handle = target.register_forward_hook(lambda m, inp, out: _zero_block_output(out))

    out: List[TrajectoryRecord] = []
    try:
        for rec in tqdm(records, desc=f"patch[{model_name.split('/')[-1]}@L{patch}]"):
            prompt = rec["prompt"]; response = rec["response"]
            if not isinstance(response, str) or not response.strip():
                continue
            full, r_start, r_end = _build_chat(tokenizer, prompt, response)
            inputs = tokenizer(full, return_tensors="pt").to(device)
            with torch.no_grad():
                out_t = model(**inputs, output_hidden_states=True, use_cache=False)
            h = _pool_response_tokens(out_t.hidden_states, r_start, r_end, pooling=pooling)
            delta_h = h[1:] - h[:-1]
            out.append(TrajectoryRecord(
                model_name=model_name, language=rec["language"], domain=rec["domain"],
                t=rec["t"], c=rec["c"], s=rec["s"], tau=int(rec["tau"]),
                delta_h=delta_h.astype(np.float32),
            ))
    finally:
        handle.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def _decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers          # LLaMA, Qwen2, Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h         # GPT-style
    raise RuntimeError("Cannot locate decoder block list on this model.")


def _zero_block_output(out):
    if isinstance(out, tuple):
        out = (torch.zeros_like(out[0]),) + out[1:]
        return out
    return torch.zeros_like(out)
