from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..utils.config import Config, ensure_dir, resolve_path
from ..utils.hf_loader import load_causal_lm


@dataclass
class TrajectoryRecord:
    """Trajectory for one (prompt, response, model) pair."""
    model_name: str
    language: str
    domain: str
    t: str
    c: str
    s: str
    tau: int
    delta_h: np.ndarray   # shape (L-1, hidden_dim)


def _pool_response_tokens(
    hidden_states: Tuple[torch.Tensor, ...],
    response_start: int,
    response_end: int,
    pooling: str = "mean",
) -> np.ndarray:
    pooled = []
    for layer_h in hidden_states[1:]:           # drop embedding
        seg = layer_h[0, response_start:response_end, :]   # (T_resp, m)
        if seg.shape[0] == 0:
            pooled.append(torch.zeros(layer_h.shape[-1], dtype=layer_h.dtype, device=layer_h.device))
            continue
        if pooling == "last":
            pooled.append(seg[-1])
        elif pooling == "first":
            pooled.append(seg[0])
        else:                                 
            pooled.append(seg.mean(dim=0))
    return torch.stack(pooled, dim=0).float().cpu().numpy()


def _build_chat(tokenizer, prompt: str, response: str) -> Tuple[str, int, int]:
    """Build a chat string with prompt/response and return offsets for response tokens."""
    chat_pre = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    full = chat_pre + response
    pre_ids = tokenizer(chat_pre, return_tensors="pt").input_ids
    full_ids = tokenizer(full, return_tensors="pt").input_ids
    response_start = pre_ids.shape[1]
    response_end = full_ids.shape[1]
    return full, response_start, response_end


def extract_trajectories(
    records: List[Dict],
    model_name: str,
    cfg: Config,
    cache_path: Optional[Path] = None,
) -> List[TrajectoryRecord]:
    tokenizer, model, device = load_causal_lm(
        model_name,
        dtype=cfg.trajectory.dtype,
        output_hidden_states=True,
    )
    pooling = cfg.trajectory.pooling
    out: List[TrajectoryRecord] = []

    for rec in tqdm(records, desc=f"trajectory[{model_name.split('/')[-1]}]"):
        prompt = rec["prompt"]
        response = rec["response"]
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

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cache_path is not None:
        save_trajectories(out, cache_path)
    return out


def save_trajectories(records: List[TrajectoryRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        delta_h=np.stack([r.delta_h for r in records]) if records else np.zeros(0),
        meta=np.array([(r.model_name, r.language, r.domain, r.t, r.c, r.s, r.tau)
                       for r in records], dtype=object),
    )


def load_trajectories(path: Path) -> List[TrajectoryRecord]:
    data = np.load(path, allow_pickle=True)
    delta_h = data["delta_h"]
    meta = data["meta"]
    out: List[TrajectoryRecord] = []
    for i, (model_name, lang, domain, t, c, s, tau) in enumerate(meta):
        out.append(TrajectoryRecord(
            model_name=str(model_name), language=str(lang), domain=str(domain),
            t=str(t), c=str(c), s=str(s), tau=int(tau),
            delta_h=delta_h[i],
        ))
    return out
