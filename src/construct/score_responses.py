from __future__ import annotations
import json
import re
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm

from ..utils.config import Config
from ..utils.hf_loader import load_causal_lm
from ..utils.prompts import RUBRIC_PROMPT


def _parse_rubric(text: str) -> Tuple[int, int, int] | None:
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    try:
        return int(obj["interaction"]), int(obj["cognitive"]), int(obj["subtype"])
    except Exception:  # noqa: BLE001
        return None


def _split_multistep(text: str) -> List[str]:
    parts = re.split(r"(?:^|\n)\s*Step\s*\d+\s*[:.\-]\s*", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[:4]


def score_responses(records: List[Dict], cfg: Config, max_new_tokens: int = 64) -> List[Dict]:

    tokenizer, model, device = load_causal_lm(
        cfg.models.scorer,
        dtype=cfg.trajectory.dtype,
        output_hidden_states=False,
    )
    out = []
    for rec in tqdm(records, desc="score-responses"):
        prompt = rec.get("prompt_translated", rec.get("prompt_en"))
        responses = (
            _split_multistep(rec["response"]) if rec["tau"] != 1 else [rec["response"]]
        )
        if not responses:
            new_rec = {**rec, "score": 0, "keep": False}
            out.append(new_rec); continue

        per_step_scores = []
        for resp in responses:
            text_in = RUBRIC_PROMPT.format(t=rec["t"], c=rec["c"], s=rec["s"],
                                           prompt=prompt, response=resp)
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": text_in}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(chat, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=0.0, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            triple = _parse_rubric(text) or (0, 0, 0)
            per_step_scores.append(sum(triple))

        worst = min(per_step_scores)
        new_rec = {**rec, "score": worst, "keep": worst == cfg.hyperparameters.response_score_threshold}
        out.append(new_rec)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
