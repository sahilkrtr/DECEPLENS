from __future__ import annotations
import json
import re
import statistics
from collections import Counter
from typing import List, Optional, Tuple, Dict

import torch
from tqdm import tqdm

from ..utils.config import Config
from ..utils.hf_loader import load_causal_lm
from ..utils.prompts import fewshot_taxonomy_messages


_VALID_T = {"Verbal", "Behavioral", "Structural"}
_VALID_C = {"Falsification", "Concealment", "Equivocation"}
_VALID_S = {
    "False Assertion", "Strategic Omission", "Misleading Framing", "Sycophantic Misrepresentation",
    "Covert Action", "Plausible Deniability", "Camouflage Execution", "Evidence Tampering",
    "Lock-in Creation", "Oversight Sabotage", "Audit Manipulation", "Precedent Engineering",
}


def _parse_label(text: str) -> Optional[Tuple[str, str, str]]:
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    t = (obj.get("interaction") or "").strip()
    c = (obj.get("cognitive") or "").strip()
    s = (obj.get("subtype") or "").strip()
    if t in _VALID_T and c in _VALID_C and s in _VALID_S:
        return (t, c, s)
    return None


def _majority_vote(labels: List[Optional[Tuple[str, str, str]]]) -> Optional[Tuple[str, str, str]]:
    valid = [l for l in labels if l is not None]
    if not valid:
        return None
    counts = Counter(valid).most_common()
    top, n_top = counts[0]
    if len(counts) > 1 and counts[1][1] == n_top:
        return None  # tie => unreliable
    if n_top * 2 <= len(labels):
        return None  # not a strict majority
    return top


def classify_prompts(
    prompts: List[str],
    cfg: Config,
    k: Optional[int] = None,
    temperature: Optional[float] = None,
    max_new_tokens: int = 64,
) -> List[Optional[Tuple[str, str, str]]]:
    """Run Mistral-7B-Instruct k times per prompt and majority-vote the labels."""
    k = k if k is not None else cfg.hyperparameters.classification_voting_k
    T = temperature if temperature is not None else cfg.hyperparameters.classification_temperature
    tokenizer, model, device = load_causal_lm(
        cfg.models.taxonomy_classifier,
        dtype=cfg.trajectory.dtype,
        output_hidden_states=False,
    )

    out_labels: List[Optional[Tuple[str, str, str]]] = []
    for prompt in tqdm(prompts, desc="taxonomy-classify"):
        msgs = fewshot_taxonomy_messages(prompt)
        chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors="pt").to(device)
        sample_labels: List[Optional[Tuple[str, str, str]]] = []
        for _ in range(k):
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(T, 1e-3),
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            sample_labels.append(_parse_label(text))
        out_labels.append(_majority_vote(sample_labels))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_labels
