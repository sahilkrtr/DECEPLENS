from __future__ import annotations
from collections import Counter
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm

from ..utils.config import Config
from ..utils.hf_loader import load_causal_lm
from ..utils.prompts import AUGMENT_PROMPT
from ..utils.simhash_dedup import is_duplicate


def find_low_frequency_labels(
    label_counts: Dict[Tuple[str, str, str], int],
    quantile: float = 0.3,
) -> List[Tuple[str, str, str]]:

    if not label_counts:
        return []
    counts = sorted(label_counts.values())
    cutoff_idx = max(1, int(len(counts) * quantile))
    cutoff = counts[cutoff_idx - 1]
    return [lbl for lbl, n in label_counts.items() if n <= cutoff]


def augment(
    existing_prompts: List[str],
    label_to_count: Dict[Tuple[str, str, str], int],
    target_per_label: int,
    cfg: Config,
    domains: List[str] | None = None,
    max_new_tokens: int = 160,
) -> List[Dict]:

    domains = domains or list(cfg.domains)
    low_labels = find_low_frequency_labels(label_to_count)
    if not low_labels:
        return []

    tokenizer, model, device = load_causal_lm(
        cfg.models.augmenter,
        dtype=cfg.trajectory.dtype,
        output_hidden_states=False,
    )

    pool = list(existing_prompts)
    new_records: List[Dict] = []

    for (t, c, s) in tqdm(low_labels, desc="augment-low-freq"):
        deficit = max(0, target_per_label - label_to_count.get((t, c, s), 0))
        if deficit == 0:
            continue
        attempts, generated = 0, 0
        while generated < deficit and attempts < deficit * 4:
            d = domains[attempts % len(domains)]
            text_in = AUGMENT_PROMPT.format(domain=d, t=t, c=c, s=s)
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": text_in}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(chat, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_prompt = tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            ).strip()
            attempts += 1
            if not new_prompt:
                continue
            if is_duplicate(new_prompt, pool, cfg.hyperparameters.simhash_threshold):
                continue
            pool.append(new_prompt)
            new_records.append({
                "prompt": new_prompt, "domain": d, "t": t, "c": c, "s": s,
            })
            generated += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return new_records
