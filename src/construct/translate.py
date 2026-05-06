from __future__ import annotations
import os
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from ..utils.config import Config
from ..utils.prompts import TRANSLATE_PROMPT, BACK_TRANSLATE_PROMPT
from ..utils.round_trip import round_trip_similarity


def _openai_client(cfg: Config):
    from openai import OpenAI
    api_key = os.environ.get(cfg.openai.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Set {cfg.openai.api_key_env} for translation/response generation."
        )
    return OpenAI(api_key=api_key, base_url=cfg.openai.api_base)


def _chat(client, model: str, prompt: str, temperature: float = 0.0,
          retries: int = 3, sleep: float = 2.0) -> str:
    last = None
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(sleep)
    raise RuntimeError(f"OpenAI call failed after {retries} retries: {last}")


def translate_batch(
    records: List[Dict],
    cfg: Config,
    target_languages: Optional[List[str]] = None,
) -> List[Dict]:

    target_languages = target_languages or [l for l in cfg.languages if l != "English"]
    client = _openai_client(cfg)
    sem_model = cfg.trajectory.semantic_model
    delta = cfg.hyperparameters.semantic_similarity_threshold

    out = []
    for rec in tqdm(records, desc="translate"):
        for lang in target_languages:
            tr_prompt = TRANSLATE_PROMPT.format(
                language=lang, domain=rec["domain"], t=rec["t"], c=rec["c"], s=rec["s"],
                prompt=rec["prompt"],
            )
            tr = _chat(client, cfg.models.translator, tr_prompt, temperature=0.0)
            bt_prompt = BACK_TRANSLATE_PROMPT.format(language=lang, text=tr)
            bt = _chat(client, cfg.models.translator, bt_prompt, temperature=0.0)
            sim = round_trip_similarity([rec["prompt"]], [bt], sem_model)[0]
            if sim < delta:
                continue
            out.append({
                "prompt_en": rec["prompt"], "domain": rec["domain"],
                "t": rec["t"], "c": rec["c"], "s": rec["s"],
                "language": lang, "prompt_translated": tr,
                "_round_trip_sim": sim,
            })
    return out
