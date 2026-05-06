from __future__ import annotations
import os
import time
from typing import Dict, List, Optional
from tqdm import tqdm

from ..utils.config import Config
from ..utils.prompts import RESPONSE_PROMPT_SINGLE, RESPONSE_PROMPT_MULTI


def _openai_client(cfg: Config):
    from openai import OpenAI
    api_key = os.environ.get(cfg.openai.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Set {cfg.openai.api_key_env} for translation/response generation."
        )
    return OpenAI(api_key=api_key, base_url=cfg.openai.api_base)


def _chat(client, model: str, prompt: str, temperature: float = 0.7,
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


def generate_responses(
    records: List[Dict],
    cfg: Config,
    taus: Optional[List[int]] = None,
) -> List[Dict]:

    taus = taus or list(cfg.generation.taus)
    client = _openai_client(cfg)
    out = []
    for rec in tqdm(records, desc="generate-responses"):
        for tau in taus:
            template = RESPONSE_PROMPT_SINGLE if tau == 1 else RESPONSE_PROMPT_MULTI
            text_in = template.format(
                domain=rec["domain"], t=rec["t"], c=rec["c"], s=rec["s"],
                language=rec["language"],
                prompt=rec.get("prompt_translated", rec.get("prompt_en")),
            )
            response = _chat(client, cfg.models.responder, text_in, temperature=0.7)
            new_rec = {**rec, "tau": tau, "response": response}
            out.append(new_rec)
    return out
