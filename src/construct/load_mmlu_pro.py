from __future__ import annotations
import pandas as pd
from datasets import load_dataset


CATEGORY_TO_DOMAIN = {
    "biology": "biology",
    "business": "business",
    "chemistry": "chemistry",
    "computer science": "computer science",
    "economics": "economics",
    "engineering": "engineering",
    "health": "health",
    "history": "history",
    "law": "law",
    "math": "math",
    "philosophy": "philosophy",
    "physics": "physics",
    "psychology": "psychology",
    "other": "other",
}


def load_mmlu_pro(repo_id: str = "TIGER-Lab/MMLU-Pro", split: str = "test") -> pd.DataFrame:
    ds = load_dataset(repo_id, split=split)
    rows = []
    for ex in ds:
        cat = (ex.get("category") or "").strip().lower()
        domain = CATEGORY_TO_DOMAIN.get(cat, "other")
        rows.append({
            "prompt": ex["question"],
            "domain": domain,
            "options": ex.get("options"),
            "answer": ex.get("answer"),
        })
    return pd.DataFrame(rows)
