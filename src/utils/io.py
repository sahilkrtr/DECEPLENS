from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import pandas as pd

from .config import Config, resolve_path



LANG_COLUMNS = {
    "English":    ("Eng_Single_Step",        "Eng_Multi_Step"),
    "Portuguese": ("Portuguese_Single_Step", "Portuguese_Multi_Step"),
    "Spanish":    ("Spanish_Single_Step",    "Spanish_Multi_Step"),
    "Italian":    ("Italian_Single_Step",    "Italian_Multi_Step"),
    "German":     ("German_Single_Step",     "German_Multi_Step"),
    "French":     ("French_Single_Step",     "French_Multi_Step"),
}


def load_deceplens_xlsx(cfg: Config) -> pd.DataFrame:
    path = resolve_path(cfg, cfg.paths.dataset_xlsx)
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def melt_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        base = {
            "prompt_en": r["prompt"],
            "domain": r["domain"],
            "t": r["interaction types"],
            "c": r["cognitive types"],
            "s": r["fine-grained subtypes"],
        }
        for lang, (col_s, col_m) in LANG_COLUMNS.items():
            rows.append({**base, "language": lang, "tau": 1, "response": r[col_s]})
            rows.append({**base, "language": lang, "tau": 4, "response": r[col_m]})
    return pd.DataFrame(rows)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
