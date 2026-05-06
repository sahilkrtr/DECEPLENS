from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import pandas as pd

from ..utils.config import Config, load_config, ensure_dir, resolve_path
from .load_mmlu_pro import load_mmlu_pro
from .classify_taxonomy import classify_prompts
from .augment_balance import augment
from .translate import translate_batch
from .generate_responses import generate_responses
from .score_responses import score_responses


def run(cfg: Config, n_limit: int | None = None) -> Path:
    out_dir = ensure_dir(resolve_path(cfg, cfg.paths.outputs_dir) / "constructed")


    df = load_mmlu_pro(cfg.paths.mmlu_pro_repo)
    if n_limit:
        df = df.head(n_limit).copy()


    triples = classify_prompts(df["prompt"].tolist(), cfg)
    df["taxonomy"] = triples
    df = df[df["taxonomy"].notna()].copy()
    df[["t", "c", "s"]] = pd.DataFrame(df["taxonomy"].tolist(), index=df.index)
    df.drop(columns=["taxonomy"], inplace=True)
    df.to_parquet(out_dir / "after_classify.parquet")


    label_counts = Counter(zip(df["t"], df["c"], df["s"]))
    median_count = int(pd.Series(list(label_counts.values())).median()) if label_counts else 0
    aug = augment(df["prompt"].tolist(), label_counts, target_per_label=median_count, cfg=cfg)
    if aug:
        df = pd.concat([df, pd.DataFrame(aug)], ignore_index=True)
    df.to_parquet(out_dir / "after_augment.parquet")


    base_records = df.rename(columns={"prompt": "prompt"}).to_dict(orient="records")
    translations = translate_batch(base_records, cfg)
    pd.DataFrame(translations).to_parquet(out_dir / "after_translate.parquet")


    responses = generate_responses(translations, cfg)
    pd.DataFrame(responses).to_parquet(out_dir / "after_responses.parquet")


    scored = score_responses(responses, cfg)
    scored_df = pd.DataFrame(scored)
    final = scored_df[scored_df["keep"]].drop(columns=["keep"]).copy()
    final.to_parquet(out_dir / "deceplens_final.parquet")

    print(f"[build_dataset] saved final to {out_dir / 'deceplens_final.parquet'} "
          f"({len(final)} rows)")
    return out_dir / "deceplens_final.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: limit number of MMLU-PRO rows processed.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, n_limit=args.limit)


if __name__ == "__main__":
    main()
