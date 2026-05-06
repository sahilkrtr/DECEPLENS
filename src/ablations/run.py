from __future__ import annotations
import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from ..utils.config import Config, load_config, ensure_dir, resolve_path
from ..utils.io import load_deceplens_xlsx, melt_to_long_format
from ..utils.simhash_dedup import is_duplicate
from ..benchmark.extract import extract_trajectories
from ..benchmark.metrics import condition_scores
from ..benchmark.compute_resources import ComputeStats, measure



def variant_full_phi(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return df.copy()


def variant_post_translation_cls(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    drift_p = 0.15
    by_t = {t: cfg.taxonomy.fine_grained_subtypes[t] for t in cfg.taxonomy.interaction_types}
    out = df.copy()
    new_s = []
    for _, r in out.iterrows():
        if r["language"] == "English":
            new_s.append(r["s"]); continue
        if rng.random() < drift_p:
            options = [s for s in by_t[r["t"]] if s != r["s"]]
            new_s.append(rng.choice(options) if options else r["s"])
        else:
            new_s.append(r["s"])
    out["s"] = new_s
    return out


def variant_single_sample_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rng = random.Random(cfg.seed + 1)
    flip_p = 0.10
    by_t = {t: cfg.taxonomy.fine_grained_subtypes[t] for t in cfg.taxonomy.interaction_types}
    out = df.copy()
    new_s = []
    for _, r in out.iterrows():
        if rng.random() < flip_p:
            options = [s for s in by_t[r["t"]] if s != r["s"]]
            new_s.append(rng.choice(options) if options else r["s"])
        else:
            new_s.append(r["s"])
    out["s"] = new_s
    return out


def variant_no_deduplication(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rng = random.Random(cfg.seed + 2)
    counts = Counter(zip(df["t"], df["c"], df["s"]))
    if not counts:
        return df.copy()
    cutoff = np.median(list(counts.values()))
    extras = []
    for _, r in df.iterrows():
        if counts[(r["t"], r["c"], r["s"])] <= cutoff and rng.random() < 0.30:
            extras.append(r.to_dict())
    out = pd.concat([df, pd.DataFrame(extras)], ignore_index=True)
    return out


def variant_direct_translation(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rng = random.Random(cfg.seed + 3)
    out = df.copy()
    new_resp = []
    for _, r in out.iterrows():
        text = str(r["response"])
        if r["language"] != "English" and rng.random() < 0.20:
            sents = [s for s in text.split(". ") if s.strip()]
            if len(sents) > 2:
                drop_idx = rng.randrange(len(sents))
                sents.pop(drop_idx)
                text = ". ".join(sents)
        new_resp.append(text)
    out["response"] = new_resp
    return out


def variant_no_response_filtering(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rng = random.Random(cfg.seed + 4)
    out = df.copy()
    keep_p = 0.20
    new_resp = []
    for _, r in out.iterrows():
        if rng.random() < keep_p:
            words = str(r["response"]).split()
            if len(words) > 4:
                rng.shuffle(words)
                new_resp.append(" ".join(words))
                continue
        new_resp.append(r["response"])
    out["response"] = new_resp
    return out


VARIANT_BUILDERS = {
    "full_phi":               variant_full_phi,
    "post_translation_cls":   variant_post_translation_cls,
    "single_sample_labels":   variant_single_sample_labels,
    "no_deduplication":       variant_no_deduplication,
    "direct_translation":     variant_direct_translation,
    "no_response_filtering":  variant_no_response_filtering,
}


def build_variant_full_mode(variant: str, cfg: Config) -> pd.DataFrame:
    from ..construct.load_mmlu_pro import load_mmlu_pro
    from ..construct.classify_taxonomy import classify_prompts
    from ..construct.augment_balance import augment
    from ..construct.translate import translate_batch
    from ..construct.generate_responses import generate_responses
    from ..construct.score_responses import score_responses

    df = load_mmlu_pro(cfg.paths.mmlu_pro_repo)

    # 1. Classify (k = 1 for `single_sample_labels`)
    k = 1 if variant == "single_sample_labels" else cfg.hyperparameters.classification_voting_k

    if variant != "post_translation_cls":
        triples = classify_prompts(df["prompt"].tolist(), cfg, k=k)
        df["taxonomy"] = triples
        df = df[df["taxonomy"].notna()].copy()
        df[["t", "c", "s"]] = pd.DataFrame(df["taxonomy"].tolist(), index=df.index)
        df = df.drop(columns=["taxonomy"])

    # 2. Augment (skip dedup if no_deduplication)
    if variant != "no_deduplication" and "t" in df.columns:
        from ..construct.augment_balance import find_low_frequency_labels
        label_counts = Counter(zip(df["t"], df["c"], df["s"]))
        median_count = int(pd.Series(list(label_counts.values())).median())
        aug = augment(df["prompt"].tolist(), label_counts, target_per_label=median_count, cfg=cfg)
        if aug:
            df = pd.concat([df, pd.DataFrame(aug)], ignore_index=True)

    # 3. Translate (skip round-trip if direct_translation)
    base = df.to_dict(orient="records")
    if variant == "direct_translation":
        cfg.hyperparameters.semantic_similarity_threshold = -1.0  # accept everything
    translations = translate_batch(base, cfg)

    # post_translation_cls: classify AFTER translation
    if variant == "post_translation_cls":
        triples = classify_prompts([t["prompt_translated"] for t in translations], cfg)
        rows = []
        for tr, lbl in zip(translations, triples):
            if lbl is None:
                continue
            tr["t"], tr["c"], tr["s"] = lbl
            rows.append(tr)
        translations = rows

    # 4. Responses
    responses = generate_responses(translations, cfg)

    # 5. Score / filter
    scored = score_responses(responses, cfg)
    if variant == "no_response_filtering":
        return pd.DataFrame(scored).drop(columns=["keep"], errors="ignore")
    return pd.DataFrame([r for r in scored if r["keep"]]).drop(columns=["keep"], errors="ignore")


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------

def df_to_records(df: pd.DataFrame) -> List[Dict]:
    out = []
    for _, r in df.iterrows():
        prompt = r.get("prompt_en") or r.get("prompt") or r.get("prompt_translated")
        out.append({
            "prompt": prompt, "response": r["response"],
            "language": r["language"], "domain": r["domain"],
            "t": r["t"], "c": r["c"], "s": r["s"], "tau": int(r["tau"]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["resample", "full"], default="resample",
                    help="resample: perturb the curated DECEPLENS to emulate variants. "
                         "full: re-run the whole construction pipeline (requires API).")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Subset of variants to run.")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(resolve_path(cfg, cfg.paths.outputs_dir) / "ablations")

    variants = args.variants or list(cfg.ablations.variants)
    models = args.models if args.models else list(cfg.models.trajectory_models)

    if args.mode == "resample":
        print(
            "[!] --mode resample: variants are EMULATED by perturbing the curated "
            "DECEPLENS dataset (label flips, duplicate injection, etc.).\n"
            "(re-builds DECEPLENS for each variant; needs OPENAI_API_KEY + GPU)."
        )

    if args.mode == "resample":
        df_wide = load_deceplens_xlsx(cfg)
        df_long_full = melt_to_long_format(df_wide)
        if args.limit:
            df_long_full = df_long_full.head(args.limit).reset_index(drop=True)

    rows_out: List[Dict] = []

    for variant in variants:
        if args.mode == "resample":
            df_variant = VARIANT_BUILDERS[variant](df_long_full, cfg)
        else:
            df_variant = build_variant_full_mode(variant, cfg)
            if "tau" not in df_variant.columns:
                df_variant["tau"] = 1
        records = df_to_records(df_variant)

        for model_name in models:
            stats = ComputeStats()
            with measure(stats):
                traj = extract_trajectories(records, model_name, cfg)
                stats.tokens = sum(t.delta_h.shape[0] for t in traj)
            summaries = condition_scores(traj, eps=cfg.hyperparameters.emergence_layer_threshold)
            for s in summaries:
                rows_out.append({
                    "variant": variant, **s,
                    "tok_per_s": stats.tok_per_s,
                    "peak_mem_gb": stats.peak_mem_gb,
                    "gpu_hours": stats.gpu_hours,
                })
            print(f"[{variant} | {model_name}] {summaries}")

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(out_dir / "table_ablations.csv", index=False)
    df_out.to_json(out_dir / "table_ablations.json", orient="records", indent=2)
    print(f"\nSaved Table ablations to {out_dir / 'table_ablations.csv'}")


if __name__ == "__main__":
    main()
