from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.config import Config, load_config, ensure_dir, resolve_path
from ..utils.io import load_deceplens_xlsx, melt_to_long_format, save_json
from .extract import extract_trajectories, TrajectoryRecord, save_trajectories, load_trajectories
from .baselines import base_view, magnitude_view, causal_patching_view
from .metrics import condition_scores
from .compute_resources import ComputeStats, measure


def build_record_dicts(df_long: pd.DataFrame) -> List[Dict]:
    out = []
    for _, r in df_long.iterrows():
        out.append({
            "prompt": r["prompt_en"], "response": r["response"],
            "language": r["language"], "domain": r["domain"],
            "t": r["t"], "c": r["c"], "s": r["s"], "tau": int(r["tau"]),
        })
    return out


def run_for_model(model_name: str, records: List[Dict], cfg: Config, out_dir: Path) -> Dict:
    cache_dir = ensure_dir(resolve_path(cfg, cfg.paths.cache_dir))
    safe = model_name.replace("/", "__")
    cache_full = cache_dir / f"{safe}.npz"

    full_stats = ComputeStats()
    if cache_full.exists():
        traj_full = load_trajectories(cache_full)
        n_tokens = sum(r.delta_h.shape[0] for r in traj_full)
        full_stats.tokens = n_tokens
    else:
        with measure(full_stats):
            traj_full = extract_trajectories(records, model_name, cfg, cache_path=cache_full)
            full_stats.tokens = sum(r.delta_h.shape[0] for r in traj_full)

    eps = cfg.hyperparameters.emergence_layer_threshold

    # Variants of Φ
    variants = {
        "base":             base_view(traj_full),
        "magnitude":        magnitude_view(traj_full),
        "phi":              traj_full,                            # ours
    }

    causal_n = min(len(records), 256)
    causal_stats = ComputeStats()
    with measure(causal_stats):
        traj_causal = causal_patching_view(records[:causal_n], model_name, cfg)
        causal_stats.tokens = sum(r.delta_h.shape[0] for r in traj_causal)
    variants["causal_patching"] = traj_causal

    rows = []
    for variant, traj in variants.items():
        summaries = condition_scores(traj, eps=eps)
        for s in summaries:
            row = dict(s)
            row["variant"] = variant
            row["model_name"] = model_name
            stats = causal_stats if variant == "causal_patching" else full_stats
            row.update({
                "tok_per_s": stats.tok_per_s,
                "peak_mem_gb": stats.peak_mem_gb,
                "gpu_hours": stats.gpu_hours,
            })
            rows.append(row)
    return {"summaries": rows, "n_traj": len(traj_full)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: limit the number of long-format rows.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Optional override of cfg.models.trajectory_models.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(resolve_path(cfg, cfg.paths.outputs_dir) / "benchmark")

    df_wide = load_deceplens_xlsx(cfg)
    df_long = melt_to_long_format(df_wide)
    if args.limit:
        df_long = df_long.head(args.limit).reset_index(drop=True)
    records = build_record_dicts(df_long)

    models = args.models if args.models else list(cfg.models.trajectory_models)
    all_rows = []
    for model_name in models:
        print(f"\n=== Benchmark on {model_name} ===")
        result = run_for_model(model_name, records, cfg, out_dir)
        all_rows.extend(result["summaries"])

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(out_dir / "benchmark_results.csv", index=False)
    save_json(all_rows, out_dir / "benchmark_results.json")

    print(f"\nSaved benchmark results to {out_dir / 'benchmark_results.csv'}")
    print(df_out)


if __name__ == "__main__":
    main()
