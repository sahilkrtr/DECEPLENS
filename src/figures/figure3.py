from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

from ..utils.config import Config, load_config, ensure_dir, resolve_path
from ..utils.io import load_deceplens_xlsx, melt_to_long_format
from ..benchmark.extract import extract_trajectories, load_trajectories
from ..benchmark.baselines import base_view, magnitude_view, causal_patching_view
from ..benchmark.metrics import (
    grouped_aggregates, trajectory_magnitude, trajectory_consistency,
    evolution_divergence, cumulative_shift, emergence_layer, compute_avg,
)


def _layer_curves(traj_records, eps: float):
    if not traj_records:
        return None
    L = min(t.delta_h.shape[0] for t in traj_records)
    deltas = np.stack([t.delta_h[:L] for t in traj_records])
    mean_traj = deltas.mean(axis=0)

    TM = np.linalg.norm(deltas, axis=2).mean(axis=0)            # (L,)
    TC = np.array([
        np.mean([
            np.dot(d[l], mean_traj[l])
            / (np.linalg.norm(d[l]) * np.linalg.norm(mean_traj[l]) + 1e-9)
            for d in deltas
        ])
        for l in range(L)
    ])
    CS = np.cumsum(TM)
    return {"TM": TM, "TC": TC, "CS": CS}


def make_figure3(cfg: Config, records, models, out_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    eps = cfg.hyperparameters.emergence_layer_threshold

    for row_i, tau in enumerate([1, 4]):
        sel = [r for r in records if r["tau"] == tau]
        for col_i, metric_name in enumerate(["TM", "TC", "CS", "Avg"]):
            ax = axes[row_i, col_i]
            for model_name in models:
                traj = extract_trajectories(sel, model_name, cfg)
                variants = {
                    "Base":      base_view(traj),
                    "Magnitude": magnitude_view(traj),
                    "Causal":    causal_patching_view(sel[:128], model_name, cfg),
                    "+Φ":        traj,
                }
                for v_name, v_traj in variants.items():
                    curves = _layer_curves(v_traj, eps=eps)
                    if curves is None:
                        continue
                    if metric_name == "Avg":
                        TM_n = curves["TM"] / (curves["TM"].max() + 1e-9)
                        TC_n = np.clip(curves["TC"], 0, 1)
                        CS_n = curves["CS"] / (curves["CS"].max() + 1e-9)
                        y = np.mean(np.stack([TM_n, TC_n, CS_n]), axis=0)
                    else:
                        y = curves[metric_name]
                    ax.plot(np.arange(1, len(y) + 1), y,
                            label=f"{model_name.split('/')[-1]} / {v_name}")
            ax.set_title(f"{metric_name} (τ={tau})")
            ax.set_xlabel("layer l")
            ax.legend(fontsize=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(resolve_path(cfg, cfg.paths.outputs_dir) / "figures")

    df_long = melt_to_long_format(load_deceplens_xlsx(cfg))
    if args.limit:
        df_long = df_long.head(args.limit).reset_index(drop=True)
    records = [{
        "prompt": r["prompt_en"], "response": r["response"],
        "language": r["language"], "domain": r["domain"],
        "t": r["t"], "c": r["c"], "s": r["s"], "tau": int(r["tau"]),
    } for _, r in df_long.iterrows()]

    models = args.models if args.models else list(cfg.models.trajectory_models)
    make_figure3(cfg, records, models, out_dir / "figure3_benchmark.png")


if __name__ == "__main__":
    main()
