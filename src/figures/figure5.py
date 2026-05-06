from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

from ..utils.config import Config, load_config, ensure_dir, resolve_path
from ..utils.io import load_deceplens_xlsx, melt_to_long_format
from ..benchmark.extract import extract_trajectories
from ..benchmark.baselines import base_view, magnitude_view, causal_patching_view
from ..benchmark.metrics import grouped_aggregates


def _phase_split(L: int):
    third = L // 3
    return (slice(0, third), slice(third, 2 * third), slice(2 * third, L))


def _phase_stats(traj):
    if not traj:
        return None
    L = min(t.delta_h.shape[0] for t in traj)
    deltas = np.stack([t.delta_h[:L] for t in traj])
    early, mid, late = _phase_split(L)
    mean_traj = deltas.mean(axis=0)
    out = {}
    for name, sl in zip(["Early", "Mid", "Late"], [early, mid, late]):
        seg = deltas[:, sl, :]                                
        mag = np.linalg.norm(seg, axis=2).mean()

        seg_mean = mean_traj[sl]
        cos_vals = []
        for d in deltas:
            for l in range(seg_mean.shape[0]):
                a = d[sl][l]; b = seg_mean[l]
                na = np.linalg.norm(a); nb = np.linalg.norm(b)
                if na > 0 and nb > 0:
                    cos_vals.append(np.dot(a, b) / (na * nb))
        align = float(np.mean(cos_vals)) if cos_vals else 0.0

        sep = float(np.mean([np.linalg.norm(seg[i].mean(0) - seg[j].mean(0))
                             for i in range(min(20, len(seg)))
                             for j in range(i + 1, min(20, len(seg)))])) if len(seg) > 1 else 0.0
        out[name] = {"mag": float(mag), "align": align, "sep": sep}
    return out


def make_figure5(cfg: Config, records: List[Dict], models: List[str], out_path: Path):
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4 * max(1, len(models))))
    if len(models) == 1:
        axes = np.array([axes])

    for row_i, model_name in enumerate(models):
        traj = extract_trajectories(records, model_name, cfg)
        variants = {
            "Base":      base_view(traj),
            "Magnitude": magnitude_view(traj),
            "Causal":    causal_patching_view(records[:128], model_name, cfg),
            "+Φ":        traj,
        }
        phases = ["Early", "Mid", "Late"]
        for col_i, key in enumerate(["mag", "align", "sep"]):
            ax = axes[row_i, col_i]
            width = 0.20
            xs = np.arange(len(phases))
            for j, (v_name, v_traj) in enumerate(variants.items()):
                stats = _phase_stats(v_traj)
                if stats is None:
                    continue
                vals = [stats[p][key] for p in phases]
                ax.bar(xs + j * width, vals, width=width, label=v_name)
            ax.set_xticks(xs + 1.5 * width)
            ax.set_xticklabels(phases)
            ax.set_title(f"{model_name.split('/')[-1]} — {key}")
            ax.legend(fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    make_figure5(cfg, records, models, out_dir / "figure5_phase.png")


if __name__ == "__main__":
    main()
