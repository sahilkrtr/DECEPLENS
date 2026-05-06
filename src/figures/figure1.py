from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

from ..utils.config import Config, load_config, ensure_dir, resolve_path
from ..utils.io import load_deceplens_xlsx, melt_to_long_format
from ..benchmark.extract import extract_trajectories


def make_figure1(cfg: Config, records: List[Dict], models: List[str], out_path: Path):
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4 * max(1, len(models))))
    if len(models) == 1:
        axes = np.array([axes])

    for row_i, model_name in enumerate(models):
        traj = extract_trajectories(records, model_name, cfg)
        if not traj:
            continue
        L = min(t.delta_h.shape[0] for t in traj)

        # By language
        ax = axes[row_i, 0]
        for lang in cfg.languages:
            t_l = [t for t in traj if t.language == lang]
            if not t_l:
                continue
            mags = np.stack([np.linalg.norm(t.delta_h[:L], axis=1) for t in t_l]).mean(axis=0)
            ax.plot(np.arange(1, L + 1), mags, label=lang)
        ax.set_title(f"{model_name.split('/')[-1]} — by language")
        ax.set_xlabel("layer l"); ax.set_ylabel("‖Δh^(l)‖₂")
        ax.legend(fontsize=7)

        # By domain (numerical index)
        ax = axes[row_i, 1]
        for d_idx, d in enumerate(cfg.domains):
            t_d = [t for t in traj if t.domain == d]
            if not t_d:
                continue
            mags = np.stack([np.linalg.norm(t.delta_h[:L], axis=1) for t in t_d]).mean(axis=0)
            ax.plot(np.arange(1, L + 1), mags, label=f"{d_idx + 1}.{d}")
        ax.set_title(f"{model_name.split('/')[-1]} — by domain (1..14)")
        ax.set_xlabel("layer l")
        ax.legend(fontsize=6, ncol=2)

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
    make_figure1(cfg, records, models, out_dir / "figure1_layerwise.png")


if __name__ == "__main__":
    main()
