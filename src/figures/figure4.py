from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utils.config import Config, load_config, ensure_dir, resolve_path
from ..utils.io import load_deceplens_xlsx, melt_to_long_format
from ..benchmark.extract import extract_trajectories
from ..benchmark.metrics import condition_scores


def make_figure4(cfg: Config, records: List[Dict], models: List[str], out_path: Path):
    domains = list(cfg.domains)
    languages = list(cfg.languages)
    eps = cfg.hyperparameters.emergence_layer_threshold

    fig, axes = plt.subplots(1, len(languages), figsize=(4 * len(languages), 4), sharey=True)
    if len(languages) == 1:
        axes = [axes]

    for ax, lang in zip(axes, languages):
        lang_records = [r for r in records if r["language"] == lang]
        for model_name in models:
            traj = extract_trajectories(lang_records, model_name, cfg)
            avg_per_domain = []
            for d in domains:
                t_d = [t for t in traj if t.domain == d]
                if not t_d:
                    avg_per_domain.append(np.nan); continue
                summaries = condition_scores(t_d, eps=eps)
                avg_per_domain.append(np.mean([s["Avg"] for s in summaries]))
            ax.plot(range(1, len(domains) + 1), avg_per_domain,
                    marker="o", label=model_name.split("/")[-1])
        ax.set_title(lang)
        ax.set_xlabel("domain index (1..14)")
        ax.set_xticks(range(1, len(domains) + 1))
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Avg trajectory score")
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
    make_figure4(cfg, records, models, out_dir / "figure4_domain_lang.png")


if __name__ == "__main__":
    main()
