from __future__ import annotations
import argparse
import sys

from .utils.config import load_config


def cmd_construct(args):
    from .construct.build_dataset import run as run_construct
    cfg = load_config(args.config)
    run_construct(cfg, n_limit=args.limit)


def cmd_benchmark(args):
    from .benchmark.run import main as bench_main
    sys.argv = ["benchmark"] + (["--config", args.config] if args.config else [])
    if args.limit: sys.argv += ["--limit", str(args.limit)]
    if args.models: sys.argv += ["--models", *args.models]
    bench_main()


def cmd_ablations(args):
    from .ablations.run import main as abl_main
    sys.argv = ["ablations", "--mode", args.mode]
    if args.config: sys.argv += ["--config", args.config]
    if args.limit: sys.argv += ["--limit", str(args.limit)]
    if args.models: sys.argv += ["--models", *args.models]
    abl_main()


def cmd_figures(args):
    from .figures.figure1 import main as f1
    from .figures.figure3 import main as f3
    from .figures.figure4 import main as f4
    from .figures.figure5 import main as f5
    base_argv = []
    if args.config: base_argv += ["--config", args.config]
    if args.limit: base_argv += ["--limit", str(args.limit)]
    if args.models: base_argv += ["--models", *args.models]
    for fn in (f1, f3, f4, f5):
        sys.argv = ["fig"] + base_argv
        fn()


def cmd_all(args):
    cmd_benchmark(args)
    cmd_ablations(args)
    cmd_figures(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["construct", "benchmark", "ablations", "figures", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default="resample", choices=["resample", "full"])
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    {"construct": cmd_construct, "benchmark": cmd_benchmark, "ablations": cmd_ablations,
     "figures": cmd_figures, "all": cmd_all}[args.step](args)


if __name__ == "__main__":
    main()
