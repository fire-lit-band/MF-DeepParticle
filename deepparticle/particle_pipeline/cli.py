"""A single interface for all currently discussed particle systems."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path

from .protocol import NAMES, preset
from .data import prepare, import_archive, import_current_ks, write_json


def parser():
    p = argparse.ArgumentParser(description="Versioned KS/KPP particle data, MF/DP and genuine FK warm starts", allow_abbrev=False)
    commands = p.add_subparsers(dest="command", required=True)
    show = commands.add_parser("protocols", help="Print/save the physical presets and their evidence")
    show.add_argument("--out", type=Path)
    archive = commands.add_parser("import-archive", help="Preserve supplied KPP caches as historical source evidence")
    archive.add_argument("--archive", type=Path, required=True)
    archive.add_argument("--out", type=Path, required=True)
    ks = commands.add_parser("import-ks", help="Adapt the existing audited KS3D data to the shared schema")
    ks.add_argument("--run", type=Path, required=True)
    ks.add_argument("--out", type=Path, required=True)
    data = commands.add_parser("prepare", help="Generate/verify canonical data with independent train/eval physical seeds")
    data.add_argument("--model", choices=NAMES, required=True)
    data.add_argument("--out", type=Path, required=True)
    data.add_argument("--train-seed", type=int, default=666)
    data.add_argument("--eval-seed", type=int, default=999)
    data.add_argument("--n-train", type=int)
    data.add_argument("--n-eval", type=int)
    data.add_argument("--generations", type=int)
    data.add_argument("--train-levels", nargs="+", type=float)
    data.add_argument("--eval-levels", nargs="+", type=float)
    data.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    data.add_argument("--workers", type=int, default=1, help="Independent CPU KPP data cases; KS stays sequential")
    data.add_argument("--purpose", choices=("experiment", "pilot"), default="experiment")
    data.add_argument("--smoke", action="store_true", help="64 particles, short KPP horizon and one train/eval level; labelled smoke")
    training = commands.add_parser("train", help="Fixed total-budget MF, direct DP and MF+DP, plus MF initializer ablation")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--mf-steps", type=int, default=10000)
    training.add_argument("--dp-steps", type=int, default=10000)
    training.add_argument("--mf-lr", type=float, default=.001)
    training.add_argument("--dp-lr", type=float, default=.0001)
    training.add_argument("--batch-size", type=int, default=2000)
    training.add_argument("--width", type=int, default=64)
    training.add_argument("--mf-width", type=int, help="MF-only hidden width; defaults to --width")
    training.add_argument("--mf-hidden-layers", type=int, default=3)
    training.add_argument("--dp-blocks", type=int, default=2)
    training.add_argument("--alpha", type=float, default=.1)
    training.add_argument("--hybrid-alpha", type=float, help="MF+DP correction scale; defaults to --alpha")
    training.add_argument("--profile", choices=("unified", "original_tuned"), default="unified", help="original_tuned preserves original KS MF features/RNG and DP source/batch/loss conventions")
    training.add_argument("--residual-weight", type=float, default=1.)
    training.add_argument("--batch-refresh", type=int, default=50)
    training.add_argument("--ot-refresh", type=int, default=50)
    training.add_argument("--seeds", nargs="+", type=int, default=[88, 89, 90])
    training.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluation = commands.add_parser("evaluate", help="Recompute distributions/metrics and optional genuine FK hot-start traces")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluation.add_argument("--bins", type=int)
    evaluation.add_argument("--metric-particles", type=int, default=2000)
    evaluation.add_argument("--metric-seed", type=int, default=999)
    evaluation.add_argument("--metric-workers", type=int, default=3, help="Independent exact histogram OT solves, capped by available RAM")
    evaluation.add_argument("--fk", action="store_true")
    evaluation.add_argument("--fk-generations", type=int, default=128)
    evaluation.add_argument("--fk-repeats", type=int, default=3)
    evaluation.add_argument("--fk-seed", type=int, default=1234)
    evaluation.add_argument("--fk-levels", nargs="+", type=float, default=[3., 3.75, 4.5])
    evaluation.add_argument("--fk-tolerance", type=float, default=.01)
    evaluation.add_argument("--fk-workers", type=int, default=1, help="Independent FK evolutions; concurrent wall times are diagnostics")
    evaluation.add_argument("--fk-backend", choices=("auto", "cpu", "cuda"), default="auto", help="auto follows --device; float64 and paired NumPy RNG in either backend")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "protocols":
        profiles = {name: {"physics": preset(name).metadata(), "evidence": preset(name).evidence(),
                           "train_levels": preset(name).train_levels, "eval_levels": preset(name).eval_levels}
                    for name in NAMES}
        if args.out:
            write_json(args.out, profiles)
        else:
            print(json.dumps(profiles, ensure_ascii=False, indent=2))
        return profiles
    if args.command == "import-archive":
        result = import_archive(args.archive, args.out)
    elif args.command == "import-ks":
        result = import_current_ks(args.run, args.out)
    elif args.command == "prepare":
        protocol = preset(args.model)
        if args.smoke:
            protocol = replace(protocol, n_train=64, n_eval=64, generations=2)
        protocol = replace(protocol, **{key: getattr(args, key) for key in ("n_train", "n_eval", "generations") if getattr(args, key) is not None})
        train_levels, eval_levels = args.train_levels, args.eval_levels
        if args.smoke:
            train_levels = train_levels or ([3.] if protocol.periodic else [50.])
            eval_levels = eval_levels or ([4.5] if protocol.periodic else [160.])
        result = prepare(protocol, args.out, train_seed=args.train_seed, eval_seed=args.eval_seed,
                         train_levels=train_levels, eval_levels=eval_levels, device=args.device,
                         purpose="smoke" if args.smoke else args.purpose, workers=args.workers)
    else:
        from .engine import train, evaluate
        result = train(args.data, args.out, args) if args.command == "train" else evaluate(args.run, args)
    print(str(result), flush=True)
    return result
