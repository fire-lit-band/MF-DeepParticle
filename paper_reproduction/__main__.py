"""Standalone numerical reproduction commands."""
from pathlib import Path
import argparse
import json
from .locations import fresh_directory
from .workflow import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', help='Verify package inputs and numerical specifications')
    check.add_argument('--load-models', action='store_true')
    check.add_argument('--source-only', action='store_true', help='Check the source-only installation without retained data or weights')
    check.add_argument('--report', type=Path)
    figures = commands.add_parser('figures', help='Plot new results, or redraw the 21 frozen reference plots')
    figures.add_argument('--results', type=Path, help='Completed evaluate/reproduce results directory')
    figures.add_argument('--out', type=Path, required=True)
    data = commands.add_parser('data', help='Generate every physical training/evaluation/reference dataset without retained binary files')
    data.add_argument('--models', nargs='+', choices=MODELS, required=True)
    data.add_argument('--out', type=Path, required=True)
    for name, help_text in (
        ('train', 'Train on the retained physical datasets'),
        ('generate', 'Generate fresh 100k KS/KPP clouds from selected models'),
        ('evaluate', 'Recompute model predictions and numerical errors'),
        ('reproduce', 'Train, evaluate all selected experiments, and plot the new results'),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument('--models', nargs='+', choices=MODELS if name != 'generate' else MODELS[:-1], required=True)
        command.add_argument('--out', type=Path, required=True)
        command.add_argument('--data-run', type=Path, help='Completed data command output; inferred from a new training run when omitted')
        if name in ('generate', 'evaluate', 'reproduce'):
            command.add_argument('--model-run', type=Path, help='Completed train output; reproduce trains by default, other commands use retained weights by default')
        if name in ('evaluate', 'reproduce'):
            command.add_argument('--workers', type=int, default=1)
            command.add_argument('--reference-run', type=Path, help='Completed references output for fresh KPP physical references')
        if name in ('train', 'reproduce'):
            command.add_argument('--strict-replay', action='store_true', help='Require every trained tensor to equal the historical tensor')
        if name == 'evaluate':
            command.add_argument('--generated', type=Path, help='Completed generate output; validates its model provenance and skips generation')
            command.add_argument('--with-auxiliary', action='store_true', help='Also run two-step MF and KPP FK evolution')
        if name == 'reproduce':
            command.add_argument('--from-scratch', action='store_true', help='Generate all physical data, train all models, evaluate and plot; no old NPZ/PT files required')
    timing = commands.add_parser('timing', help='Measure inference time using selected models')
    timing.add_argument('--out', type=Path, required=True)
    timing.add_argument('--model-run', type=Path)
    timing.add_argument('--data-run', type=Path)
    timing.add_argument('--models', nargs='+', choices=MODELS[:-1], default=list(MODELS[:-1]))
    multi = commands.add_parser('multistep', help='Generate and evaluate all two-step KS3D MF cases')
    multi.add_argument('--model-run', type=Path)
    multi.add_argument('--data-run', type=Path)
    multi.add_argument('--out', type=Path, required=True)
    multi.add_argument('--workers', type=int, default=1)
    warm = commands.add_parser('warm-start', help='Run complete 20k-particle KPP FK warm-start comparisons')
    warm.add_argument('--models', nargs='+', choices=['kpp2d', 'kpp3d'], default=['kpp2d', 'kpp3d'])
    warm.add_argument('--model-run', type=Path)
    warm.add_argument('--data-run', type=Path)
    warm.add_argument('--out', type=Path, required=True)
    references = commands.add_parser('references', help='Regenerate 100k-particle KPP physical references from the fixed numerical protocol')
    references.add_argument('--models', nargs='+', choices=['kpp2d', 'kpp3d'], default=['kpp2d', 'kpp3d'])
    references.add_argument('--levels', type=float, nargs='+', help='Optional subset of the eleven fixed diffusion levels')
    references.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'check':
        from .checks import audit
        result = audit(load_models=args.load_models, source_only=args.source_only)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    elif args.command == 'figures':
        if args.results is not None:
            from .result_figures import render
            paths = render(args.out, results=args.results)
            result = {'status': 'complete', 'source': 'new experiment results', 'figures': len(paths), 'output': str(args.out.resolve())}
        else:
            from . import ks, kpp, cubic
            from .checks import expected_figures
            destination = fresh_directory(args.out, create=True)
            paths = ks.render(destination) + kpp.render(destination) + cubic.render(destination)
            if {p.relative_to(destination).as_posix() for p in paths} != expected_figures() or len(paths) != 21:
                raise ValueError('Frozen reference figure set is incomplete')
            result = {'status': 'passed', 'source': 'frozen results', 'figures': len(paths), 'output': str(destination)}
    elif args.command == 'data':
        from .data_runs import prepare_models
        result = prepare_models(args.models, args.out)
    elif args.command == 'train':
        from .workflow import train_models
        result = train_models(args.models, args.out, strict_replay=args.strict_replay, data_run=args.data_run)
    elif args.command == 'generate':
        from .workflow import generate_models
        result = generate_models(args.models, args.out, model_run=args.model_run, data_run=args.data_run)
    elif args.command == 'evaluate':
        from .workflow import evaluate_models
        result = evaluate_models(args.models, args.out, model_run=args.model_run, generated=args.generated,
                                 reference_run=args.reference_run, workers=args.workers, auxiliary=args.with_auxiliary, data_run=args.data_run)
    elif args.command == 'reproduce':
        from .workflow import reproduce
        result = reproduce(args.models, args.out, model_run=args.model_run, reference_run=args.reference_run,
                           workers=args.workers, strict_replay=args.strict_replay, data_run=args.data_run, from_scratch=args.from_scratch)
    elif args.command == 'timing':
        from .timing import benchmark
        result = benchmark(fresh_directory(args.out, create=True), model_run=args.model_run, models=args.models, data_run=args.data_run)
    elif args.command == 'multistep':
        from .ks_workflows import multistep
        from .workflow import write
        from .data_runs import resolve_data_run, stage
        selected_data = resolve_data_run(args.data_run, args.model_run)
        result = multistep(args.out, model_run=args.model_run, workers=args.workers,
                           data_run=stage(selected_data, 'ks', ['ks3d']))
        write(args.out / 'reproduction_results.json', {'schema': 1, 'status': 'complete', 'models': ['ks3d'], 'paths': {'multistep': '.'}})
    elif args.command == 'warm-start':
        from .kpp_workflows import warm_start
        from .workflow import write
        from .data_runs import resolve_data_run, stage
        selected_data = resolve_data_run(args.data_run, args.model_run)
        result = warm_start(args.models, args.out, model_run=args.model_run,
                            data_run=stage(selected_data, 'kpp', args.models))
        write(args.out / 'reproduction_results.json', {'schema': 1, 'status': 'complete', 'models': args.models, 'paths': {'warm': '.'}})
    else:
        from .kpp_workflows import references
        result = references(args.models, args.out, levels=args.levels, particles=100000)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
