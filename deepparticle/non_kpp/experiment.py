"""Reproducible fixed-condition non-KPP comparison using the shared trainers.

Run from deepparticle/: python -m non_kpp.experiment --out ../output/non_kpp_20261020
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch
from particle_pipeline import engine
from particle_pipeline.models import MeanFlow, generate
from particle_pipeline.protocol import file_hash, fingerprint
from .models import CompactCorrector
from .physics import Physics, ball, sample_field, simulate, write_json
from .verify import verify


class LearningProtocol:
    dimension, periodic, family = 3, False, 'non_kpp'

    def __init__(self, physics):
        self.physics = physics
        self.name = physics.reaction + '3d'

    def condition(self, level):
        return 0.

    def metadata(self):
        return {'schema': 'non_kpp_sgip_v1', **asdict(self.physics),
                'dimension': 3, 'periodic': False, 'family': 'non_kpp',
                'learning_target': 'u(x,T) / integral u(x,T) dx',
                'coordinate_scale': self.physics.initial_radius,
                'prior': 'uniform unit 3-ball in scaled coordinates',
                'conditioning': 'constant zero; one model per fixed reaction/D/T'}


def factory(dimension, periodic, width, blocks, alpha):
    if periodic:
        raise ValueError('Non-KPP experiment is not periodic')
    return CompactCorrector(dimension, width=width, alpha=alpha)


def prepare(physics, folder, n_train, n_eval, device):
    simulate(physics, folder, device)
    settings = {'n_train': n_train, 'n_eval': n_eval, 'physics': asdict(physics),
                'sampling_seeds': {'train_target': 12345, 'train_prior': 22345,
                     'eval_target': 33345, 'eval_prior': 43345, 'eval_target_independent': 53345}}
    signature = fingerprint({**settings, 'field_sha256': file_hash(folder/'field.npz')})
    destination = folder/'samples.npz'
    if destination.exists():
        previous = json.loads((folder/'samples.json').read_text())
        if previous['signature'] != signature:
            raise ValueError('Existing sample protocol differs')
    else:
        with np.load(folder/'field.npz') as data:
            density = data['density']
        samples = {
            'train_target': sample_field(density, physics, n_train, 12345),
            'train_prior': ball(n_train, physics.initial_radius, 22345),
            'eval_target': sample_field(density, physics, n_eval, 33345),
            'eval_prior': ball(n_eval, physics.initial_radius, 43345),
            'eval_target_independent': sample_field(density, physics, n_eval, 53345)}
        # All stored samples retain physical units; training alone is scaled.
        np.savez_compressed(destination, **samples)
        write_json(folder/'samples.json', {**settings, 'signature': signature,
                 'reference': 'Independent draws from one fixed SGIP histogram, not independent PDE solves',
                 'target_representation': 'piecewise constant density; uniform jitter within bins'})
    with np.load(destination) as data:
        samples = {key: data[key] for key in data.files}
    return samples, signature


def train(physics, folder, samples, signature, args, seeds):
    protocol = LearningProtocol(physics)
    case = {'level': 0., 'prior': samples['train_prior']/physics.initial_radius,
            'target': samples['train_target']/physics.initial_radius}
    folder.mkdir(parents=True, exist_ok=True)
    config = {'protocol': protocol.metadata(), 'dataset_signature': signature,
              'training': vars(args), 'seeds': seeds,
              'budgets': {'MF': args.mf_steps+args.dp_steps,
                          'DP': args.mf_steps+args.dp_steps,
                          'MF_DP': [args.mf_steps, args.dp_steps]},
              'selection': 'Fixed final checkpoints; no test-set tuning',
              'status': 'training'}
    if (folder/'config.json').exists():
        old = json.loads((folder/'config.json').read_text())
        assert old['training'] == config['training'] and old['dataset_signature'] == signature
        assert old['seeds'] == seeds
    write_json(folder/'config.json', config)
    for seed in seeds:
        current = folder/f'seed{seed}'
        current.mkdir(exist_ok=True)
        timings = {}
        if (current/'timings.json').exists():
            timings = json.loads((current/'timings.json').read_text())
        if not (current/'mf.pt').exists() or torch.load(current/'mf.pt', map_location='cpu', weights_only=True)['step'] != args.mf_steps+args.dp_steps:
            _, _, timings['MF'] = engine.train_mf([case], protocol, args, current, signature, seed)
            write_json(current/'timings.json', timings)
        mf_init = MeanFlow(3, args.width).to(args.device)
        mf_init.load_state_dict(torch.load(current/'mf_init.pt', map_location='cpu', weights_only=True)['model'])
        mf_init.eval().requires_grad_(False)
        for method in ('dp', 'mf_dp'):
            desired_steps = args.dp_steps if method == 'mf_dp' else args.mf_steps+args.dp_steps
            path = current/f'{method}.pt'
            if path.exists() and torch.load(path, map_location='cpu', weights_only=True)['step'] == desired_steps:
                continue
            _, timings[method] = engine.train_dp([case], protocol, args, current, signature, seed,
                       mf=mf_init if method == 'mf_dp' else None, corrector_factory=factory)
            write_json(current/'timings.json', timings)
        write_json(current/'complete.json', {'seed': seed, 'status': 'complete',
                   'files': {name: file_hash(current/f'{name}.pt') for name in ('mf','mf_init','dp','mf_dp')}})
    config['status'] = 'complete'
    write_json(folder/'config.json', config)


def read_models(folder, seed, args, signature):
    result = {}
    for method in ('mf', 'mf_init', 'dp', 'mf_dp'):
        payload = torch.load(folder/f'seed{seed}'/f'{method}.pt', map_location='cpu', weights_only=True)
        assert payload['dataset_signature'] == signature
        model = (MeanFlow(3, args.width) if method.startswith('mf') and method != 'mf_dp'
                 else CompactCorrector(width=args.width, alpha=args.hybrid_alpha if method == 'mf_dp' else args.alpha))
        model.load_state_dict(payload['model'], strict=True)
        result[method] = model.to(args.device).eval().requires_grad_(False)
    return result


@torch.no_grad()
def infer(method, physical_prior, protocol, models, device):
    outputs = []
    scale = protocol.physics.initial_radius
    for start in range(0, len(physical_prior), 10000):
        x = torch.from_numpy(physical_prior[start:start+10000]/scale).to(device)
        condition = torch.zeros((len(x),1), device=device)
        mf = models['mf'] if method == 'mf' else models['mf_init']
        dp = models['mf_dp'] if method == 'mf_dp' else models['dp']
        outputs.append((generate('mf' if method == 'mf_init' else method, x, condition,
                                 protocol, mf=mf, dp=dp)*scale).cpu().numpy())
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite prediction')
    return result


def evaluate(physics, folder, samples, signature, args, seeds, n_metric, repeats):
    from .histogram_ot import Grid, histogram, score
    protocol = LearningProtocol(physics)
    with np.load(folder.parent.parent/'datasets'/physics.reaction/'field.npz') as field:
        ref_density = field['density'].astype(np.float64)
    ref_prob = ref_density/ref_density.sum()
    edges = [np.linspace(-physics.half_width, physics.half_width, physics.bins+1)]*3
    metric_indices = [np.random.default_rng(7000+i).choice(len(samples['eval_prior']), n_metric, replace=False) for i in range(repeats)]
    target_indices = [np.random.default_rng(8000+i).choice(len(samples['eval_target']), n_metric, replace=False) for i in range(repeats)]
    grid = Grid.from_reference(samples['eval_target'][target_indices[0]])
    reference_measures = [histogram(samples['eval_target'][ti],grid) for ti in target_indices]
    floor_values = [score(ref,histogram(samples['eval_target_independent'][si],grid))['w2_squared']
                    for si,ref in zip(metric_indices,reference_measures)]
    rows, distributions = [], {}
    for seed in seeds:
        models = read_models(folder, seed, args, signature)
        for method in ('mf', 'dp', 'mf_dp', 'mf_init'):
            cloud = infer(method, samples['eval_prior'], protocol, models, args.device)
            costs = [score(ref,histogram(cloud[si],grid))['w2_squared'] for si,ref in zip(metric_indices,reference_measures)]
            hist = np.histogramdd(cloud, bins=edges)[0]
            outside = len(cloud)-int(hist.sum())
            prob = hist/len(cloud)
            tv = .5*(np.abs(prob-ref_prob).sum()+outside/len(cloud))
            row = {'reaction': physics.reaction, 'seed': seed, 'method': method,
                   'w2': statistics.mean(math.sqrt(v) for v in costs),
                   'w2_squared': statistics.mean(costs), 'histogram_tv_100bins': float(tv),
                   'outside_fraction': outside/len(cloud), 'metric_particles': n_metric,
                   'metric_repeats': repeats, 'generated_particles': len(cloud),
                   'metric': 'full_coordinate_ks_histogram', 'bins_per_axis':24}
            for i, cost in enumerate(costs):
                row[f'w2_squared_repeat{i+1}'] = cost
            rows.append(row)
            distributions[f'{method}_seed{seed}'] = (prob*ref_density.sum()).astype(np.float32)
            np.savez_compressed(folder/f'seed{seed}'/f'{method}_generated.npz', particles=cloud)
            engine.csv_write(folder/'metrics_per_seed.csv', rows)
            print(f'{physics.reaction} seed={seed} {method} W2={row["w2"]:.5f}', flush=True)
    summary = []
    for method in ('mf','dp','mf_dp','mf_init'):
        subset = [row for row in rows if row['method'] == method]
        result = {'reaction': physics.reaction, 'method': method, 'seeds': len(seeds)}
        for key in ('w2', 'w2_squared', 'histogram_tv_100bins'):
            values = [r[key] for r in subset]
            result[key+'_mean'] = statistics.mean(values)
            result[key+'_sd'] = statistics.stdev(values) if len(values)>1 else 0.
        summary.append(result)
    engine.csv_write(folder/'summary.csv', summary)
    np.savez_compressed(folder/'predicted_densities.npz', **distributions)
    write_json(folder/'evaluation.json', {'status': 'complete', 'physical_units': True,
        'ground_metric': 'Euclidean; exact unregularized POT EMD, convergence required',
        'w2': 'mean of sqrt(EMD squared) over independent fixed subsamples per model seed',
        'w2_squared': 'mean of EMD squared over subsamples; not square of averaged W2',
        'uncertainty': 'Sample SD over model seeds, conditional on one fixed SGIP density',
        'metric_particles': n_metric, 'metric_repeats': repeats,
        'metric': 'full_coordinate_ks_histogram', 'evaluation_grid': grid.metadata(),
        'reference_sampling_floor_w2': statistics.mean(math.sqrt(v) for v in floor_values),
        'reference_sampling_floor_w2_squared': statistics.mean(floor_values),
        'reference_sampling_floor_squared_repeats': floor_values,
        'density_scale': 'Known SGIP mass supplied to every method; mass is not learned',
        'fdm_validated': False})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000, help='Equal total updates per method')
    parser.add_argument('--seeds', nargs='+', type=int, default=[88,89,90])
    parser.add_argument('--batch-size', type=int, default=2000)
    parser.add_argument('--n-train', type=int, default=20000)
    parser.add_argument('--n-eval', type=int, default=200000)
    parser.add_argument('--metric-particles', type=int, default=100000)
    parser.add_argument('--metric-repeats', type=int, default=1)
    parser.add_argument('--particles', type=int, default=5_000_000)
    parser.add_argument('--reactions', nargs='+', choices=['cubic','arrhenius'], default=['cubic','arrhenius'])
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--evaluate-only', action='store_true')
    cli = parser.parse_args()
    if cli.steps < 2 or cli.steps % 2:
        raise ValueError('Even positive total budget required')
    if not 1 <= cli.metric_particles <= cli.n_eval or len(set(cli.seeds)) != len(cli.seeds):
        raise ValueError('Invalid metric sample count or duplicate seeds')
    out = cli.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    args = SimpleNamespace(mf_steps=cli.steps//2, dp_steps=cli.steps//2, mf_lr=.001, dp_lr=.0001,
                           batch_size=cli.batch_size, width=64, mf_width=64, mf_hidden_layers=3,
                           dp_blocks=0, alpha=.1, hybrid_alpha=10., profile='original_tuned',
                           residual_weight=.1, batch_refresh=500, ot_refresh=500, device=cli.device)
    plan = {'training': vars(args), 'seeds': cli.seeds, 'source': 'Supplied cubic_3d and arrhenius_3d text',
            'n_train': cli.n_train, 'n_eval': cli.n_eval, 'metric_particles': cli.metric_particles,
            'metric_repeats': cli.metric_repeats, 'particles': cli.particles,
            'reactions': cli.reactions, 'scope': 'fixed D=1,T=20, not a parameter sweep; SGIP reference only'}
    if (out/'plan.json').exists():
        assert json.loads((out/'plan.json').read_text()) == plan, 'Existing experiment settings differ'
    write_json(out/'plan.json', plan)
    write_json(out/'port_verification.json', verify(cli.device))
    write_json(out/'runtime.json', {'torch': str(torch.__version__), 'numpy': np.__version__,
               'device': cli.device, 'gpu': torch.cuda.get_device_name(0) if cli.device=='cuda' else None,
               'pid': os.getpid()})
    source_root = Path(__file__).resolve().parents[1]
    paths = sorted((source_root/'non_kpp').rglob('*.py')) + [source_root/'particle_pipeline'/x for x in ('engine.py','models.py','geometry.py','protocol.py')]
    write_json(out/'source_manifest.json', {str(p.relative_to(source_root)): file_hash(p) for p in paths})
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(source_root)))
    all_summary = []
    for reaction in cli.reactions:
        write_json(out/'progress.json', {'reaction': reaction, 'stage': 'data', 'pid': os.getpid()})
        physics = Physics(reaction, particles=cli.particles)
        samples, signature = prepare(physics, out/'datasets'/reaction, cli.n_train, cli.n_eval, cli.device)
        folder = out/'runs'/reaction
        if not cli.evaluate_only:
            write_json(out/'progress.json', {'reaction': reaction, 'stage': 'training', 'pid': os.getpid()})
            train(physics, folder, samples, signature, args, cli.seeds)
        write_json(out/'progress.json', {'reaction': reaction, 'stage': 'evaluation', 'pid': os.getpid()})
        all_summary.extend(evaluate(physics, folder, samples, signature, args, cli.seeds,
                                    cli.metric_particles, cli.metric_repeats))
        engine.csv_write(out/'summary.csv', all_summary)
    write_json(out/'progress.json', {'stage': 'complete', 'pid': os.getpid()})


if __name__ == '__main__':
    main()
