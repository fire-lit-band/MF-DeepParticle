"""Reevaluate frozen non-KPP models with the KS3D 100k histogram-OT protocol.

The metric is exact weighted EMD between 3D bin centers, not 100k-by-100k
unbinned OT. A reference-anchored 24-cell grid is extended without changing
its spacing. At D=1, nested 2k controls use the identical grid and old draws.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np
import ot
import psutil
from scipy.spatial.distance import cdist
import torch

from particle_pipeline.engine import csv_write
from particle_pipeline.protocol import file_hash
from .experiment import read_models
from .physics import ball, write_json
from .sweep import DiffusionProtocol, TRAIN_D, dkey, infer

METHODS = ('mf', 'dp', 'mf_dp')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def nested_indices(pool, count, old_seed, extra_seed):
    first = np.random.default_rng(old_seed).choice(pool, 2000, replace=False)
    remaining = np.setdiff1d(np.arange(pool), first)
    extra = np.random.default_rng(extra_seed).choice(remaining, count-2000, replace=False)
    result = np.concatenate((first, extra))
    assert len(np.unique(result)) == count
    return result


def nested_ball(count, radius, seed):
    # ball(n, seed) is not prefix-invariant in n: preserve the old 2k draw
    # explicitly and use a separate independent stream for the added samples.
    return np.concatenate((ball(2000, radius, seed), ball(count-2000, radius, seed+1000000)))


def histogram(points, origin, width):
    """Infinite extension of a fixed lattice: no clipping or mass truncation."""
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
    index, counts = np.unique(np.floor((points-origin)/width).astype(np.int64),
                              axis=0, return_counts=True)
    assert int(counts.sum()) == len(points)
    positions = origin+(index+.5)*width
    weights = counts.astype(np.float64)/len(points)
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-14)
    return positions, weights, index, counts


def score(first, second):
    started = time.perf_counter()
    cost = cdist(first[0], second[0], 'sqeuclidean')
    value, info = ot.emd2(first[1], second[1], cost, log=True,
                          numItermax=5000000, numThreads=1)
    if info['result_code'] != 1 or info.get('warning'):
        raise RuntimeError(f'Exact histogram OT failed: {info}')
    if not np.isfinite(value) or value < -1e-10:
        raise ValueError(f'Invalid OT cost {value}')
    squared = max(0., float(value))
    return {'w2': math.sqrt(squared), 'w2_squared': squared,
            'first_occupied_bins': len(first[0]), 'second_occupied_bins': len(second[0]),
            'first_particles_retained': int(first[3].sum()),
            'second_particles_retained': int(second[3].sum()),
            'solver_result_code': int(info['result_code']),
            'seconds': time.perf_counter()-started}


def predict(method, source, d, protocol, models, device):
    return np.concatenate([infer(method, source[i:i+10000], d, protocol, models, device)
                           for i in range(0, len(source), 10000)])


def evaluate_case(root, out, reaction, d, plan, cli, models):
    destination = out/reaction/dkey(d)
    destination.mkdir(parents=True, exist_ok=True)
    case_path = destination/'complete.json'
    if case_path.exists():
        return read_json(case_path)
    started = time.perf_counter()
    protocol = DiffusionProtocol(reaction, plan['particles'])
    args = SimpleNamespace(**plan['training'])
    source_path = root/'datasets'/reaction/dkey(d)/'samples.npz'
    si = nested_indices(plan['n_eval_per_D'], cli.particles, 7000, 700000)
    ti = nested_indices(plan['n_eval_per_D'], cli.particles, 8000, 800000)
    with np.load(source_path) as samples:
        prior = samples['eval_prior'][si]
        reference = samples['eval_target'][ti]
        independent = samples['eval_target_independent'][si]
    # Same grid recipe as the inherited KS evaluation cache: 10k reference,
    # 24 bins per axis, tiny padding; later particles extend this lattice.
    anchor = reference[:10000].astype(np.float64)
    low, high = anchor.min(0), anchor.max(0)
    padding = np.maximum(high-low, 1.)*1e-6
    edges = np.stack([np.linspace(low[a]-padding[a], high[a]+padding[a], cli.bins+1)
                      for a in range(3)])
    origin, width = edges[:, 0], edges[:, 1]-edges[:, 0]
    sizes = [2000, cli.particles] if d == 1. else [cli.particles]
    measures, jobs, prefix_errors = {}, [], []
    archive = {'anchor_edges': edges, 'source_indices': si, 'target_indices': ti}

    def add(label, points):
        for n in sizes:
            value = histogram(points[:n], origin, width)
            key = f'{label}_n{n}'
            measures[key] = value
            archive[key+'_indices'] = value[2]
            archive[key+'_counts'] = value[3]

    def job(kind, method, seed, repeat, first, second):
        for n in sizes:
            jobs.append(({'reaction': reaction, 'D': d, 'kind': kind, 'method': method,
                          'seed': seed, 'repeat': repeat, 'particles': n},
                         f'{first}_n{n}', f'{second}_n{n}'))

    add('reference', reference)
    add('reference_independent', independent)
    job('reference_self', 'reference', 0, 1, 'reference', 'reference_independent')
    for seed in plan['seeds']:
        for method in METHODS:
            label = f'seed{seed}_{method}'
            generated = predict(method, prior, d, protocol, models[seed], args.device)
            add(label, generated)
            job('target_error', method, seed, 1, label, 'reference')
            if d == 1.:
                old = root/'evaluation'/reaction/dkey(d)/f'seed{seed}_metric_clouds.npz'
                with np.load(old) as previous:
                    np.testing.assert_array_equal(prior[:2000], previous['prior'][0])
                    np.testing.assert_array_equal(reference[:2000], previous['target'][0])
                    error = float(np.abs(generated[:2000]-previous[method][0]).max())
                    np.testing.assert_allclose(generated[:2000], previous[method][0], atol=5e-5, rtol=1e-5)
                    prefix_errors.append({'seed': seed, 'method': method, 'max_error': error})
                for repeat in range(3):
                    pair = []
                    for side, source_seed in [('A', 91001+repeat*10), ('B', 92003+repeat*10)]:
                        source = nested_ball(cli.particles, protocol.physics.initial_radius, source_seed)
                        generated_self = predict(method, source, d, protocol, models[seed], args.device)
                        self_label = f'{label}_self{repeat+1}_{side}'
                        add(self_label, generated_self)
                        pair.append(self_label)
                    job('model_self', method, seed, repeat+1, *pair)

    # Independent implementation check: sparse lattice counts agree exactly
    # with NumPy histogramdd on explicitly extended edges.
    ref_measure = measures[f'reference_n{cli.particles}']
    minidx, maxidx = ref_measure[2].min(0), ref_measure[2].max(0)
    extended = [origin[a]+np.arange(minidx[a], maxidx[a]+2)*width[a] for a in range(3)]
    dense = np.histogramdd(reference, bins=extended)[0]
    np.testing.assert_array_equal(dense[tuple((ref_measure[2]-minidx).T)], ref_measure[3])
    assert int(dense.sum()) == cli.particles
    np.savez_compressed(destination/'histograms.npz', **archive)
    # Bound memory using an intentionally conservative 8x cost-matrix estimate.
    largest = max(len(measures[a][0])*len(measures[b][0]) for _, a, b in jobs)
    capacity = max(1, int(psutil.virtual_memory().available*.6/(largest*8*8)))
    workers = min(cli.workers, capacity)
    print(f'{reaction} D={d:g}: {len(jobs)} OT solves, workers={workers}, '
          f'reference bins={len(ref_measure[0])}, widths={width.round(3)}', flush=True)
    rows_path = destination/'metrics.csv'
    if rows_path.exists():
        with rows_path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in ('D', 'w2', 'w2_squared', 'seconds'):
                row[key] = float(row[key])
            for key in ('seed', 'repeat', 'particles', 'first_occupied_bins', 'second_occupied_bins',
                        'first_particles_retained', 'second_particles_retained', 'solver_result_code'):
                row[key] = int(row[key])
    else:
        rows = []
    def rowkey(row):
        return tuple(row[k] for k in ('kind', 'method', 'seed', 'repeat', 'particles'))
    finished = {rowkey(row) for row in rows}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(score, measures[a], measures[b]): metadata
                   for metadata, a, b in jobs if rowkey(metadata) not in finished}
        for future in as_completed(futures):
            row = {**futures[future], **future.result()}
            assert row['first_particles_retained'] == row['second_particles_retained'] == row['particles']
            rows.append(row)
            csv_write(rows_path, rows)
            write_json(out/'progress.json', {'stage': 'OT', 'reaction': reaction, 'D': d,
                       'case_solves_done': len(rows), 'case_solves_total': len(jobs)})
            if len(rows) % 10 == 0 or len(rows) == len(jobs):
                print(f'  {reaction} D={d:g}: {len(rows)}/{len(jobs)} solved; '
                      f'latest={row["kind"]}/{row["method"]} n={row["particles"]} '
                      f'W2={row["w2"]:.5f} ({row["seconds"]:.1f}s)', flush=True)
    assert len(rows) == len(jobs) and len({rowkey(r) for r in rows}) == len(jobs)
    result = {'reaction': reaction, 'D': d, 'status': 'complete', 'rows': rows,
              'source_samples_sha256': file_hash(source_path), 'bin_widths': width.tolist(),
              'histograms_sha256': file_hash(destination/'histograms.npz'),
              'old_2k_prefix_checks': prefix_errors, 'all_mass_retained': True,
              'dense_histogram_count_check': True, 'seconds': time.perf_counter()-started}
    write_json(case_path, result)
    return result


def summarize(out, cases):
    rows = [r for case in cases for r in case['rows']]
    csv_write(out/'metrics.csv', rows)
    summary = []
    keys = sorted({(r['reaction'], r['D'], r['kind'], r['method'], r['particles']) for r in rows})
    for reaction, d, kind, method, n in keys:
        subset = [r for r in rows if (r['reaction'], r['D'], r['kind'], r['method'], r['particles'])
                  == (reaction, d, kind, method, n)]
        seeds = sorted({r['seed'] for r in subset})
        means = [statistics.mean(r['w2'] for r in subset if r['seed'] == seed) for seed in seeds]
        summary.append({'reaction': reaction, 'D': d, 'kind': kind, 'method': method, 'particles': n,
                        'w2_mean': statistics.mean(means),
                        'w2_seed_sd': statistics.stdev(means) if len(means)>1 else 0.,
                        'model_seeds': len(seeds) if kind != 'reference_self' else 0,
                        'repeats_per_seed': len(subset)//len(seeds)})
    csv_write(out/'summary.csv', summary)
    return summary


def plot(out, summary, cli):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'mf': '#5470C6', 'dp': '#D78A32', 'mf_dp': '#268A72'}
    labels = {'mf': 'MF', 'dp': 'DP', 'mf_dp': 'MF + DP refinement'}
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.5))
    fig.subplots_adjust(left=.065, right=.985, bottom=.2, top=.76, wspace=.22)
    for ax, reaction in zip(axes, ('cubic', 'arrhenius')):
        for method in METHODS:
            group = [r for r in summary if r['reaction'] == reaction and r['method'] == method
                     and r['kind'] == 'target_error' and r['particles'] == cli.particles]
            group.sort(key=lambda r: r['D'])
            x = np.array([r['D'] for r in group]); y = np.array([r['w2_mean'] for r in group])
            sd = np.array([r['w2_seed_sd'] for r in group])
            ax.plot(x, y, 'o-', color=colors[method], label=labels[method], lw=1.8, ms=4.5)
            ax.fill_between(x, y-sd, y+sd, color=colors[method], alpha=.13, linewidth=0)
        ref = sorted([r for r in summary if r['reaction'] == reaction and r['kind'] == 'reference_self'
                      and r['particles'] == cli.particles], key=lambda r: r['D'])
        ax.plot([r['D'] for r in ref], [r['w2_mean'] for r in ref], '--', color='#777777',
                label='Reference vs reference', lw=1.4)
        ax.plot(TRAIN_D, [.02]*len(TRAIN_D), '|', ms=10, color='#333333',
                transform=ax.get_xaxis_transform(), label='Training D')
        ax.set(title=reaction.capitalize(), xlabel='Diffusion coefficient D',
               ylabel=r'Histogram $W_2$ (physical coordinates)', xlim=(.2, 2.05))
        ax.set_xticks([.25, .5, .75, 1., 1.5, 2.]); ax.grid(axis='y', alpha=.2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .91), ncol=5,
               frameon=False, fontsize=10)
    fig.suptitle(f'Non-KPP | {cli.particles:,} particles per cloud | KS3D-style histogram OT', y=.98, fontsize=13)
    fig.text(.5, .055, 'Frozen conditional models; mean +/- SD over 3 model seeds. '
             'Exact weighted EMD on an anchored 3D grid (24 initial bins/axis).', ha='center', fontsize=9)
    fig.savefig(out/'diffusion_w2_100k.png', dpi=200)
    fig.savefig(out/'diffusion_w2_100k.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--particles', type=int, default=100000)
    parser.add_argument('--bins', type=int, default=24)
    parser.add_argument('--workers', type=int, default=2)
    cli = parser.parse_args()
    root = cli.run.resolve()
    plan = read_json(root/'plan.json')
    assert 10000 <= cli.particles <= plan['n_eval_per_D'] and cli.bins >= 2 and cli.workers >= 1
    out = root/f'histogram_ot_n{cli.particles}_bins{cli.bins}'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    hashes = {f'{r}/seed{s}/{m}': file_hash(root/'runs'/r/f'seed{s}'/f'{m}.pt')
              for r in ('cubic', 'arrhenius') for s in plan['seeds']
              for m in ('mf', 'dp', 'mf_dp', 'mf_init')}
    settings = {'particles_per_cloud': cli.particles, 'initial_bins_per_axis': cli.bins,
                'grid_anchor_particles': 10000, 'grid': 'reference min/max + 1e-6 relative padding; extend fixed lattice',
                'metric': 'exact full-3D nonperiodic Euclidean weighted histogram W2, physical coordinates',
                'not_unbinned_point_cloud_OT': True, 'numItermax': 5000000, 'regularization': 0,
                'model_seeds': plan['seeds'], 'curve_draws_per_model': 1,
                'self_D': [1.], 'model_self_repeats': 3, 'nested_control_particles': 2000,
                'same_grid_for_all_methods_seeds_sample_sizes_and_self_draws': True,
                'prior_self_A_seeds': [91001, 91011, 91021], 'prior_self_B_seeds': [92003, 92013, 92023],
                'self_extra_seed_offset': 1000000,
                'sampling': 'Preserve original 2k first-repeat indices; append disjoint samples from independent evaluation cache',
                'parameter_values': plan['eval_D'], 'no_training': True, 'checkpoint_sha256': hashes,
                'KS3D_protocol_sources': ['output/particle_count100k_refresh500_all_budgets_20261009/run_evaluation.py',
                    'output/dp_alpha_lambda_tuning_20261001/experiment.py',
                    'output/total_budget_refresh50_20261003/runtime/training/core.py']}
    if (out/'protocol.json').exists():
        assert read_json(out/'protocol.json') == settings
    write_json(out/'protocol.json', settings)
    cases = []
    for reaction in ('cubic', 'arrhenius'):
        folder = root/'runs'/reaction
        config = read_json(folder/'config.json')
        models = {s: read_models(folder, s, SimpleNamespace(**plan['training']), config['dataset_signature'])
                  for s in plan['seeds']}
        for d in [1.] + [d for d in plan['eval_D'] if d != 1.]:
            cases.append(evaluate_case(root, out, reaction, d, plan, cli, models))
            summary = summarize(out, cases)
    assert all(file_hash(root/'runs'/key.rsplit('/', 1)[0]/f'{key.rsplit("/", 1)[1]}.pt') == value
               for key, value in hashes.items())
    plot(out, summary, cli)
    source_copy = out/'high_count_ot.py'
    source_copy.write_bytes(Path(__file__).read_bytes())
    write_json(out/'verification.json', {'status': 'passed', 'cases': len(cases),
        'OT_solves': sum(len(c['rows']) for c in cases), 'all_mass_retained': True,
        'solver_converged_every_pair': True, 'frozen_checkpoints_unchanged': True,
        'old_D1_2k_prefixes_reproduced': True, 'same_grid_nested_controls': True,
        'source_sha256': file_hash(source_copy),
        'sampling_uncertainty_only': 'Conditional on existing SGIP fields; no new physics simulation or convergence claim'})
    write_json(out/'progress.json', {'stage': 'complete'})
    print(str(out), flush=True)


if __name__ == '__main__':
    main()
