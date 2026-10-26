"""Evaluate frozen cubic models at D=2.5,3,4 with the manuscript metric."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from types import SimpleNamespace
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
DLL = os.add_dll_directory(str(Path(sys.prefix)/'Library/bin')) if os.name == 'nt' and (Path(sys.prefix)/'Library/bin').is_dir() else None
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT/'deepparticle'))

import numpy as np
import torch
from non_kpp.experiment import read_models
from non_kpp.physics import Physics, ball, sample_field, simulate, write_json
from non_kpp.sweep import DiffusionProtocol, dkey, infer
from particle_pipeline.protocol import file_hash

SOURCE = ROOT/'output/non_kpp_diffusion_sweep_20261020'
OLD = ROOT/'output/other_histogram_100k_bins16_20261023'
SEEDS, METHODS = (88, 89, 90), ('mf', 'dp', 'mf_dp')
TRAIN_D = (.25, .5, 1., 2.)
OLD_D = (.25, .375, .5, .75, 1., 1.5, 2.)
OOD_D = (2.5, 3., 4.)
N, POOL, SCALE = 100000, 200000, 5.


def reference_physics(d):
    # Extend the physical box on the same 1.2-unit lattice. All neural models
    # retain their original training metadata and conditioning.
    return Physics('cubic', diffusion=d, half_width=96., bins=160)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoints():
    values = {}
    for seed in SEEDS:
        folder = SOURCE/'runs/cubic'/f'seed{seed}'
        expected = read(folder/'complete.json')['sha256']
        for method in (*METHODS, 'mf_init'):
            path = folder/f'{method}.pt'
            assert file_hash(path) == expected[method]
            values[str(path.relative_to(ROOT))] = expected[method]
    return values


def metric_module():
    # Reuse exactly the complete-cloud 16-bin lattice and certified solver
    # already used for the seven manuscript parameter values.
    spec = importlib.util.spec_from_file_location('manuscript_histogram_metric', OLD/'evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.setup()
    return module


def new_reference(d):
    folder = OUT/'datasets/cubic'/dkey(d)
    physics = reference_physics(d)
    simulate(physics, folder, 'cuda')
    simulation = read(folder/'simulation.json')
    assert simulation['physics'] == asdict(physics)
    assert all(row['discarded'] == 0 for row in simulation['records']), 'Reference boundary loss'
    path = folder/'samples.npz'
    if not path.exists():
        with np.load(folder/'field.npz', allow_pickle=False) as saved:
            density = saved['density']
        np.savez_compressed(path,
            eval_prior=ball(POOL, physics.initial_radius, 43345),
            eval_target=sample_field(density, physics, POOL, 33345),
            eval_target_independent=sample_field(density, physics, POOL, 53345))
        write_json(folder/'samples.json', {
            'physics': asdict(physics), 'pool_particles': POOL,
            'sampling_seeds': {'eval_prior': 43345, 'eval_target': 33345,
                               'eval_target_independent': 53345},
            'field_sha256': file_hash(folder/'field.npz'),
            'purpose': 'Evaluation only; no training or tuning samples are produced'})
    assert read(folder/'samples.json')['field_sha256'] == file_hash(folder/'field.npz')
    return path


def evaluate(d, path, protocol, models, metric, si, ti):
    folder = OUT/'evaluation'/dkey(d)
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder/'complete.json'
    provenance = {'samples_sha256': file_hash(path), 'checkpoint_sha256': checkpoints(),
                  'metric_sha256': file_hash(OLD/'evaluate.py')}
    if complete.exists():
        saved = read(complete)
        assert saved['provenance'] == provenance
        return saved['rows']
    with np.load(path, allow_pickle=False) as cache:
        source = cache['eval_prior'][si]
        clouds = {'reference': cache['eval_target'][ti],
                  'independent_reference': cache['eval_target_independent'][si]}
    for seed in SEEDS:
        for method in METHODS:
            clouds[f'seed{seed}_{method}'] = np.concatenate([
                infer(method, source[start:start+10000], d, protocol, models[seed], 'cuda')
                for start in range(0, N, 10000)])
    reference = clouds['reference'].astype(np.float64)
    origin, width = reference.min(0), np.ptp(reference, axis=0)/16
    assert np.all(width > 0)
    measures, archive = {}, {'origin': origin, 'width': width, 'source_indices': si, 'target_indices': ti}
    for label, points in clouds.items():
        assert points.shape == (N, 3) and np.isfinite(points).all()
        index, counts = np.unique(np.floor((points.astype(np.float64)-origin)/width).astype(np.int64),
                                  axis=0, return_counts=True)
        assert int(counts.sum()) == N
        measures[label] = metric.histogram(points, origin, width, False)
        np.testing.assert_allclose(measures[label][0], origin+(index+.5)*width, rtol=0, atol=0)
        np.testing.assert_array_equal(measures[label][1], counts.astype(np.float64)/N)
        archive[label+'_indices'], archive[label+'_counts'] = index, counts
    # An independent histogram implementation verifies the reference bins.
    index, counts = archive['reference_indices'], archive['reference_counts']
    minimum, maximum = index.min(0), index.max(0)
    edges = [origin[a]+np.arange(minimum[a], maximum[a]+2)*width[a] for a in range(3)]
    dense = np.histogramdd(reference, bins=edges)[0]
    np.testing.assert_array_equal(dense[tuple((index-minimum).T)], counts)
    assert dense.sum() == N
    np.savez_compressed(folder/'histograms.npz', **archive)
    write_json(folder/'inputs.json', {'D': d, 'role': 'extrapolation' if d > 2 else 'control',
        'particles': N, 'bins': 16, 'origin': origin.tolist(), 'width': width.tolist(),
        'conditioning': protocol.condition(d), 'provenance': provenance})
    jobs = [(f'seed{s}_{m}', s, m, 'target_error') for s in SEEDS for m in METHODS]
    jobs.append(('independent_reference', 0, 'reference', 'reference_self'))
    rows = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = {executor.submit(metric.solve, measures['reference'], measures[label], False):
                   (seed, method, kind) for label, seed, method, kind in jobs}
        for future in as_completed(pending):
            seed, method, kind = pending[future]
            result = future.result()
            rows.append({'model': 'non_kpp_cubic', 'level': d, 'seed': seed, 'method': method,
                'kind': kind, 'particles': N, 'bins': 16, 'dimension': 3, 'periodic': False,
                'normalization_scale': SCALE, **result, 'w2_normalized': result['w2']/SCALE})
            write_json(folder/'partial.json', rows)
            print(f'D={d:g} {method} seed={seed}: normalized W2={result["w2"]/SCALE:.8f}', flush=True)
    rows.sort(key=lambda row: (row['kind'], row['seed'], row['method']))
    write_json(complete, {'status': 'complete', 'provenance': provenance, 'rows': rows,
        'histograms_sha256': file_hash(folder/'histograms.npz'), 'all_particles_retained': True,
        'independent_histogram_check': True})
    return rows


def main():
    metric = metric_module()
    hashes = checkpoints()
    source_plan = read(SOURCE/'plan.json')
    assert source_plan['train_D'] == list(TRAIN_D)
    assert source_plan['seeds'] == list(SEEDS)
    assert source_plan['n_eval_per_D'] == POOL
    plan = {'material_passport': {'origin_skill': 'academic-research-suite',
        'mode': 'run + validate', 'date': '2026-10-23'},
        'train_D': list(TRAIN_D), 'interpolation_D': [.375, .75, 1.5],
        'extrapolation_D': list(OOD_D), 'training_seeds': list(SEEDS),
        'training_updates': {'mf': 2000, 'dp': 2000, 'mf_dp': [1000, 1000]},
        'frozen_checkpoint_sha256': hashes, 'new_training': False, 'test_based_tuning': False,
        'physics': {str(d): asdict(reference_physics(d)) for d in OOD_D},
        'reference_domain_extension': {
            'original_half_width': 60., 'new_half_width': 96., 'spatial_step_unchanged': 1.2,
            'reason': 'Original D=3 reference lost particles at the numerical boundary; expand all OOD references before reporting.',
            'initial_attempt': 'initial_box60/', 'all_other_physical_and_time_settings_unchanged': True},
        'reference_samples': N, 'generated_samples': N, 'bins_per_axis': 16,
        'grid': 'Full 100k reference min/max, fixed lattice extended without clipping',
        'normalization': 'physical W2 / 5', 'uncertainty': 'mean and observed min-max over 3 model seeds',
        'metric_source_sha256': file_hash(OLD/'evaluate.py'),
        'old_results_sha256': file_hash(OLD/'metrics_per_seed.csv')}
    if (OUT/'plan.json').exists():
        assert read(OUT/'plan.json') == plan
    else:
        write_json(OUT/'plan.json', plan)
    args = SimpleNamespace(**source_plan['training'])
    cfg = read(SOURCE/'runs/cubic/config.json')
    protocol = DiffusionProtocol('cubic', source_plan['particles'])
    assert protocol.metadata() == cfg['protocol']
    models = {s: read_models(SOURCE/'runs/cubic', s, args, cfg['dataset_signature']) for s in SEEDS}
    for seed in SEEDS:
        for method in (*METHODS, 'mf_init'):
            payload = torch.load(SOURCE/'runs/cubic'/f'seed{seed}'/f'{method}.pt',
                                 map_location='cpu', weights_only=True)
            assert payload['protocol'] == cfg['protocol']
            assert payload['step'] == (1000 if method in ('mf_init', 'mf_dp') else 2000)
    si = metric.nested_indices(POOL, N, 7000, 700000)
    ti = metric.nested_indices(POOL, N, 8000, 800000)
    old_rows = []
    for d in OLD_D:
        old_rows.extend(read(OLD/'results/non_kpp_cubic'/('level_'+format(d, '.6g').replace('.', 'p'))/'complete.json')['rows'])
    assert len(old_rows) == 70
    control = evaluate(2., SOURCE/'datasets/cubic/D_2/samples.npz', protocol, models, metric, si, ti)
    baseline = {(r['kind'], r['seed'], r['method']): r for r in old_rows if r['level'] == 2.}
    control_error = max(abs(r['w2']-baseline[r['kind'], r['seed'], r['method']]['w2']) for r in control)
    assert control_error < 1e-10, control_error
    new_rows = []
    for d in OOD_D:
        write_json(OUT/'progress.json', {'stage': 'reference simulation', 'D': d})
        path = new_reference(d)
        write_json(OUT/'progress.json', {'stage': 'evaluation', 'D': d})
        new_rows.extend(evaluate(d, path, protocol, models, metric, si, ti))
    rows = sorted(old_rows+new_rows, key=lambda r: (r['level'], r['kind'], r['seed'], r['method']))
    csv_write(OUT/'metrics_per_seed.csv', rows)
    summary = []
    for d in (*OLD_D, *OOD_D):
        for method in (*METHODS, 'reference'):
            group = [r for r in rows if r['level'] == d and r['method'] == method]
            assert len(group) == (1 if method == 'reference' else 3)
            values = [r['w2_normalized'] for r in group]
            summary.append({'D': d, 'role': 'training' if d in TRAIN_D else 'extrapolation' if d > 2 else 'interpolation',
                'method': method, 'mean': statistics.mean(values), 'minimum': min(values), 'maximum': max(values),
                'sd': statistics.stdev(values) if len(values)>1 else 0., 'runs': len(values),
                'particles_per_distribution': N, 'bins_per_axis': 16, 'normalization_scale': SCALE})
    csv_write(OUT/'summary.csv', summary)
    write_json(OUT/'summary.json', summary)
    assert checkpoints() == hashes
    assert file_hash(OLD/'metrics_per_seed.csv') == plan['old_results_sha256']
    checks = {'status': 'passed', 'planned_extrapolation_D': list(OOD_D), 'new_target_comparisons': 27,
        'new_reference_self_comparisons': 3, 'control_D2_max_absolute_W2_difference': control_error,
        'original_seven_levels_reused_unchanged': True, 'frozen_checkpoints_unchanged': True,
        'new_training': False, 'all_reference_simulations_zero_boundary_loss': True,
        'all_evaluation_particles_retained': True, 'independent_histogram_counts_verified': True,
        'max_marginal_residual': max(r['marginal_residual'] for r in new_rows),
        'max_primal_dual_gap': max(r['primal_dual_gap'] for r in new_rows),
        'max_dual_violation': max(r['dual_violation'] for r in new_rows),
        'solver_result_codes': sorted({r['solver_result_code'] for r in new_rows}),
        'limitations': 'Conditional on one numerical reference per D; reference uncertainty and PDE convergence are not estimated.'}
    write_json(OUT/'verification.json', checks)
    write_json(OUT/'progress.json', {'stage': 'complete'})
    print(json.dumps({'status': 'passed', 'new_results': [r for r in summary if r['D'] > 2]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
