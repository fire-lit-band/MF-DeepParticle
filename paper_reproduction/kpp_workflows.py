"""Generate, score and evolve KPP samples with the published numerical kernels.

All outputs are written apart from package inputs. Saved physical data are read-only.
Model runs may contain newly trained weights; historical weight equality is not
required, but completion, protocol, architecture and dataset identity are.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from . import ROOT
from .experiments import _training_kernel, _kpp_evaluation_kernel, _load, _models
from .portability import protocols_match, resolve_recorded_path, verified_source_identity
from .locations import fresh_directory

MODELS = ('kpp2d', 'kpp3d')
SEEDS = (88, 89, 90)
METHODS = ('mf', 'dp', 'mf_dp')
WEIGHTS = ('mf', 'mf_init', 'dp', 'mf_dp')
LEVELS = tuple(2 + .25 * i for i in range(11))
PARTICLES = 100000
SELECTED = ROOT / 'output/dp_alpha01_refinement_alpha10_20261017'
REFERENCE = ROOT / 'output/other_histogram_100k_bins16_20261023'


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _destination(path, *inputs):
    path = fresh_directory(path)
    for source in inputs:
        if source is not None and path.is_relative_to(Path(source).expanduser().resolve()):
            raise ValueError('Output must not be inside an input model or generated-data directory')
    return path


def _relative(path, base):
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()
    except ValueError:  # A separately supplied input can be on another drive.
        return str(Path(path).resolve())


def _inside(base, value):
    base = Path(base).resolve(strict=True)
    path = (base / str(value).replace('\\', '/')).resolve(strict=True)
    if not path.is_relative_to(base):
        raise ValueError('Manifest input escaped its output directory')
    return path


def _model_run(path):
    if path is None:
        return None
    supplied = Path(path).expanduser().resolve(strict=True)
    candidates = [p for p in (supplied, supplied / 'ks_kpp')
                  if (p / 'training_replay').is_dir() and (p / 'training_replay_verification.json').is_file()]
    if len(candidates) != 1:
        raise ValueError('model_run must be a completed train output or its ks_kpp directory')
    run = candidates[0]
    completion = _json(run / 'training_replay_verification.json')
    if completion.get('status') != 'passed' or not isinstance(completion.get('checks'), list):
        raise ValueError('Model training has no successful completion record')
    if not (run / 'replay_request.json').is_file():
        raise ValueError('Model training request and dataset identities are missing')
    return run


def _checkpoint_inputs(measure, name, protocol, signature, model_run=None, *, device='cpu'):
    """Load selected or completed-run weights without requiring historical equality."""
    torch = measure.torch
    run = _model_run(model_run)
    selected = _json(SELECTED / 'checkpoint_manifest.json')[name] if run is None else None
    if run is not None:
        request = _json(run / 'replay_request.json')
        if request.get('datasets', {}).get(name, {}).get('signature') != signature:
            raise ValueError(f'Training dataset signature differs for {name}')
        checks = _json(run / 'training_replay_verification.json')['checks']
        wanted = {(name, seed, method) for seed in SEEDS for method in WEIGHTS}
        actual = [(row.get('model'), row.get('seed'), row.get('method')) for row in checks if row.get('model') == name]
        if len(actual) != len(wanted) or set(actual) != wanted:
            raise ValueError(f'Incomplete checkpoint completion records for {name}')
        saved_manifest = _json(run / 'checkpoint_manifest.json') if (run / 'checkpoint_manifest.json').is_file() else None
    loaded, records = {}, []
    for seed in SEEDS:
        loaded[seed] = {}
        for method in WEIGHTS:
            step = 500 if method in ('mf_init', 'mf_dp') else 1000
            if run is None:
                record = selected[str(seed)][method]
                path = resolve_recorded_path(record['file'], root=ROOT)
                if _hash(path) != record['sha256']:
                    raise ValueError(f'Selected checkpoint hash changed: {path}')
            else:
                path = _inside(run, f'training_replay/{name}/seed{seed}/{method}.pt')
                if saved_manifest is not None:
                    record = saved_manifest[name][str(seed)][method]
                    if _inside(run, record['file']) != path or _hash(path) != record['sha256'] or record['step'] != step:
                        raise ValueError(f'Training checkpoint manifest mismatch: {path}')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if (not protocols_match(payload['protocol'], protocol.metadata(), root=ROOT)
                    or payload['dataset_signature'] != signature or payload['step'] != step):
                raise ValueError(f'Checkpoint protocol, signature or update count differs: {path}')
            architecture = payload['architecture']
            if (architecture.get('width') != 64 or architecture.get('hidden_layers') != 3
                    or architecture.get('condition_mode') != 'protocol'):
                raise ValueError(f'Checkpoint architecture differs: {path}')
            if method in ('mf', 'mf_init'):
                if architecture.get('kind') != 'mf':
                    raise ValueError('Expected a MeanFlow checkpoint')
                model = measure.e.MeanFlow(protocol.dimension, 64, 'protocol', 3)
            else:
                alpha = .1 if method == 'dp' else 10.
                if (architecture.get('kind') != 'position_condition_dp'
                        or architecture.get('alpha') != alpha
                        or architecture.get('input_features') != protocol.dimension + 1
                        or architecture.get('activation') != 'SELU'):
                    raise ValueError(f'Corrector architecture differs: {path}')
                model = measure.r.make_model('dplus1', protocol, alpha)
            model.load_state_dict(payload['model'], strict=True)
            if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
                raise ValueError(f'Nonfinite checkpoint tensors: {path}')
            loaded[seed][method] = model.to(device).eval().requires_grad_(False)
            records.append({'model': name, 'seed': seed, 'method': method, 'step': step,
                            'file': str(path), 'sha256': _hash(path)})
    return loaded, records


@contextmanager
def _kernels(*, cuda=False):
    with _training_kernel() as measure:
        with _kpp_evaluation_kernel() as metric:
            if cuda:
                if not measure.torch.cuda.is_available():
                    raise RuntimeError('This KPP workflow requires CUDA')
                measure.e.configure_device('cuda')
            metric.setup()
            yield measure, metric


def _file_record(path, base, **fields):
    return {'path': Path(path).relative_to(base).as_posix(), 'sha256': _hash(path), **fields}


def _data_run(path, names):
    if path is None:
        return None
    from .kpp_data import resolve
    root, _ = resolve(path, names)
    return root


def _dataset(measure, name, data_run):
    if data_run is None:
        return measure.e.dataset(name)
    from .kpp_data import load_dataset
    folder, protocol, manifest, splits = load_dataset(name, data_run)
    return folder, None, (protocol, manifest, splits)


def generate(models, destination, *, model_run=None, data_run=None) -> dict:
    """Generate all 100k distributions plus the 20k qualitative panels, without OT."""
    names = _models(models, MODELS)
    model_run = _model_run(model_run)
    data_run = _data_run(data_run, names)
    if data_run is not None and model_run is None:
        raise ValueError('New physical data require a completed model_run trained on those data')
    destination = _destination(destination, model_run, data_run)
    manifest = {'schema': 1, 'type': 'kpp_generation', 'status': 'running', 'models': names,
                'levels': list(LEVELS), 'seeds': list(SEEDS), 'methods': list(METHODS),
                'particles': PARTICLES, 'density_particles': 20000, 'source_seed': 411002,
                'model_run': _relative(model_run, destination) if model_run else None,
                'data_run': _relative(data_run, destination) if data_run else None,
                'checkpoints': [], 'datasets': {}, 'generated_files': {}, 'density_files': {}}
    with _kernels(cuda=True) as (measure, metric):
        destination.mkdir(parents=True, exist_ok=True)
        _write(destination / 'manifest.json', manifest)
        for name in names:
            _, _, (protocol, dataset, splits) = _dataset(measure, name, data_run)
            networks, records = _checkpoint_inputs(measure, name, protocol, dataset['signature'], model_run, device='cuda')
            manifest['checkpoints'].extend(records)
            manifest['datasets'][name] = {'signature': dataset['signature'], 'protocol': protocol.metadata()}
            source = metric.prior(protocol, PARTICLES, 411002)
            folder = destination / 'generated' / name
            folder.mkdir(parents=True, exist_ok=True)
            metric.write_npz(folder / 'source.npz', meta={'seed': 411002, 'distribution': 'Normal(0,pi)',
                             'particles': PARTICLES, 'model': name}, source=source)
            manifest.setdefault('source_files', {})[name] = _file_record(folder / 'source.npz', destination)
            manifest['generated_files'][name] = []
            manifest['density_files'][name] = []
            for index, level in enumerate(LEVELS):
                clouds = metric.predict_kpp(source, level, protocol, networks)
                if any(cloud.shape != (PARTICLES, protocol.dimension) or not metric.np.isfinite(cloud).all() for cloud in clouds.values()):
                    raise ValueError('Invalid generated KPP cloud')
                path = folder / f'level_{index:02d}.npz'
                metric.write_npz(path, meta={'model': name, 'level': level, 'particles': PARTICLES,
                                 'source_seed': 411002, 'dataset_signature': dataset['signature'],
                                 'protocol': protocol.metadata()}, **clouds)
                manifest['generated_files'][name].append(_file_record(path, destination, index=index, level=level))
                print(f'{name} level={level:g}: generated 9 x {PARTICLES} particles', flush=True)
            for index in ((7, 8, 10) if name == 'kpp2d' else (10,)):
                case = splits['eval'][index]
                clouds = metric.predict_kpp(case['prior'], case['level'], protocol, networks)
                path = destination / 'benchmarks' / name / 'evaluation' / f'particles_{index:02d}.npz'
                metric.write_npz(path, meta={'model': name, 'level': case['level'], 'particles': 20000,
                                 'dataset_signature': dataset['signature'], 'protocol': protocol.metadata()},
                                 reference=case['target'], **clouds)
                manifest['density_files'][name].append(_file_record(path, destination, index=index, level=case['level']))
            _write(destination / 'manifest.json', manifest)
            del networks, clouds
            measure.torch.cuda.empty_cache()
    manifest['status'] = 'complete'
    _write(destination / 'manifest.json', manifest)
    return {'status': 'completed', 'models': names, 'destination': str(destination),
            'manifest': str(destination / 'manifest.json'), 'generated_clouds': len(names) * 99}


def _generation(path, names):
    path = Path(path).expanduser().resolve(strict=True)
    manifest = _json(path / 'manifest.json')
    if (manifest.get('type') != 'kpp_generation' or manifest.get('status') != 'complete'
            or manifest.get('particles') != PARTICLES or manifest.get('levels') != list(LEVELS)
            or manifest.get('seeds') != list(SEEDS) or manifest.get('methods') != list(METHODS)
            or not set(names).issubset(manifest.get('models', []))):
        raise ValueError('Generated data are incomplete or use a different KPP protocol')
    return path, manifest


def _reference_run(path, names):
    """Resolve either retained references or a completed public regeneration."""
    if path is None:
        return REFERENCE, None
    root = Path(path).expanduser().resolve(strict=True)
    plan = _json(root / 'reference_plan.json')
    manifest = _json(root / 'manifest.json')
    if (plan.get('type') != 'kpp_references' or plan.get('status') != 'complete'
            or manifest.get('type') != 'kpp_references' or manifest.get('status') != 'complete'
            or not set(names).issubset(plan.get('models', []))
            or plan.get('levels') != list(LEVELS) or plan.get('particles_per_reference') != PARTICLES
            or plan.get('generations') != 4096 or plan.get('physical_seed') != 10999
            or plan.get('from_uniform_start') is not True):
        raise ValueError('Reference run must be complete with all11 levels,100k particles and T4096')
    if manifest != plan:
        raise ValueError('Reference completion manifest and reference plan disagree')
    for name in names:
        relative = plan.get('reference_paths', {}).get(name)
        if not relative:
            raise ValueError(f'Reference manifest is missing for {name}')
        location = _inside(root, relative)
        if not location.is_relative_to(root / 'references' / name):
            raise ValueError('Reference manifest escaped its model directory')
    return root, plan


def _reference_records(root, plan, name, protocol):
    if plan is None:
        location = root / 'references' / name / 'manifest.json'
    else:
        location = _inside(root, plan['reference_paths'][name])
        # Early public runs did not duplicate the protocol in the top-level
        # plan; every NPZ is still checked against the full numerical protocol.
        if 'protocols' in plan and not protocols_match(plan['protocols'][name], protocol.metadata(), root=ROOT):
            raise ValueError('Reference plan physical protocol differs')
    manifest = _json(location)
    settings = manifest.get('plan', {})
    if (manifest.get('status') != 'complete' or settings.get('particles_per_reference') != PARTICLES
            or settings.get('generations') != 4096 or settings.get('physical_seed') != 10999
            or settings.get('levels') != list(LEVELS)):
        raise ValueError('Reference model manifest is incomplete or has different simulation settings')
    records = manifest.get('outputs', [])
    if len(records) != 11 or {row.get('level') for row in records} != set(LEVELS):
        raise ValueError('Reference condition records are incomplete or duplicated')
    resolved = {}
    for row in records:
        level = row['level']
        if (row.get('dimension') != protocol.dimension or row.get('seed') != 10999
                or row.get('kappa') != protocol.parameter(level)):
            raise ValueError('Reference record model, physical seed or diffusivity differs')
        if plan is None:
            path = resolve_recorded_path(row['file'], root=ROOT, allowed_root=root / 'references' / name)
        else:
            path = _inside(root, row['file'])
            if not path.is_relative_to(root / 'references' / name):
                raise ValueError('Reference cloud escaped its model directory')
        resolved[level] = (path, row)
    return location, resolved


def _reference_cloud(metric, path, record, name, level, protocol):
    if _hash(path) != record['sha256']:
        raise ValueError(f'Physical reference hash changed: {path}')
    arrays, metadata = metric.read_npz(path)
    expected = {'model': name, 'level': level, 'seed': 10999, 'particles': PARTICLES,
                'generations': 4096, 'selection_count': 4096}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Reference metadata differs from the requested KPP condition')
    numerical = metadata.get('protocol', {})
    expected_numerical = {
        'dimension': protocol.dimension, 'particles': PARTICLES, 'generations': 4096,
        'dt': 2. ** -8, 'steps_per_period': 256, 'lambda': 2.,
        'theta': 1. if protocol.dimension == 2 else None,
        'resample_every': 256, 'period': 1., 'phase': 0,
        'drift': '2*kappa*lambda*e1 + velocity',
        'potential': 'kappa*lambda^2 + lambda*v1' if protocol.dimension == 2 else 'kappa*lambda^2 - lambda*v1',
        'reaction_in_weights': 0., 'eigenvalue_offset': '+kappa after growth/time',
        'weight_position': 'post-propagation',
        'weight_time': 'j*dt (forward), same substep as propagation',
        'resampler': 'multinomial inverse CDF once per period',
        'initial_rng': 'numpy.default_rng(seed), uniform(0,2*pi,(N,d))',
        'evolution_rng': 'per-seed CUDA Philox, manual_seed(seed+7919), shared across kappa values',
        'dtype': 'float64'}
    if any(numerical.get(key) != value for key, value in expected_numerical.items()):
        raise ValueError('Reference numerical operator, resampling, RNG or discretization differs')
    identity = verified_source_identity(numerical['source_file'], root=ROOT)
    if (numerical.get('source_sha256') != identity['source_sha256']
            or metadata.get('source_hash') != identity['source_sha256']
            or identity['source_sha256'] != protocol.metadata()['source_sha256']):
        raise ValueError('Reference physical source identity differs')
    for key in ('initial', 'target'):
        cloud = arrays.get(key)
        if (cloud is None or cloud.shape != (PARTICLES, protocol.dimension)
                or not metric.np.isfinite(cloud).all() or (cloud < 0).any() or (cloud > 2 * math.pi).any()):
            raise ValueError(f'Invalid 100k periodic reference {key} cloud')
    return arrays, metadata


def evaluate(models, destination, *, model_run=None, generated=None, reference_run=None, data_run=None) -> dict:
    """Score new100k clouds against retained or fully regenerated FK references."""
    names = _models(models, MODELS)
    model_run = _model_run(model_run)
    data_run = _data_run(data_run, names)
    if data_run is not None and reference_run is None:
        reference_run = data_run / 'reference_run'
    reference_root, reference_plan = _reference_run(reference_run, names)
    destination = _destination(destination, model_run, generated, reference_run, data_run)
    destination.mkdir(parents=True, exist_ok=True)
    if generated is None:
        generated = destination / 'generated'
        generate(names, generated, model_run=model_run, data_run=data_run)
    generated, generation = _generation(generated, names)
    recorded_data = generation.get('data_run')
    if ((data_run is None) != (recorded_data is None)
            or (data_run is not None and (generated / recorded_data).resolve(strict=True) != data_run)):
        raise ValueError('Generated clouds do not belong to the requested physical data_run')
    if data_run is not None and model_run is None:
        recorded_model = generation.get('model_run')
        if not recorded_model:
            raise ValueError('Generated clouds do not identify a completed new model run')
        model_run = _model_run(generated / recorded_model)
        if destination.is_relative_to(model_run):
            raise ValueError('Evaluation output must be outside the input model run')
    manifest = {'schema': 1, 'type': 'kpp_evaluation', 'status': 'running', 'models': names,
                'generated': _relative(generated, destination), 'particles': PARTICLES, 'bins': 16,
                'data_run': _relative(data_run, destination) if data_run else None,
                'model_run': _relative(model_run, destination) if model_run else None,
                'seeds': list(SEEDS), 'methods': list(METHODS), 'levels': list(LEVELS),
                'reference_source': 'regenerated' if reference_plan is not None else 'retained',
                'reference_run': _relative(reference_root, destination),
                'reference_plan_sha256': _hash(reference_root / 'reference_plan.json') if reference_plan is not None else None,
                'reference_manifests': {},
                'checkpoints': [r for r in generation['checkpoints'] if r['model'] in names]}
    _write(destination / 'manifest.json', manifest)
    rows = []
    with _kernels() as (measure, metric):
        old_out = metric.OUT
        metric.OUT = destination
        try:
            for name in names:
                _, _, (protocol, dataset, _) = _dataset(measure, name, data_run)
                if generation['datasets'][name]['signature'] != dataset['signature']:
                    raise ValueError('Generated data use a different physical dataset')
                if not protocols_match(generation['datasets'][name]['protocol'], protocol.metadata(), root=ROOT):
                    raise ValueError('Generated data physical protocol differs')
                if model_run is not None:
                    networks, records = _checkpoint_inputs(measure, name, protocol, dataset['signature'], model_run)
                    expected = {(r['seed'], r['method']): r['sha256'] for r in records}
                    actual = {(r['seed'], r['method']): r['sha256'] for r in generation['checkpoints'] if r['model'] == name}
                    if actual != expected:
                        raise ValueError('Generated data do not belong to the requested model run')
                    del networks
                reference_manifest, references_by_level = _reference_records(reference_root, reference_plan, name, protocol)
                manifest['reference_manifests'][name] = {
                    'path': _relative(reference_manifest, destination), 'sha256': _hash(reference_manifest)}
                records = generation['generated_files'][name]
                if len(records) != 11 or {(r['index'], r['level']) for r in records} != set(enumerate(LEVELS)):
                    raise ValueError('Generated condition records are incomplete')
                for index, level in enumerate(LEVELS):
                    record = next(r for r in records if r['index'] == index)
                    path = _inside(generated, record['path'])
                    if _hash(path) != record['sha256']:
                        raise ValueError(f'Generated cloud changed: {path}')
                    clouds, metadata = metric.read_npz(path)
                    if (metadata.get('model') != name or metadata.get('level') != level
                            or metadata.get('dataset_signature') != dataset['signature']):
                        raise ValueError('Generated cloud metadata differs')
                    if set(clouds) != {f'seed{s}_{m}' for s in SEEDS for m in METHODS}:
                        raise ValueError('Generated cloud methods or seeds are incomplete')
                    ref_path, ref_record = references_by_level[level]
                    arrays, reference_meta = _reference_cloud(metric, ref_path, ref_record, name, level, protocol)
                    clouds['reference'] = arrays['target']
                    rows.extend(metric.evaluate_clouds(name, level, clouds, True, 1., {
                        'generated': _relative(path, destination), 'generated_sha256': record['sha256'],
                        'reference': _relative(ref_path, destination), 'reference_sha256': ref_record['sha256'],
                        'reference_source': manifest['reference_source']}))
        finally:
            metric.OUT = old_out
        expected = {(n, level, seed, method) for n in names for level in LEVELS for seed in SEEDS for method in METHODS}
        if (len(rows) != len(expected) or {(r['model'], r['level'], r['seed'], r['method']) for r in rows} != expected
                or any(r['solver_result_code'] != 1 or r['marginal_residual'] >= 1e-9
                       or r['primal_dual_gap'] >= 1e-8 or r['dual_violation'] >= 1e-8 for r in rows)):
            raise RuntimeError('KPP evaluation failed completeness or exact-OT certification')
        rows.sort(key=lambda r: (r['model'], r['level'], r['seed'], r['method']))
        metric.csv_write(destination / 'metrics_per_seed.csv', rows)
        summary = []
        for name in names:
            for level in LEVELS:
                for method in METHODS:
                    values = [r['w2'] for r in rows if (r['model'], r['level'], r['method']) == (name, level, method)]
                    summary.append({'model': name, 'level': level, 'method': method, 'kind': 'target_error',
                                    'w2_mean': statistics.mean(values), 'w2_sd': statistics.stdev(values),
                                    'w2_min': min(values), 'w2_max': max(values), 'seeds': 3,
                                    'particles': PARTICLES, 'bins': 16, 'normalization_scale': 1.})
        metric.csv_write(destination / 'summary.csv', summary)
    manifest.update(status='complete', rows=len(rows), summary='summary.csv', metrics='metrics_per_seed.csv',
                    summary_sha256=_hash(destination / 'summary.csv'),
                    metrics_sha256=_hash(destination / 'metrics_per_seed.csv'))
    _write(destination / 'verification.json', {'status': 'passed', 'rows': len(rows),
           'max_marginal_residual': max(r['marginal_residual'] for r in rows),
           'max_primal_dual_gap': max(r['primal_dual_gap'] for r in rows),
           'max_dual_violation': max(r['dual_violation'] for r in rows),
           'comparison_to_historical_scores_required': False})
    _write(destination / 'manifest.json', manifest)
    return {'status': 'completed', 'models': names, 'destination': str(destination),
            'rows': len(rows), 'manifest': str(destination / 'manifest.json')}


def warm_start(models, destination, *, model_run=None, data_run=None) -> dict:
    """Generate new 20k initial clouds and run all30 paired FK starts to128 periods."""
    names = _models(models, MODELS)
    model_run = _model_run(model_run)
    data_run = _data_run(data_run, names)
    if data_run is not None and model_run is None:
        raise ValueError('New physical data require a completed model_run trained on those data')
    destination = _destination(destination, model_run, data_run)
    manifest = {'schema': 1, 'type': 'kpp_warm_start', 'status': 'running', 'models': names,
                'model_run': _relative(model_run, destination) if model_run else None,
                'data_run': _relative(data_run, destination) if data_run else None,
                'particles': 20000, 'generations': 128, 'level': 4.5,
                'model_seeds': list(SEEDS), 'physical_seeds': [1234, 1235, 1236],
                'checkpoints': [], 'trace_paths': {}, 'trace_sha256': {}}
    with _kernels(cuda=True) as (measure, metric):
        previous = sys.modules.get('experiment')
        sys.modules['experiment'] = measure.e.uploaded
        try:
            helper = _load('_public_kpp_warm_helper', ROOT / 'output/kpp_original_training_total1k_20261014/warm_starts.py')
        finally:
            if previous is None:
                sys.modules.pop('experiment', None)
            else:
                sys.modules['experiment'] = previous
        destination.mkdir(parents=True, exist_ok=True)
        _write(destination / 'manifest.json', manifest)
        for name in names:
            _, _, (protocol, dataset, splits) = _dataset(measure, name, data_run)
            networks, records = _checkpoint_inputs(measure, name, protocol, dataset['signature'], model_run, device='cuda')
            manifest['checkpoints'].extend(records)
            case = next(c for c in splits['eval'] if c['level'] == 4.5)
            clouds = metric.predict_kpp(case['prior'], 4.5, protocol, networks)
            folder = destination / 'benchmarks' / name / 'warm_starts'
            metric.write_npz(folder / 'initial.npz', meta={'model': name, 'level': 4.5,
                             'dataset_signature': dataset['signature'], 'protocol': protocol.metadata(),
                             'particles': 20000}, source=case['prior'], **clouds)
            del networks
            measure.torch.cuda.empty_cache()
            jobs = []
            for physical_seed in (1234, 1235, 1236):
                entries = [('cold', None, None)] + [(method, seed, clouds[f'seed{seed}_{method}'])
                           for seed in SEEDS for method in METHODS]
                for method, seed, cloud in entries:
                    jobs.append({'dimension': protocol.dimension, 'level': 4.5, 'kappa': protocol.parameter(4.5),
                                 'seed': physical_seed, 'particles': 20000, 'method': method,
                                 'model_seed': seed, 'cloud': cloud})
            gpu = measure.e.uploaded.load_accelerator()
            started = time.perf_counter()
            simulation = helper.make_simulation(gpu, protocol.dimension, jobs)
            identities = [{'method': job['method'], 'model_seed': job['model_seed'], 'fk_seed': job['seed'],
                           'initial_sha256': hashlib.sha256(initial.tobytes()).hexdigest()}
                          for job, initial in zip(jobs, simulation.initial)]
            rows = []
            for generation in range(1, 129):
                simulation.graph.replay()
                if generation & (generation - 1) == 0:
                    measure.torch.cuda.synchronize()
                    values = ((simulation.kappa + simulation.growth / generation) / simulation.lam).cpu().numpy()
                    if not metric.np.isfinite(values).all():
                        raise FloatingPointError('Nonfinite FK growth-rate estimate')
                    for job, value in zip(jobs, values):
                        reference = float(case['meta']['c_hat'])
                        rows.append({'model': name, 'level': 4.5, 'method': job['method'],
                                     'model_seed': job['model_seed'], 'fk_seed': job['seed'],
                                     'generation': generation, 'c_hat': float(value), 'reference_c': reference,
                                     'absolute_error': abs(float(value) - reference)})
                    measure.e.CSV(folder / 'trace.csv', rows)
                if generation == 1 or generation % 32 == 0:
                    print(f'{name}: FK {generation}/128,30 paired ensembles', flush=True)
            if len(rows) != 240:
                raise RuntimeError('Incomplete FK trace')
            _write(folder / 'verification.json', {'status': 'passed', 'trace_rows': len(rows),
                   'initial_clouds': identities, 'initial_file_sha256': _hash(folder / 'initial.npz'),
                   'elapsed_seconds': time.perf_counter() - started,
                   'kernel_sha256': _hash(Path(gpu.__file__)), 'helper_sha256': _hash(Path(helper.__file__)),
                   'model_seeds': list(SEEDS), 'physical_seeds': [1234, 1235, 1236],
                   'reference': 'Independent fixed20000-particle T4096 estimate; uncertainty not estimated',
                   'batch_order': 'physical-seed major; cold then model-seed major, methods mf,dp,mf_dp'})
            manifest['trace_paths'][name] = (folder / 'trace.csv').relative_to(destination).as_posix()
            manifest['trace_sha256'][name] = _hash(folder / 'trace.csv')
            _write(destination / 'manifest.json', manifest)
            del simulation, clouds
            measure.torch.cuda.empty_cache()
    manifest['status'] = 'complete'
    _write(destination / 'manifest.json', manifest)
    return {'status': 'completed', 'models': names, 'destination': str(destination),
            'trace_rows': len(names) * 240, 'manifest': str(destination / 'manifest.json')}


def references(models, destination, *, levels=None, particles=100000) -> dict:
    """Regenerate independent FK references from uniform starts to T4096.

    The paper workload uses every level and100000particles, requiring millions
    of integration steps across the batched ensembles. Subsets or other positive
    counts are explicitly recorded and do not constitute the full paper workload.
    """
    names = _models(models, MODELS)
    selected_levels = list(LEVELS) if levels is None else [float(value) for value in levels]
    if (not selected_levels or len(set(selected_levels)) != len(selected_levels)
            or any(value not in LEVELS for value in selected_levels)):
        raise ValueError('Choose distinct KPP levels from2,2.25,...,4.5')
    selected_levels.sort()
    if isinstance(particles, bool) or not isinstance(particles, int) or particles < 1:
        raise ValueError('particles must be a positive integer')
    destination = _destination(destination)
    with _kernels(cuda=True) as (measure, metric):
        gpu = measure.e.uploaded.load_accelerator()
        validation_path = ROOT / 'output/kpp_original_sampling_20261014/gpu_validation.json'
        previous = _json(validation_path)
        if previous.get('status') != 'passed':
            raise ValueError('Prior numerical kernel validation did not pass')
        for filename, digest in previous['source_hashes'].items():
            if verified_source_identity(gpu.ARCHIVE / filename, root=ROOT)['source_sha256'] != digest:
                raise ValueError('Original source identity differs from prior validation')
        gpu.LEVELS, gpu.SEEDS = tuple(selected_levels), (10999,)
        destination.mkdir(parents=True, exist_ok=True)
        plan = {'schema': 1, 'type': 'kpp_references', 'status': 'running', 'models': names,
                'levels': selected_levels, 'particles_per_reference': particles, 'generations': 4096,
                'physical_seed': 10999, 'from_uniform_start': True, 'resume_from_old_inputs': False,
                'full_paper_workload': selected_levels == list(LEVELS) and particles == PARTICLES,
                'accelerator_sha256': _hash(Path(gpu.__file__)),
                'prior_validation_sha256': _hash(validation_path), 'reference_paths': {},
                'protocols': {name: measure.e.uploaded.protocol(name).metadata() for name in names}}
        _write(destination / 'reference_plan.json', plan)
        _write(destination / 'manifest.json', plan)
        for name in names:
            dimension = int(name[-2])
            gpu.OUT = destination / 'references' / name
            gpu.OUT.mkdir(parents=True, exist_ok=True)
            print(f'{name}: generating {len(selected_levels)} references from scratch;'
                  f' {particles} particles each,4096 periods x256 substeps', flush=True)
            validation = gpu.validate_dimension(dimension)
            _write(gpu.OUT / 'validation.json', validation)
            started = time.perf_counter()
            outputs = gpu.run_dimension(dimension, particles, 4096, validation)
            if len(outputs) != len(selected_levels):
                raise RuntimeError('Incomplete physical reference generation')
            for row in outputs:
                row['file'] = Path(row['file']).relative_to(destination).as_posix()
            reference_manifest = {'status': 'complete', 'plan': plan, 'validation': validation,
                                  'outputs': outputs, 'elapsed_this_invocation': time.perf_counter() - started}
            _write(gpu.OUT / 'manifest.json', reference_manifest)
            plan['reference_paths'][name] = (gpu.OUT / 'manifest.json').relative_to(destination).as_posix()
        plan['status'] = 'complete'
        _write(destination / 'manifest.json', plan)
        _write(destination / 'reference_plan.json', plan)
    return {'status': 'completed', 'models': names, 'destination': str(destination),
            'reference_clouds': len(names) * len(selected_levels), 'manifest': str(destination / 'manifest.json')}
