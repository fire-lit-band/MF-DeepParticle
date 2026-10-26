"""Generate all nine cubic physical datasets from uniform initial particles.

No retained NPZ or checkpoint is read. The numerical kernels, random seeds,
five-million-particle budget, spatial boxes and sampling protocols are fixed.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

from . import ROOT
from .locations import fresh_directory

TRAIN_D = (.25, .5, 1., 2.)
LEVELS = (.25, .375, .5, .75, 1., 1.5, 2., 2.5, 3.)
SAMPLING_SEEDS = {'train_target': 12345, 'train_prior': 22345,
                  'eval_target': 33345, 'eval_prior': 43345, 'eval_target_independent': 53345}


def _numerics():
    package = str(ROOT / 'deepparticle')
    if package not in sys.path:
        sys.path.insert(0, package)
    import numpy as np
    import torch
    from non_kpp.physics import Physics, ball, sample_field, simulate, write_json
    from non_kpp.experiment import prepare
    from non_kpp.sweep import DiffusionProtocol
    from particle_pipeline.protocol import fingerprint
    return np, torch, Physics, ball, sample_field, simulate, write_json, prepare, DiffusionProtocol, fingerprint


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _key(d):
    return 'D_' + format(d, '.6g').replace('.', 'p')


def _physics(Physics, d):
    return Physics('cubic', diffusion=d, particles=5_000_000,
                   half_width=60. if d <= 2 else 96., bins=100 if d <= 2 else 160)


def _inside(root, value):
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('Cubic data file escapes its declared data directory')
    return path


def _signature(records, protocol, fingerprint):
    selected = [{'D': row['D'], 'role': 'training', 'sample_signature': row['sample_signature'],
                 'field_sha256': row['files']['field.npz']['sha256'], 'total_mass': row['total_mass']}
                for row in records if row['D'] in TRAIN_D]
    return fingerprint({'protocol': protocol, 'training_data': selected})


def _check_samples(np, path, d):
    names = {'eval_target', 'eval_prior', 'eval_target_independent'}
    if d <= 2:
        names |= {'train_target', 'train_prior'}
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != names:
            raise ValueError(f'Cubic sample arrays are incomplete: D={d}')
        for name in names:
            values = saved[name]
            count = 20000 if name.startswith('train_') else 200000
            if values.shape != (count, 3) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f'Cubic sample shape, dtype or values differ: D={d}, {name}')


def load(directory: Path) -> dict:
    """Validate a completed independent data run, including every file hash."""
    root = Path(directory).expanduser().resolve(strict=True)
    manifest = _read(root / 'manifest.json')
    np, _, Physics, _, _, _, _, _, DiffusionProtocol, fingerprint = _numerics()
    protocol = DiffusionProtocol('cubic', 5_000_000).metadata()
    if (manifest.get('schema') != 1 or manifest.get('type') != 'cubic_physical_data'
            or manifest.get('status') != 'complete' or manifest.get('models') != ['cubic']
            or manifest.get('evaluation_D') != list(LEVELS) or manifest.get('training_D') != list(TRAIN_D)
            or manifest.get('protocol') != protocol or manifest.get('physical_simulations_from_scratch') is not True):
        raise ValueError('Cubic data run is incomplete or uses another physical protocol')
    sources = ('deepparticle/non_kpp/physics.py', 'deepparticle/non_kpp/experiment.py', 'deepparticle/non_kpp/sweep.py')
    if manifest.get('source_sha256') != {name: _hash(ROOT / name) for name in sources}:
        raise ValueError('Cubic physical or sampling numerical source differs from the recorded data generator')
    records = manifest.get('conditions', [])
    if len(records) != 9 or [row.get('D') for row in records] != list(LEVELS):
        raise ValueError('Cubic data run must contain all nine ordered diffusion conditions')
    for row in records:
        d = row['D']
        physics = asdict(_physics(Physics, d))
        files = row['files']
        if row['physics'] != physics or set(files) != {'field.npz', 'simulation.json', 'samples.npz', 'samples.json'}:
            raise ValueError(f'Cubic physical dataset metadata differs: D={d}')
        paths = {}
        for name, item in files.items():
            path = _inside(root, item['path'])
            if path != (root / 'datasets/cubic' / _key(d) / name).resolve() or _hash(path) != item['sha256']:
                raise ValueError(f'Cubic data path or hash differs: D={d}, {name}')
            paths[name] = path
        simulation, sampling = _read(paths['simulation.json']), _read(paths['samples.json'])
        if (simulation.get('status') != 'complete' or simulation.get('physics') != physics
                or len(simulation.get('records', [])) != 40
                or [record.get('step') for record in simulation['records']] != list(range(1, 41))
                or any(record.get('discarded') != 0 for record in simulation['records'])
                or not math.isfinite(row['total_mass']) or row['total_mass'] <= 0
                or simulation['total_mass'] != row['total_mass'] or sampling.get('physics') != physics):
            raise ValueError(f'Cubic physical simulation is incomplete or loses particles: D={d}')
        with np.load(paths['field.npz'], allow_pickle=False) as field:
            bins = physics['bins']
            dx = 2 * physics['half_width'] / bins
            centers = np.arange(bins) * dx - physics['half_width'] + dx / 2
            if (set(field.files) != {'centers', 'density', 'mass_history'}
                    or field['density'].shape != (bins, bins, bins) or field['density'].dtype != np.float32
                    or not np.isfinite(field['density']).all() or (field['density'] < 0).any()
                    or field['density'].max() <= 0 or field['density'].max() > 1
                    or not np.array_equal(field['centers'], centers)
                    or field['mass_history'].shape != (41,) or not np.isfinite(field['mass_history']).all()
                    or field['mass_history'][0] != 4 / 3 * math.pi * physics['initial_radius'] ** 3
                    or (field['mass_history'] <= 0).any() or field['mass_history'][-1] != row['total_mass']):
                raise ValueError(f'Cubic physical field arrays differ: D={d}')
        if d <= 2:
            settings = {'n_train': 20000, 'n_eval': 200000, 'physics': physics, 'sampling_seeds': SAMPLING_SEEDS}
            expected = fingerprint({**settings, 'field_sha256': files['field.npz']['sha256']})
            if any(sampling.get(key) != value for key, value in settings.items()) or sampling.get('signature') != expected or row['sample_signature'] != expected:
                raise ValueError(f'Cubic training or sampling signature differs: D={d}')
        elif (sampling.get('pool_particles') != 200000
                or sampling.get('sampling_seeds') != {key: value for key, value in SAMPLING_SEEDS.items() if key.startswith('eval_')}
                or sampling.get('field_sha256') != files['field.npz']['sha256']):
            raise ValueError(f'Cubic OOD sampling protocol differs: D={d}')
        _check_samples(np, paths['samples.npz'], d)
    if manifest.get('training_signature') != _signature(records, protocol, fingerprint):
        raise ValueError('Cubic training data signature differs from its physical records')
    return manifest


def prepare(destination: Path) -> dict:
    """Generate exactly the manuscript's nine five-million-particle conditions."""
    destination = fresh_directory(destination, create=True)
    np, torch, Physics, ball, sample_field, simulate, write_json, sampling, DiffusionProtocol, fingerprint = _numerics()
    if not torch.cuda.is_available():
        raise RuntimeError('The original cubic physical simulation protocol requires CUDA')
    torch.set_num_threads(4)
    manifest = {'schema': 1, 'type': 'cubic_physical_data', 'status': 'running', 'models': ['cubic'],
                'training_D': list(TRAIN_D), 'evaluation_D': list(LEVELS),
                'physical_particles_per_condition': 5_000_000, 'n_train_per_D': 20000, 'n_eval_per_D': 200000,
                'physical_simulations_from_scratch': True, 'retained_NPZ_or_checkpoint_used': False,
                'protocol': DiffusionProtocol('cubic', 5_000_000).metadata(), 'conditions': [],
                'source_sha256': {name: _hash(ROOT / name) for name in
                    ('deepparticle/non_kpp/physics.py', 'deepparticle/non_kpp/experiment.py', 'deepparticle/non_kpp/sweep.py')},
                'environment': {'torch': str(torch.__version__), 'numpy': str(np.__version__), 'device': torch.cuda.get_device_name()}}
    write_json(destination / 'manifest.json', manifest)
    try:
        for d in LEVELS:
            physics = _physics(Physics, d)
            folder = destination / 'datasets/cubic' / _key(d)
            # The destination is new; neither saved fields nor resume states exist.
            if d <= 2:
                values, signature = sampling(physics, folder, 20000, 200000, 'cuda')
                del values
            else:
                simulate(physics, folder, 'cuda')
                with np.load(folder / 'field.npz', allow_pickle=False) as field:
                    density = field['density']
                np.savez_compressed(folder / 'samples.npz',
                    eval_prior=ball(200000, physics.initial_radius, 43345),
                    eval_target=sample_field(density, physics, 200000, 33345),
                    eval_target_independent=sample_field(density, physics, 200000, 53345))
                write_json(folder / 'samples.json', {'physics': asdict(physics), 'pool_particles': 200000,
                    'sampling_seeds': {key: value for key, value in SAMPLING_SEEDS.items() if key.startswith('eval_')},
                    'field_sha256': _hash(folder / 'field.npz'),
                    'purpose': 'Evaluation only; no training or tuning samples are produced'})
                signature = None
                del density
            simulation = _read(folder / 'simulation.json')
            if len(simulation['records']) != 40 or any(row['discarded'] != 0 for row in simulation['records']):
                raise ValueError(f'Cubic physical simulation lost particles or ended early: D={d}')
            _check_samples(np, folder / 'samples.npz', d)
            manifest['conditions'].append({'D': d, 'role': 'training' if d in TRAIN_D else 'interpolation' if d <= 2 else 'extrapolation',
                'physics': asdict(physics), 'total_mass': simulation['total_mass'], 'sample_signature': signature,
                'files': {name: {'path': (folder / name).relative_to(destination).as_posix(), 'sha256': _hash(folder / name)}
                          for name in ('field.npz', 'simulation.json', 'samples.npz', 'samples.json')}})
            write_json(destination / 'manifest.json', manifest)
            print(f'Cubic physical data complete: D={d:g}, 5000000 particles, 40 steps', flush=True)
        manifest.update(status='complete', training_signature=_signature(manifest['conditions'], manifest['protocol'], fingerprint))
        write_json(destination / 'manifest.json', manifest)
        load(destination)
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(destination / 'manifest.json', manifest)
        raise
    return manifest
