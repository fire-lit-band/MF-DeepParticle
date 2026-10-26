"""Generate and load independent KPP training, plotting and evaluation data."""
from __future__ import annotations

from pathlib import Path

from . import ROOT
from .locations import fresh_directory

MODELS = ('kpp2d', 'kpp3d')
TRAIN_LEVELS = tuple(2 + .25 * i for i in range(8))
EVAL_LEVELS = tuple(2 + .25 * i for i in range(11))
PHYSICAL_SEEDS = (666, 999)


def resolve(data_run, models=MODELS):
    """Validate a completed KPP family data run without substituting fixed data."""
    from .kpp_workflows import _json, _hash, _inside
    root = Path(data_run).expanduser().resolve(strict=True)
    manifest = _json(root / 'manifest.json')
    if (manifest.get('schema') != 1 or manifest.get('type') != 'kpp_data'
            or manifest.get('status') != 'complete'
            or not set(models).issubset(manifest.get('models', []))
            or manifest.get('train_levels') != list(TRAIN_LEVELS)
            or manifest.get('eval_levels') != list(EVAL_LEVELS)
            or manifest.get('training_particles') != 20000
            or manifest.get('evaluation_particles') != 20000
            or manifest.get('physical_seeds') != list(PHYSICAL_SEEDS)
            or manifest.get('generations') != 4096):
        raise ValueError('KPP data_run is incomplete or has a different physical protocol')
    reference = _inside(root, manifest['reference_run'])
    if (_hash(reference / 'manifest.json') != manifest['reference_manifest_sha256']
            or _json(reference / 'manifest.json').get('status') != 'complete'):
        raise ValueError('KPP physical reference completion record changed')
    return root, manifest


def load_dataset(name, data_run):
    """Return (folder, protocol, manifest, splits) for newly generated KPP data."""
    from .experiments import _load, _source_environment
    from .kpp_workflows import _json, _hash, _inside
    if name not in MODELS:
        raise ValueError(f'Unknown KPP model: {name}')
    root, run = resolve(data_run, (name,))
    record = run['datasets'][name]
    folder = _inside(root, record['path'])
    if not folder.is_relative_to(root / 'datasets'):
        raise ValueError('KPP dataset escaped its data directory')
    if _hash(folder / 'manifest.json') != record['manifest_sha256']:
        raise ValueError('KPP dataset manifest changed')
    with _source_environment([]):
        uploaded = _load('_public_kpp_data_loader', ROOT / 'output/kpp_original_training_total1k_20261014/experiment.py')
        protocol, manifest, splits = uploaded.load_dataset(folder)
        if protocol.name != name or manifest['signature'] != record['signature']:
            raise ValueError('KPP dataset name or signature differs from the completed data run')
        for split in ('train', 'eval'):
            for case in splits[split]:
                source = _inside(root, case['meta']['source_file'])
                if not source.is_relative_to(root / 'raw') or _hash(source) != case['meta']['source_sha256']:
                    raise ValueError('KPP exported dataset source changed')
                for key in ('initial', 'target', 'prior'):
                    array = case[key]
                    if array.shape != (20000, protocol.dimension) or not uploaded.np.isfinite(array).all():
                        raise ValueError(f'Invalid KPP {key} array')
    return folder, protocol, manifest, splits


def _export(uploaded, model, destination, raw_records):
    """Preserve the original 20k schema while deriving identities from new files."""
    from .kpp_workflows import _hash, _inside
    protocol = uploaded.protocol(model)
    folder = destination / 'datasets' / model
    identity = {'protocol': protocol.metadata(), 'train_levels': list(TRAIN_LEVELS),
                'eval_levels': list(EVAL_LEVELS), 'train_physical_seed': 666,
                'eval_physical_seed': 999,
                'prior': 'Normal(0,pi), distinct training/evaluation pools; same pool across kappa',
                'source': 'Validated GPU port of uploaded source, independent Philox streams by seed'}
    manifest = {'identity': identity, 'protocol': protocol.metadata(), 'purpose': 'experiment',
                'status': 'preparing', 'train': [], 'eval': []}
    records = {(row['level'], row['seed']): row for row in raw_records}
    if len(records) != 22:
        raise ValueError('KPP raw physical conditions are incomplete or duplicated')
    for split, levels, seed in (('train', TRAIN_LEVELS, 666), ('eval', EVAL_LEVELS, 999)):
        for index, level in enumerate(levels):
            row = records[level, seed]
            source = _inside(destination, row['file'])
            if _hash(source) != row['sha256']:
                raise ValueError('New KPP raw cloud hash changed')
            arrays, raw_meta = uploaded.read_npz(source)
            expected = {'model': model, 'level': level, 'seed': seed,
                        'particles': 20000, 'generations': 4096, 'selection_count': 4096}
            if (any(raw_meta.get(key) != value for key, value in expected.items())
                    or raw_meta['source_hash'] != protocol.metadata()['source_sha256']
                    or raw_meta['protocol']['resample_every'] != 256
                    or raw_meta['protocol']['potential'] != protocol.metadata()['potential']):
                raise ValueError('New raw KPP reference has a different physical protocol')
            path = folder / 'clouds' / f'{split}_{index:02d}.npz'
            meta = {'protocol': protocol.metadata(), 'source_file': source.relative_to(destination).as_posix(),
                    'source_sha256': row['sha256'], 'source_meta': raw_meta, 'split': split,
                    'level': level, 'seed': seed, 'trace': raw_meta['trace'],
                    'c_hat': raw_meta['c_hat'], 'mu_hat': raw_meta['mu_hat'],
                    'endpoint_pairing': 'unpaired FK marginals; MF pairs independent Gaussian source and target samples'}
            uploaded.write_npz(path, meta=meta, initial=arrays['initial'], target=arrays['target'],
                               prior=uploaded.prior(protocol, 20000, seed + 400003))
            manifest[split].append({'level': level, 'parameter': protocol.parameter(level),
                                    'file': path.relative_to(folder).as_posix(), 'sha256': _hash(path)})
    manifest['signature'] = uploaded.fingerprint({key: manifest[key] for key in ('identity', 'train', 'eval')})
    manifest['status'] = 'complete'
    uploaded.write_json(folder / 'manifest.json', manifest)
    uploaded.load_dataset(folder)
    return {'path': folder.relative_to(destination).as_posix(), 'signature': manifest['signature'],
            'manifest_sha256': _hash(folder / 'manifest.json'), 'training_files': 8, 'evaluation_files': 11}


def prepare(models, destination):
    """Simulate all 20k data and 100k references from uniform particles to T4096.

    The original two batch groups are retained: levels3.75/4.5, then the other
    nine levels. Each group uses independent seeds666/999. Their44 raw outputs
    yield38 required train/evaluation files; six unused training-seed extrapolation
    outputs are retained as physical provenance. No previous NPZ or PT is read.
    """
    from .kpp_workflows import _models, _kernels, _json, _hash, _write, references
    from .portability import verified_source_identity
    names = _models(models, MODELS)
    destination = fresh_directory(destination, create=True)
    manifest = {'schema': 1, 'type': 'kpp_data', 'status': 'running', 'models': names,
                'train_levels': list(TRAIN_LEVELS), 'eval_levels': list(EVAL_LEVELS),
                'physical_seeds': list(PHYSICAL_SEEDS), 'training_particles': 20000,
                'evaluation_particles': 20000, 'generations': 4096,
                'uniform_initialization': True, 'datasets': {}, 'raw_batches': [],
                'reference_run': 'reference_run', 'reference_manifest_sha256': None,
                'historical_NPZ_or_PT_required': False}
    _write(destination / 'manifest.json', manifest)
    with _kernels(cuda=True) as (measure, _):
        uploaded = measure.e.uploaded
        gpu = uploaded.load_accelerator()
        prior_validation = _json(ROOT / 'output/kpp_original_sampling_20261014/gpu_validation.json')
        if prior_validation.get('status') != 'passed':
            raise ValueError('Prior physical source validation did not pass')
        for filename, digest in prior_validation['source_hashes'].items():
            if verified_source_identity(gpu.ARCHIVE / filename, root=ROOT)['source_sha256'] != digest:
                raise ValueError('Physical source identity differs from numerical validation')
        gpu.SEEDS = PHYSICAL_SEEDS
        groups = (('representative', (3.75, 4.5)),
                  ('additional', tuple(level for level in EVAL_LEVELS if level not in (3.75, 4.5))))
        for name in names:
            raw_records = []
            for label, levels in groups:
                gpu.LEVELS = levels
                gpu.OUT = destination / 'raw' / name / label
                gpu.OUT.mkdir(parents=True)
                validation = gpu.validate_dimension(int(name[-2]))
                _write(gpu.OUT / 'validation.json', validation)
                print(f'{name}: generating20k {label} data, {len(levels)} levels x2 physical seeds, T4096', flush=True)
                rows = gpu.run_dimension(int(name[-2]), 20000, 4096, validation)
                if len(rows) != len(levels) * 2:
                    raise RuntimeError('Incomplete 20k KPP simulation')
                for row in rows:
                    row['file'] = Path(row['file']).relative_to(destination).as_posix()
                raw_records.extend(rows)
                batch = {'model': name, 'group': label, 'levels': list(levels), 'seeds': list(PHYSICAL_SEEDS),
                         'outputs': rows, 'validation': validation, 'status': 'complete'}
                _write(gpu.OUT / 'manifest.json', batch)
                manifest['raw_batches'].append({'model': name, 'group': label,
                    'manifest': (gpu.OUT / 'manifest.json').relative_to(destination).as_posix(),
                    'sha256': _hash(gpu.OUT / 'manifest.json')})
                _write(destination / 'manifest.json', manifest)
            manifest['datasets'][name] = _export(uploaded, name, destination, raw_records)
            _write(destination / 'manifest.json', manifest)
    references(names, destination / 'reference_run')
    manifest['reference_manifest_sha256'] = _hash(destination / 'reference_run/manifest.json')
    manifest['status'] = 'complete'
    _write(destination / 'manifest.json', manifest)
    for name in names:
        load_dataset(name, destination)
    return manifest
