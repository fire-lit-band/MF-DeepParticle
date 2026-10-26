"""Create and consume physical data independently of retained binary artifacts."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import time

from .locations import fresh_directory


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def identity(data_run):
    if data_run is None:
        return None
    return hashlib.sha256((Path(data_run).resolve(strict=True) / 'data_results.json').read_bytes()).hexdigest()


def resolve_data_run(data_run=None, model_run=None):
    """Infer the training data run and reject silently selecting other data."""
    recorded, expected = None, None
    if model_run is not None:
        model = Path(model_run).expanduser().resolve(strict=True)
        for candidate in (model / 'training_results.json', model.parent / 'training_results.json',
                          model / 'replay_request.json'):
            if candidate.is_file():
                record = read(candidate)
                if record.get('data_run') is not None:
                    recorded = record['data_run']
                    expected = record.get('data_run_sha256')
                    break
    value = data_run if data_run is not None else recorded
    if value is None:
        return None
    root = Path(value).expanduser().resolve(strict=True)
    record = read(root / 'data_results.json')
    if (record.get('schema'), record.get('type'), record.get('status')) != (1, 'physical_data_run', 'complete'):
        raise ValueError('Physical data run must be complete before training or evaluation.')
    if expected is not None and identity(root) != expected:
        raise ValueError('Selected physical data differs from the completed training run.')
    return root


def stage(data_run, family, models=()):
    """Resolve a completed family under a public data command output."""
    if data_run is None:
        return None
    root = Path(data_run).expanduser().resolve(strict=True)
    record = read(root / 'data_results.json')
    if (record.get('schema') != 1 or record.get('type') != 'physical_data_run'
            or record.get('status') != 'complete' or not set(models).issubset(record.get('models', []))):
        raise ValueError('Choose a completed data run covering the requested models.')
    relative = record.get('paths', {}).get(family)
    if not isinstance(relative, str) or not relative:
        raise ValueError(f'Data run does not contain {family}')
    folder = (root / relative).resolve(strict=True)
    if folder == root or not folder.is_relative_to(root):
        raise ValueError('Physical data family escapes its declared run.')
    manifest = folder / 'manifest.json'
    expected = record.get('manifest_sha256', {}).get(family)
    if not expected or hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Physical data manifest changed after completion: {family}')
    if read(manifest).get('status') != 'complete':
        raise ValueError(f'Physical data generation is incomplete: {family}')
    return folder


def prepare_models(models, destination):
    from .workflow import select
    names = select(models)
    destination = fresh_directory(destination, create=True)
    record = {'schema': 1, 'type': 'physical_data_run', 'status': 'running',
              'models': names, 'paths': {}, 'results': {}, 'manifest_sha256': {},
              'uses_retained_binary_data': False}
    path = destination / 'data_results.json'
    started = time.perf_counter()
    write(path, record)
    try:
        for family in ('ks', 'kpp', 'cubic'):
            selected = [n for n in names if n.startswith(family)]
            if not selected:
                continue
            record['current_stage'] = family
            write(path, record)
            print(f'Physical data generation: {family}', flush=True)
            if family == 'ks':
                from .ks_data import prepare
                result = prepare(selected, destination / family)
            elif family == 'kpp':
                from .kpp_data import prepare
                result = prepare(selected, destination / family)
            else:
                from .cubic_data import prepare
                result = prepare(destination / family)
            manifest = destination / family / 'manifest.json'
            if read(manifest).get('status') != 'complete':
                raise RuntimeError(f'{family} did not complete physical generation.')
            record['results'][family] = result
            record['paths'][family] = family
            record['manifest_sha256'][family] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            record['elapsed_seconds'] = time.perf_counter() - started
            write(path, record)
        record.update(status='complete', current_stage='complete')
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['elapsed_seconds'] = time.perf_counter() - started
        write(path, record)
    return record


@contextmanager
def bind_datasets(engine, data_run):
    """Select new datasets without changing the original training kernels."""
    if data_run is None:
        yield
        return
    original = engine.dataset
    def dataset(name):
        family = 'ks' if name.startswith('ks') else 'kpp'
        folder = stage(data_run, family, [name])
        if family == 'ks':
            from .ks_data import load_dataset
        else:
            from .kpp_data import load_dataset
        path, protocol, manifest, splits = load_dataset(name, folder)
        return path, None, (protocol, manifest, splits)
    engine.dataset = dataset
    try:
        yield
    finally:
        engine.dataset = original
