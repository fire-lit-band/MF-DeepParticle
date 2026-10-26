"""Public orchestration of fresh training, inference, evaluation and figures."""
from pathlib import Path
import json
import time
from .locations import fresh_directory
from .data_runs import stage as data_stage, resolve_data_run, identity as data_identity

MODELS = ('ks3d', 'ks4d', 'kpp2d', 'kpp3d', 'cubic')


def select(models):
    names = list(models)
    if not names or len(names) != len(set(names)) or any(n not in MODELS for n in names):
        raise ValueError(f'Choose distinct model names from {MODELS}')
    return names


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def train_models(models, destination, *, strict_replay=False, data_run=None):
    names = select(models)
    data_run = resolve_data_run(data_run)
    if strict_replay and data_run is not None:
        raise ValueError('Strict replay requires retained data; omit --strict-replay for newly generated physical data.')
    destination = fresh_directory(destination, create=True)
    record = {'schema': 1, 'status': 'running', 'models': names, 'paths': {}, 'results': {},
              'strict_replay': strict_replay, 'physical_reference_data': 'generated' if data_run else 'retained',
              'data_run': str(data_run) if data_run else None, 'data_run_sha256': data_identity(data_run)}
    path = destination / 'training_results.json'
    write(path, record)
    try:
        body = [n for n in names if n != 'cubic']
        if body:
            from .experiments import train
            record['results']['ks_kpp'] = train(body, destination / 'ks_kpp', strict_replay=strict_replay, data_run=data_run)
            record['paths']['ks_kpp'] = 'ks_kpp'
            write(path, record)
        if 'cubic' in names:
            from .cubic_experiment import train
            result = train(destination / 'cubic', data_run=data_stage(data_run, 'cubic', ['cubic']))
            record['results']['cubic'] = result
            record['paths']['cubic'] = 'cubic'
            if strict_replay and not result.get('all_weights_bitwise_equal', False):
                raise RuntimeError('Cubic strict replay did not reproduce every historical tensor; inspect its verification report.')
        record['status'] = 'complete'
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(path, record)
        raise
    write(path, record)
    return record


def generate_models(models, destination, *, model_run=None, data_run=None):
    names = select(models)
    data_run = resolve_data_run(data_run, model_run)
    if data_run is not None and model_run is None:
        raise ValueError('New physical data requires models trained on that data; pass --model-run.')
    if 'cubic' in names:
        raise ValueError('Cubic inference is integrated into evaluate; generate accepts KS/KPP only.')
    destination = fresh_directory(destination, create=True)
    record = {'schema': 1, 'status': 'running', 'models': names, 'paths': {}, 'results': {},
              'data_run': str(data_run) if data_run else None}
    path = destination / 'generation_results.json'
    write(path, record)
    try:
        for family in ('ks', 'kpp'):
            selected = [n for n in names if n.startswith(family)]
            if not selected:
                continue
            if family == 'ks':
                from .ks_workflows import generate
            else:
                from .kpp_workflows import generate
            record['results'][family] = generate(selected, destination / family, model_run=model_run,
                                                 data_run=data_stage(data_run, family, selected))
            record['paths'][family] = family
            write(path, record)
        record['status'] = 'complete'
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(path, record)
        raise
    write(path, record)
    return record


def _generated_stage(root, family, names):
    if root is None:
        return None
    root = Path(root).expanduser().resolve(strict=True)
    record = load(root / 'generation_results.json')
    if record.get('status') != 'complete' or not set(names).issubset(record.get('models', [])):
        raise ValueError('The generate output is incomplete or lacks requested models.')
    stage = (root / record['paths'][family]).resolve(strict=True)
    if not stage.is_relative_to(root):
        raise ValueError('Generated stage escapes its output directory.')
    return stage


def evaluate_models(models, destination, *, model_run=None, generated=None, reference_run=None, workers=1, auxiliary=False, data_run=None):
    names = select(models)
    data_run = resolve_data_run(data_run, model_run)
    if data_run is not None and model_run is None:
        raise ValueError('New physical data requires models trained on that data; pass --model-run.')
    if data_run is not None and reference_run is not None:
        raise ValueError('A complete data run already contains the physical references; omit --reference-run.')
    if reference_run is not None and not any(n.startswith('kpp') for n in names):
        raise ValueError('reference_run supplies KPP references; select at least one KPP model.')
    if isinstance(workers, bool) or workers < 1:
        raise ValueError('workers must be positive')
    destination = fresh_directory(destination, create=True)
    record = {'schema': 1, 'status': 'running', 'models': names, 'paths': {}, 'results': {},
              'model_run': str(Path(model_run).expanduser().resolve()) if model_run else None,
              'physical_reference_data': 'generated' if data_run else 'regenerated_KPP' if reference_run else 'retained',
              'data_run': str(data_run) if data_run else None, 'data_run_sha256': data_identity(data_run),
              'reference_run': str(Path(reference_run).expanduser().resolve()) if reference_run else None,
              'fresh_model_inference': generated is None}
    path = destination / 'reproduction_results.json'
    write(path, record)
    try:
        for family in ('ks', 'kpp'):
            selected = [n for n in names if n.startswith(family)]
            if not selected:
                continue
            source = _generated_stage(generated, family, selected)
            if family == 'ks':
                from .ks_workflows import evaluate
                result = evaluate(selected, destination / family, model_run=model_run, generated=source, workers=workers,
                                  data_run=data_stage(data_run, family, selected))
            else:
                from .kpp_workflows import evaluate
                result = evaluate(selected, destination / family, model_run=model_run, generated=source, reference_run=reference_run,
                                  data_run=data_stage(data_run, family, selected))
            record['results'][family] = result
            record['paths'][family] = family
            write(path, record)
        if 'cubic' in names:
            from .cubic_experiment import evaluate
            record['results']['cubic'] = evaluate(destination / 'cubic', model_run=model_run,
                                                  data_run=data_stage(data_run, 'cubic', ['cubic']))
            record['paths']['cubic'] = 'cubic'
            write(path, record)
        if auxiliary and 'ks3d' in names:
            from .ks_workflows import multistep
            record['results']['multistep'] = multistep(destination / 'multistep', model_run=model_run, workers=workers,
                                                        data_run=data_stage(data_run, 'ks', ['ks3d']))
            record['paths']['multistep'] = 'multistep'
            write(path, record)
        kpp = [n for n in names if n.startswith('kpp')]
        if auxiliary and kpp:
            from .kpp_workflows import warm_start
            record['results']['warm'] = warm_start(kpp, destination / 'warm', model_run=model_run,
                                                    data_run=data_stage(data_run, 'kpp', kpp))
            record['paths']['warm'] = 'warm'
        record['status'] = 'complete'
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(path, record)
        raise
    write(path, record)
    return record


def reproduce(models, destination, *, workers=1, strict_replay=False, model_run=None, reference_run=None,
              data_run=None, from_scratch=False):
    names = select(models)
    if from_scratch and (data_run is not None or model_run is not None or reference_run is not None or strict_replay):
        raise ValueError('--from-scratch creates all data and weights; omit --data-run, --model-run, --reference-run and --strict-replay.')
    data_run = resolve_data_run(data_run, model_run)
    if data_run is not None and (strict_replay or reference_run is not None):
        raise ValueError('A new data run must not be combined with strict replay or separate references.')
    if reference_run is not None and not any(n.startswith('kpp') for n in names):
        raise ValueError('reference_run supplies KPP references; select at least one KPP model.')
    destination = fresh_directory(destination, create=True)
    started = time.perf_counter()
    record = {'schema': 1, 'status': 'running', 'models': names,
              'physical_reference_data': 'generated' if from_scratch or data_run else 'regenerated_KPP' if reference_run else 'retained',
              'from_scratch': from_scratch,
              'new_training': model_run is None, 'paths': {}, 'stages': []}
    path = destination / 'reproduction_run.json'
    def stage(name):
        record.update(current_stage=name, elapsed_seconds=time.perf_counter() - started)
        write(path, record)
        print(f'Reproduction stage: {name}', flush=True)
    try:
        if from_scratch:
            from .data_runs import prepare_models
            stage('data')
            prepare_models(names, destination / 'data')
            data_run = destination / 'data'
            record['paths']['data'] = 'data'
            record['stages'].append('data')
        record['data_run'] = str(data_run) if data_run else None
        record['data_run_sha256'] = data_identity(data_run)
        if model_run is None:
            stage('training')
            train_models(names, destination / 'training', strict_replay=strict_replay, data_run=data_run)
            model_run = destination / 'training'
            record['paths']['training'] = 'training'
            record['stages'].append('training')
        else:
            model_run = Path(model_run).expanduser().resolve(strict=True)
            record['external_model_run'] = str(model_run)
        stage('evaluation')
        evaluate_models(names, destination / 'results', model_run=model_run, reference_run=reference_run,
                        workers=workers, auxiliary=True, data_run=data_run)
        record['paths']['results'] = 'results'
        record['stages'].append('evaluation')
        body = [n for n in names if n != 'cubic']
        if body:
            stage('timing')
            from .timing import benchmark
            benchmark(destination / 'results/timing', model_run=model_run, models=body, data_run=data_run)
            results = load(destination / 'results/reproduction_results.json')
            results['paths']['timing'] = 'timing'
            write(destination / 'results/reproduction_results.json', results)
            record['stages'].append('timing')
        stage('figures')
        from .result_figures import render
        figures = render(destination / 'figures', results=destination / 'results')
        if not figures:
            raise RuntimeError('No figures were produced from the new results.')
        record['paths']['figures'] = 'figures'
        record['figure_count'] = len(figures)
        record['stages'].append('figures')
        record.update(status='complete', current_stage='complete', elapsed_seconds=time.perf_counter() - started)
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.perf_counter() - started)
        write(path, record)
        raise
    write(path, record)
    return record
