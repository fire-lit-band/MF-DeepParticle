"""Plot explicitly declared fresh results without falling back to frozen data.

The result root must contain reproduction_results.json with schema=1,
status='complete', and a nonempty paths mapping. Scientific figure stages are
ks, kpp, cubic, multistep and warm; optional timing tables remain as CSVs.
Each figure stage must carry its completed producer
manifest. Partial experiments produce only their declared scientific figures.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

import numpy as np


class _Inputs:
    def __init__(self):
        self.files = {}

    def bytes(self, path):
        path = Path(path).resolve(strict=True)
        value = path.read_bytes()
        self.files[str(path)] = hashlib.sha256(value).hexdigest()
        return value

    def json(self, path):
        return json.loads(self.bytes(path).decode('utf-8-sig'))

    def csv(self, path):
        rows = list(csv.DictReader(io.StringIO(self.bytes(path).decode('utf-8-sig'))))
        if not rows:
            raise ValueError(f'Empty result table: {path}')
        return rows

    def clouds(self, path, methods, *, size, dimension, expected_hash=None):
        payload = self.bytes(path)
        if expected_hash is not None and hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f'Result particle hash differs: {path}')
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            values = {method: archive['reference' if method == 'ref' else f'seed88_{method}'].copy()
                      for method in methods}
        if any(points.shape != (size, dimension) or not np.isfinite(points).all() for points in values.values()):
            raise ValueError(f'Unexpected result particle arrays: {path}')
        return values


def _path(base, value, *, inside=True):
    if not isinstance(value, str) or not value:
        raise ValueError('A result manifest must declare a nonempty path')
    base = Path(base).resolve(strict=True)
    candidate = Path(value.replace('\\', '/'))
    result = (candidate if candidate.is_absolute() else base / candidate).resolve(strict=True)
    if inside and not result.is_relative_to(base):
        raise ValueError(f'Result artifact escapes its declared directory: {value}')
    return result


def _manifest(inputs, path, *, kind=None, schema=None):
    value = inputs.json(path)
    if value.get('status') not in ('complete', 'passed'):
        raise ValueError(f'Result producer has not completed: {path}')
    if kind is not None and (value.get('type') != kind or value.get('schema') != 1):
        raise ValueError(f'Unexpected result manifest type: {path}')
    if schema is not None and value.get('schema') != schema:
        raise ValueError(f'Unexpected result schema: {path}')
    return value


def _models(value, allowed):
    names = value.get('models')
    if not isinstance(names, list) or not names or len(set(names)) != len(names) or not set(names) <= set(allowed):
        raise ValueError(f'Result manifest requires explicit supported models: {allowed}')
    return names


def _series(summary, raw, model, methods, levels, *, fields, value='w2'):
    x_key, mean_key, low_key, high_key = fields
    result = {}
    for method in methods:
        rows = sorted((row for row in summary if row.get('model', 'cubic') == model and row['method'] == method),
                      key=lambda row: float(row[x_key]))
        if [float(row[x_key]) for row in rows] != list(levels):
            raise ValueError(f'Incomplete or duplicated curve: {model}, {method}')
        values = np.array([[float(row[key]) for row in rows] for key in fields], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values[2] < 0) or np.any(values[2] > values[1]) or np.any(values[1] > values[3]):
            raise ValueError(f'Invalid curve values: {model}, {method}')
        for row in rows:
            level = float(row[x_key])
            trials = [record for record in raw if record['model'] == ('non_kpp_cubic' if model == 'cubic' else model)
                      and record['method'] == method and float(record.get('level', record.get('sigma'))) == level]
            if len(trials) != 3 or {int(record['seed']) for record in trials} != {88, 89, 90}:
                raise ValueError(f'Curve point must have three model seeds: {model}, {method}, {level}')
            samples = [float(record[value]) for record in trials]
            if not np.isfinite(samples).all():
                raise ValueError(f'Nonfinite per-seed measurements: {model}, {method}, {level}')
            for key, expected in ((mean_key, statistics.mean(samples)), (low_key, min(samples)), (high_key, max(samples))):
                if abs(float(row[key]) - expected) > 1e-10:
                    raise ValueError(f'Fresh summary differs from its per-seed measurements: {model}, {method}, {level}')
        result[method] = values
    return result


def _curve(destination, relative, series, *, xlabel, boundary, title, normalized=False, training=None):
    import matplotlib.pyplot as plt
    from . import kpp

    colors = dict(kpp.COLORS, mf_two_step='#D37B2F')
    markers = dict(kpp.MARKERS, mf_two_step='^')
    labels = dict(kpp.LABELS, mf_two_step='MF (2 steps)')
    with plt.rc_context(kpp.STYLE):
        fig, ax = plt.subplots(figsize=(6.2, 3.8), layout='constrained')
        for method, (x, mean, low, high) in series.items():
            ax.plot(x, mean, color=colors[method], lw=1.7, label=labels[method],
                    marker=None if training is not None else markers[method], ms=4)
            ax.fill_between(x, low, high, color=colors[method], alpha=.13, lw=0)
            if training is not None:
                for held_out in (False, True):
                    mask = np.array([(level not in training) == held_out for level in x])
                    ax.plot(x[mask], mean[mask], ls='none', marker=markers[method], ms=5,
                            markerfacecolor='white' if held_out else colors[method], markeredgecolor=colors[method])
        ax.axvline(boundary, color='#777777', ls=':', lw=1.2)
        # New models may have errors outside the historical figure limits.
        upper = max(float(values[3].max()) for values in series.values())
        ax.set(xlabel=xlabel, ylabel=r'Normalized $W_2$' if normalized else r'$W_2$',
               title=title, ylim=(0, max(upper * 1.08, 1e-8)))
        ax.grid(alpha=.23, ls='--')
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        return kpp._save(fig, destination, Path(relative))


def _density_panels(destination, clouds, *, name, level, tag):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from . import kpp

    pairs = ((0, 1),) if name == 'kpp2d' else ((0, 1), (0, 2), (1, 2))
    histograms = kpp._histograms(clouds, pairs)
    maxima = [max(panels[column].max() for panels in histograms.values()) for column in range(len(pairs))]
    paths = []
    with plt.rc_context(kpp.STYLE):
        for method, panels in histograms.items():
            fig, axes = plt.subplots(1, len(pairs), figsize=(3.5, 2.7) if len(pairs) == 1 else (9.5, 2.7),
                                     layout='constrained', squeeze=False)
            for column, ((first, second), density) in enumerate(zip(pairs, panels)):
                ax = axes[0, column]
                mesh = ax.imshow(density.T, origin='lower', extent=[0, 2 * np.pi, 0, 2 * np.pi],
                                 norm=Normalize(0, maxima[column]), cmap='viridis', interpolation='nearest')
                ax.set(xlabel='$' + 'xyz'[first] + '$', ylabel='$' + 'xyz'[second] + '$')
                kpp._periodic_ticks(ax)
                if column == (0 if len(pairs) == 1 else 1):
                    ax.set_title(kpp.LABELS[method] + f', $s={level:g}$')
                fig.colorbar(mesh, ax=ax, pad=.025, shrink=.85)
            paths.append(kpp._save(fig, destination, Path('kpp') / f'{name}_{tag}_{method}.pdf'))
    return paths


def _warm_curve(destination, name, rows):
    import matplotlib.pyplot as plt
    from . import kpp

    rows = [row for row in rows if row['model'] == name and float(row['level']) == 4.5]
    generations = [1, 2, 4, 8, 16, 32, 64, 128]
    references = {float(row['reference_c']) for row in rows}
    if len(references) != 1 or not np.isfinite(list(references)).all():
        raise ValueError(f'Missing or inconsistent FK reference in new results: {name}')
    series = {}
    for method in ('mf', 'dp', 'mf_dp', 'cold'):
        trials = [[float(row['c_hat']) for row in rows if row['method'] == method and int(row['generation']) == generation]
                  for generation in generations]
        if any(len(values) != (3 if method == 'cold' else 9) for values in trials):
            raise ValueError(f'Incomplete new FK traces: {name}, {method}')
        values = np.array(trials)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite new FK traces: {name}, {method}')
        series[method] = values
    with plt.rc_context(kpp.STYLE):
        fig, ax = plt.subplots(figsize=(6.2, 3.8), layout='constrained')
        for method, values in series.items():
            ax.plot(generations, values.mean(1), color=kpp.COLORS[method], marker=kpp.MARKERS[method],
                    ms=3, lw=1.4, label=kpp.LABELS[method])
            ax.fill_between(generations, values.min(1), values.max(1), color=kpp.COLORS[method], alpha=.12, lw=0)
        ax.axhline(references.pop(), color='#222222', ls='--', lw=1, label='FK reference')
        ax.set_xscale('log', base=2)
        ax.set_xticks(generations, [str(value) for value in generations])
        ax.set(xlabel=r'FK generations $T$', ylabel=r'$\widehat c_T(2)$', title=f'{name.upper()}, $s=4.5$')
        ax.grid(alpha=.23, ls='--')
        ax.legend(frameon=False, ncol=2, fontsize=8)
        return kpp._save(fig, destination, Path('warm') / f'{name}_fk.pdf')


def render(destination: Path, *, results: Path) -> list[Path]:
    """Render only completed, explicitly declared new experiment results.

    Curves are cross-checked against new per-seed CSV rows. Particle panels use
    newly generated clouds and their producer hashes. Missing declared stages
    or files are errors; no frozen result paths are consulted.
    """
    from . import ks
    from .locations import fresh_directory

    root = Path(results).expanduser().resolve(strict=True)
    destination = fresh_directory(destination)
    inputs = _Inputs()
    manifest = _manifest(inputs, root / 'reproduction_results.json')
    declared = manifest.get('paths')
    if manifest.get('schema') != 1 or manifest.get('status') != 'complete' or not isinstance(declared, dict) or not declared:
        raise ValueError('A schema-1 result manifest must declare at least one experiment path')
    if not set(declared) <= {'ks', 'kpp', 'cubic', 'multistep', 'warm', 'timing'}:
        raise ValueError(f'Unknown result stages: {set(declared) - {"ks", "kpp", "cubic", "multistep", "warm", "timing"}}')
    folders = {stage: _path(root, value) for stage, value in declared.items()}
    tasks, sources, ks_series, ks_clouds = [], {}, {}, None
    levels_ks = list(range(10, 201, 10))
    ks_fields = ('sigma', 'mean_w2', 'min_w2', 'max_w2')

    if 'ks' in folders:
        folder = folders['ks']
        spec = _manifest(inputs, folder / 'verification.json', schema='ks_evaluation_v1')
        names = _models(spec, ('ks3d', 'ks4d'))
        summary, raw = inputs.csv(folder / 'summary.csv'), inputs.csv(folder / 'metrics_per_seed.csv')
        cloud_root = _path(folder, spec['clouds_root'], inside=False)
        for name in names:
            series = _series(summary, raw, name, ('mf', 'dp', 'mf_dp'), levels_ks, fields=ks_fields)
            ks_series[name] = series
            tasks.append(('curve', (f'ks/{name}_w2.pdf', series),
                          dict(xlabel=r'$\sigma$', boundary=150., title=name.upper())))
        if 'ks3d' in names:
            path = _path(cloud_root, 'ks3d/sigma160.npz')
            identity = inputs.json(path.with_suffix('.json'))
            ks_clouds = inputs.clouds(path, ('ref', 'mf', 'mf_dp'), size=100000, dimension=3, expected_hash=identity['sha256'])
            tasks.append(('ks_projection', (ks_clouds, 'ks3d_projection'), dict(bins=None, density=True, combined=True)))
        sources['ks'] = spec

    if 'multistep' in folders:
        folder = folders['multistep']
        spec = _manifest(inputs, folder / 'verification.json', schema='ks_multistep_v1')
        summary, raw = inputs.csv(folder / 'summary.csv'), inputs.csv(folder / 'metrics_per_seed.csv')
        series = _series(summary, raw, 'ks3d', ('mf_two_step',), levels_ks, fields=ks_fields)
        if 'ks3d' in ks_series:
            series = {'mf': ks_series['ks3d']['mf'], **series, 'mf_dp': ks_series['ks3d']['mf_dp']}
        tasks.append(('curve', ('multistep/ks3d_w2.pdf', series),
                      dict(xlabel=r'$\sigma$', boundary=150., title='3D KS, two-step MF')))
        cloud_root = _path(folder, spec['clouds_root'], inside=False)
        path = _path(cloud_root, 'ks3d/sigma160.npz')
        identity = inputs.json(path.with_suffix('.json'))
        clouds = inputs.clouds(path, ('mf_two_step', 'ref'), size=100000, dimension=3, expected_hash=identity['sha256'])
        if ks_clouds is not None:
            if not np.array_equal(clouds['ref'], ks_clouds['ref']):
                raise ValueError('Fresh KS and multistep panels use different reference particles')
            clouds = {'mf_two_step': clouds['mf_two_step'], 'mf_dp': ks_clouds['mf_dp'], 'ref': clouds['ref']}
        tasks.append(('ks_projection', (clouds, 'multistep_projection'), dict(bins=60)))
        sources['multistep'] = spec

    if 'kpp' in folders:
        folder = folders['kpp']
        spec = _manifest(inputs, folder / 'manifest.json', kind='kpp_evaluation')
        names = _models(spec, ('kpp2d', 'kpp3d'))
        generation = _path(folder, spec['generated'], inside=False)
        generated = _manifest(inputs, generation / 'manifest.json', kind='kpp_generation')
        if not set(names) <= set(_models(generated, ('kpp2d', 'kpp3d'))):
            raise ValueError('KPP evaluation and particle-generation models differ')
        summary, raw = inputs.csv(folder / 'summary.csv'), inputs.csv(folder / 'metrics_per_seed.csv')
        for filename, expected in (('summary.csv', spec['summary_sha256']), ('metrics_per_seed.csv', spec['metrics_sha256'])):
            if inputs.files[str((folder / filename).resolve())] != expected:
                raise ValueError(f'Fresh KPP measurement table hash differs: {filename}')
        for name in names:
            series = _series(summary, raw, name, ('mf', 'dp', 'mf_dp'), list(np.arange(2., 4.51, .25)),
                             fields=('level', 'w2_mean', 'w2_min', 'w2_max'))
            tasks.append(('curve', (f'kpp/{name}_w2.pdf', series),
                          dict(xlabel=r'$s=-\log_2\kappa$', boundary=3.75, title=name.upper())))
            panels = ((7, 3.75, 'boundary'), (8, 4., 'extrapolation')) if name == 'kpp2d' else ((10, 4.5, 'projection'),)
            for index, level, tag in panels:
                records = [row for row in generated['density_files'][name] if int(row['index']) == index]
                if len(records) != 1 or float(records[0]['level']) != level:
                    raise ValueError(f'Missing fresh density record: {name}, {level}')
                record = records[0]
                path = _path(generation, record['path'])
                clouds = inputs.clouds(path, ('mf', 'mf_dp', 'ref'), size=20000, dimension=int(name[-2]), expected_hash=record['sha256'])
                tasks.append(('density', (clouds,), dict(name=name, level=level, tag=tag)))
        sources['kpp'] = spec

    if 'cubic' in folders:
        folder = folders['cubic']
        spec = _manifest(inputs, folder / 'manifest.json', kind='cubic_evaluation')
        _models(spec, ('cubic',))
        summary, raw = inputs.json(_path(folder, spec['summary'])), inputs.csv(_path(folder, spec['metrics']))
        series = _series(summary, raw, 'cubic', ('mf', 'dp', 'mf_dp'), [.25, .375, .5, .75, 1., 1.5, 2., 2.5, 3.],
                         fields=('D', 'mean', 'minimum', 'maximum'), value='w2_normalized')
        tasks.append(('curve', ('cubic/cubic_w2.pdf', series),
                      dict(xlabel=r'Diffusion coefficient $D$', boundary=2., title='Non-KPP Cubic',
                           normalized=True, training=(.25, .5, 1., 2.))))
        sources['cubic'] = spec

    if 'warm' in folders:
        folder = folders['warm']
        spec = _manifest(inputs, folder / 'manifest.json', kind='kpp_warm_start')
        for name in _models(spec, ('kpp2d', 'kpp3d')):
            trace = _path(folder, spec['trace_paths'][name])
            rows = inputs.csv(trace)
            if inputs.files[str(trace)] != spec['trace_sha256'][name]:
                raise ValueError(f'Fresh FK trace hash differs: {name}')
            tasks.append(('warm', (name, rows), {}))
        sources['warm'] = spec
    if not tasks:
        raise ValueError('No declared experiment has renderable results')

    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind, args, kwargs in tasks:
        if kind == 'curve':
            paths.append(_curve(destination, *args, **kwargs))
        elif kind == 'ks_projection':
            clouds, stem = args
            paths.extend(ks._projections(clouds, destination, stem, **kwargs))
        elif kind == 'density':
            paths.extend(_density_panels(destination, *args, **kwargs))
        else:
            paths.append(_warm_curve(destination, *args, **kwargs))
    if len(paths) != len(set(paths)) or not paths:
        raise RuntimeError('Fresh-result figure paths are empty or duplicated')
    report = {'schema': 1, 'type': 'fresh_result_figures', 'status': 'complete', 'results': str(root),
              'stages': list(folders), 'figure_count': len(paths),
              'figures': [path.relative_to(destination).as_posix() for path in paths],
              'input_sha256': inputs.files, 'producer_manifests': sources, 'frozen_result_fallback': False}
    (destination / 'figures_manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return paths
