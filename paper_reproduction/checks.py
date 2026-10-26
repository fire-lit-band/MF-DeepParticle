"""Verify a standalone numerical reproduction package and its saved inputs."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from . import ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def expected_figures() -> set[str]:
    spec = json.loads((ROOT / 'paper_reproduction/reproduction_spec.json').read_text(encoding='utf-8'))
    figures = set(spec['figure_outputs'])
    if len(figures) != 21 or any(Path(p).is_absolute() or '..' in Path(p).parts or Path(p).suffix != '.pdf' for p in figures):
        raise ValueError('Invalid expected figure output specification')
    return figures


def checkpoint_records():
    manifest = json.loads((ROOT / 'output/dp_alpha01_refinement_alpha10_20261017/checkpoint_manifest.json').read_text(encoding='utf-8'))
    records = []
    for name in ('ks3d', 'ks4d', 'kpp2d', 'kpp3d'):
        for seed in (88, 89, 90):
            for method in ('mf', 'mf_init', 'dp', 'mf_dp'):
                record = manifest[name][str(seed)][method]
                records.append(dict(record, model=name, seed=seed, method=method))
    for seed in (88, 89, 90):
        folder = ROOT / f'output/non_kpp_diffusion_sweep_20261020/runs/cubic/seed{seed}'
        complete = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
        for method in ('mf', 'mf_init', 'dp', 'mf_dp'):
            records.append({'file': (folder / f'{method}.pt').relative_to(ROOT).as_posix(),
                            'sha256': complete['sha256'][method], 'model': 'cubic', 'seed': seed, 'method': method})
    return records


def _forward_check(record):
    import torch
    from particle_pipeline.models import MeanFlow
    from non_kpp.models import CompactCorrector as CubicCorrector

    path = ROOT / record['file'].replace('\\', '/')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    dimension = 3 if record['model'] == 'cubic' else int(record['model'][-2])
    method = record['method']
    architecture = payload['architecture']
    mode = architecture.get('condition_mode', 'protocol')
    if method in ('mf', 'mf_init'):
        model = MeanFlow(dimension, width=64, hidden_layers=3, condition_mode=mode)
    elif record['model'] == 'cubic':
        model = CubicCorrector(dimension, alpha=architecture['alpha'])
    else:
        module_name = '_paper_compact_architecture'
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, ROOT / 'output/dp_dplus1_four_benchmarks_20261016/compact_architecture.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        model = sys.modules[module_name].CompactCorrector(dimension, alpha=architecture['alpha'], condition_mode=mode)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    x = torch.linspace(-.5, .5, 17 * dimension).reshape(17, dimension)
    condition = torch.full((17, 1), .5)
    with torch.no_grad():
        value = (model(x, torch.zeros_like(condition), torch.ones_like(condition), condition)
                 if method in ('mf', 'mf_init') else model(x, condition))
    if value.shape != x.shape or not torch.isfinite(value).all():
        raise ValueError(f'Invalid checkpoint forward pass: {path}')


def audit(*, load_models=False, source_only=False):
    if load_models and source_only:
        raise ValueError('Choose --source-only or --load-models; source-only opens no retained models.')
    figures = expected_figures()
    if len(figures) != 21:
        raise ValueError(f'Expected 21 paper figures; found {len(figures)}')
    source_provenance = json.loads((ROOT / 'paper_reproduction/source_provenance.json').read_text(encoding='utf-8'))
    for rel, provenance in source_provenance.items():
        path = (ROOT / rel).resolve(strict=True)
        if not path.is_relative_to(ROOT) or sha256(path) != provenance['published_sha256']:
            raise ValueError(f'Published physical source changed: {rel}')
    code_manifest = ROOT / 'paper_reproduction/retained_code.json'
    code_count = 0
    if code_manifest.is_file():
        retained = json.loads(code_manifest.read_text(encoding='utf-8'))
        for rel in retained['code_files']:
            path = ROOT / rel
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=rel)
            code_count += 1
    if source_only:
        return {'status': 'passed', 'scope': 'source and physical-source identity checks only',
                'reference_documents_required': False, 'retained_code_syntax_checked': code_count,
                'retained_binary_files_read': 0, 'physical_simulation_executed': False,
                'model_training_executed': False, 'numerical_results_verified': False}
    from .runtime import audit as audit_runtime
    from .cubic import _values
    data_manifest = ROOT / 'paper_reproduction/retained_data.json'
    data_count = data_bytes = 0
    if data_manifest.is_file():
        retained_data = json.loads(data_manifest.read_text(encoding='utf-8'))
        for record in retained_data['files']:
            if not (ROOT / record['path']).is_file():
                raise FileNotFoundError(f"Missing input {record['path']}. Download and extract the numerical data archive at the repository root; see README.md.")
            path = (ROOT / record['path']).resolve(strict=True)
            if not path.is_relative_to(ROOT) or path.stat().st_size != record['bytes']:
                raise ValueError(f'Invalid retained data file: {record["path"]}')
            if sha256(path) != record['sha256']:
                raise ValueError(f'Retained data hash mismatch: {record["path"]}')
            data_count += 1
            data_bytes += record['bytes']
    records = checkpoint_records()
    for record in records:
        path = ROOT / record['file'].replace('\\', '/')
        if sha256(path) != record['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {path}')
        if load_models:
            _forward_check(record)
    return {'status': 'passed', 'reference_documents_required': False, 'expected_figure_outputs': len(figures),
            'retained_code_syntax_checked': code_count, 'checkpoint_hashes_verified': len(records),
            'retained_data_hashes_verified': data_count, 'retained_data_bytes_verified': data_bytes,
            'checkpoint_forward_passes': len(records) if load_models else 0,
            'cubic_curve_points_verified': len(_values()), 'runtime': audit_runtime(),
            'full_training_rerun': False, 'full_physical_simulation_rerun': False}

