"""Cubic-only reruns with retained or independently generated physical data.

train() fits new models in a fresh directory. evaluate() uses the published
checkpoints by default; model_run can point to a completed train() directory.
Pass data_run from cubic_data.prepare() to use new physical simulations.
That mode never reads retained NPZ files or retained model checkpoints.
"""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output/non_kpp_diffusion_sweep_20261020"
OOD = ROOT / "output/non_kpp_cubic_ood_20261023"
METRIC = ROOT / "output/other_histogram_100k_bins16_20261023"
TRAIN_D = (.25, .5, 1., 2.)
LEVELS = (.25, .375, .5, .75, 1., 1.5, 2., 2.5, 3.)
SEEDS = (88, 89, 90)
METHODS = ("mf", "dp", "mf_dp", "mf_init")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _key(d):
    return "D_" + format(d, ".6g").replace(".", "p")


def _model_root(model_run=None):
    """Accept a cubic run itself or the parent directory produced by the CLI."""
    if model_run is None:
        return DATA.resolve()
    requested = Path(model_run).expanduser().resolve(strict=True)
    candidates = [path for path in (requested, requested / "cubic")
                  if (path / "plan.json").is_file() and (path / "runs/cubic/config.json").is_file()]
    if len(candidates) != 1:
        raise ValueError("Choose a cubic train directory containing runs/cubic, or its unambiguous parent")
    return candidates[0].resolve()


def _data_context(data_run):
    if data_run is None:
        return None, None
    from .cubic_data import load
    root = Path(data_run).expanduser().resolve(strict=True)
    return root, load(root)


def _samples(d, *, data_run=None, data_manifest=None):
    if data_run is not None:
        if data_manifest is None:
            data_run, data_manifest = _data_context(data_run)
        records = [row for row in data_manifest['conditions'] if row['D'] == d]
        if len(records) != 1:
            raise ValueError(f'New cubic data do not contain one condition D={d}')
        record = records[0]['files']['samples.npz']
        path = (Path(data_run) / record['path']).resolve(strict=True)
        if not path.is_relative_to(Path(data_run).resolve()) or _hash(path) != record['sha256']:
            raise ValueError(f'New cubic sample path or SHA256 differs: D={d}')
        return path
    path = (DATA if d <= 2 else OOD) / "datasets/cubic" / _key(d) / "samples.npz"
    if d <= 2:
        provenance = _read(METRIC / "results/non_kpp_cubic" /
                           ("level_" + format(d, ".6g").replace(".", "p")) / "inputs.json")["provenance"]
        expected = provenance["dataset_sha256"]
    else:
        expected = _read(OOD / "evaluation" / _key(d) / "complete.json")["provenance"]["samples_sha256"]
    if _hash(path) != expected:
        raise ValueError(f"Saved cubic samples differ from the published evaluation: D={d}")
    return path


def _destination(path, *, model_run=None, data_run=None):
    from .locations import fresh_directory
    destination = fresh_directory(path)
    protected = (DATA, OOD, METRIC) + ((Path(model_run).resolve(),) if model_run is not None else ())
    protected += ((Path(data_run).resolve(),) if data_run is not None else ())
    if any(destination.is_relative_to(p.resolve()) for p in protected):
        raise ValueError("Choose a fresh output directory outside the saved input runs")
    return destination


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _context(model_run=DATA):
    package = str(ROOT / "deepparticle")
    if package not in sys.path:
        sys.path.insert(0, package)
    import numpy as np
    import torch
    from non_kpp import sweep

    original = _read(DATA / "plan.json")
    model_root = _model_root(model_run)
    plan = _read(model_root / "plan.json")
    config = _read(model_root / "runs/cubic/config.json")
    if (plan["training"] != original["training"] or plan["train_D"] != list(TRAIN_D)
            or plan["seeds"] != list(SEEDS) or plan["particles"] != 5_000_000
            or plan["n_eval_per_D"] != 200000 or plan["n_train_per_D"] != 20000):
        raise ValueError("Cubic rerun must use the original publication training protocol")
    args = SimpleNamespace(**plan["training"])
    protocol = sweep.DiffusionProtocol("cubic", plan["particles"])
    if config["protocol"] != protocol.metadata() or config["training"] != plan["training"]:
        raise ValueError("Cubic model protocol or training configuration changed")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The original cubic numerical protocol requires CUDA")
    return np, torch, sweep, plan, config, protocol, args


def _checkpoints(model_root):
    model_root = Path(model_root).resolve()
    hashes = {}
    for seed in SEEDS:
        folder = model_root / "runs/cubic" / f"seed{seed}"
        expected = _read(folder / "complete.json")["sha256"]
        for method in METHODS:
            path = folder / f"{method}.pt"
            actual = _hash(path)
            if actual != expected[method]:
                raise ValueError(f"Cubic checkpoint hash mismatch: {path}")
            hashes[str(path)] = actual
    return hashes


def _compare_checkpoints(model_root, torch):
    """Compare model tensors, excluding optimizer and serialization metadata."""
    records = []
    for seed in SEEDS:
        for method in METHODS:
            relative = Path("runs/cubic") / f"seed{seed}/{method}.pt"
            original = torch.load(DATA / relative, map_location="cpu", weights_only=True)["model"]
            current = torch.load(model_root / relative, map_location="cpu", weights_only=True)["model"]
            if set(original) != set(current):
                raise ValueError(f"Cubic model tensor names differ: seed={seed}, method={method}")
            tensors = []
            for name in sorted(original):
                before, after = original[name], current[name]
                if before.shape != after.shape or before.dtype != after.dtype:
                    raise ValueError(f"Cubic model tensor shape or dtype differs: {seed}, {method}, {name}")
                if not torch.isfinite(before).all() or not torch.isfinite(after).all():
                    raise ValueError(f"Cubic model has nonfinite tensors: {seed}, {method}, {name}")
                difference = float((after.double() - before.double()).abs().max())
                equal = bool(torch.equal(before.contiguous().view(torch.uint8),
                                         after.contiguous().view(torch.uint8)))
                tensors.append({"name": name, "max_absolute_difference": difference, "bitwise_equal": equal})
            records.append({"seed": seed, "method": method, "tensors": tensors,
                            "max_absolute_difference": max(row["max_absolute_difference"] for row in tensors),
                            "bitwise_equal": all(row["bitwise_equal"] for row in tensors)})
    return {"models_compared": len(records), "tensors_compared": sum(len(row["tensors"]) for row in records),
            "model_run": str(model_root), "comparison": "model state tensors only; optimizer and metadata excluded",
            "all_bitwise_equal": all(row["bitwise_equal"] for row in records),
            "max_absolute_difference": max(row["max_absolute_difference"] for row in records), "models": records}


def _independent_checkpoint_record(model_root):
    return {'model_run': str(model_root), 'status': 'not_compared', 'models_compared': 0,
            'tensors_compared': 0, 'all_bitwise_equal': None, 'max_absolute_difference': None, 'models': [],
            'reason': 'Independent physical data run; retained checkpoints are not read.'}


def _validate_new_checkpoints(model_root, protocol, config, args, torch):
    """Validate all newly trained payloads without reading historical weights."""
    from non_kpp.sweep import factory
    from particle_pipeline.models import MeanFlow
    expected_mf = {'kind': 'mf', 'width': args.mf_width, 'hidden_layers': args.mf_hidden_layers,
                   'conditioning': 'protocol.condition(level)', 'condition_mode': 'protocol', 'profile': args.profile}
    for seed in SEEDS:
        for method in METHODS:
            path = model_root / 'runs/cubic' / f'seed{seed}/{method}.pt'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            # Construct validators without changing later CPU random draws.
            with torch.random.fork_rng(devices=[]):
                if method in ('mf', 'mf_init'):
                    model = MeanFlow(3, args.mf_width, 'protocol', args.mf_hidden_layers)
                    expected_architecture = expected_mf
                else:
                    alpha = args.hybrid_alpha if method == 'mf_dp' else args.alpha
                    model = factory(3, False, args.width, 0, alpha)
                    expected_architecture = {'kind': 'dp', 'width': args.width, 'blocks': args.dp_blocks,
                        'alpha': alpha, 'periodic_position_encoding': False,
                        'conditioning': 'protocol.condition(level)', 'profile': args.profile, **model.architecture}
                model.load_state_dict(payload['model'], strict=True)
            if (payload['protocol'] != protocol.metadata() or payload['dataset_signature'] != config['dataset_signature']
                    or payload['step'] != (1000 if method in ('mf_init', 'mf_dp') else 2000)
                    or payload.get('architecture') != expected_architecture
                    or any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in payload['model'].values())):
                raise ValueError(f'New cubic checkpoint metadata or weights differ: seed={seed}, method={method}')


def _compare_metrics(rows):
    """Record scientific differences without requiring a new model to match."""
    def key(row):
        return float(row["level"]), int(row["seed"]), row["method"], row["kind"]
    with (OOD / "metrics_per_seed.csv").open(encoding="utf-8-sig", newline="") as stream:
        historical = {key(row): row for row in csv.DictReader(stream) if float(row["level"]) in LEVELS}
    current = {key(row): row for row in rows}
    expected = {(d, seed, method, "target_error") for d in LEVELS for seed in SEEDS
                for method in ("mf", "dp", "mf_dp")}
    expected |= {(d, 0, "reference", "reference_self") for d in LEVELS}
    if len(rows) != 90 or set(current) != expected or set(historical) != expected:
        raise ValueError("Cubic comparison requires exactly 81 model errors and nine reference diagnostics")
    metrics = ("w2", "w2_squared", "w2_normalized")
    limits = {"marginal_residual": 1e-9, "primal_dual_gap": 1e-8, "dual_violation": 1e-8}
    maxima = {field: 0. for field in metrics}
    diagnostics = {field: 0. for field in limits}
    records = []
    for identity in sorted(expected):
        before, after = historical[identity], current[identity]
        difference = {field: abs(float(after[field]) - float(before[field])) for field in metrics}
        if any(not math.isfinite(value) for value in difference.values()):
            raise ValueError(f"Nonfinite cubic metric: {identity}")
        for field, delta in difference.items():
            maxima[field] = max(maxima[field], delta)
        for field, limit in limits.items():
            value = float(after[field])
            if not math.isfinite(value) or value >= limit:
                raise ValueError(f"Cubic solver certificate failed: {identity}, {field}")
            diagnostics[field] = max(diagnostics[field], value)
        if int(after["solver_result_code"]) != 1:
            raise ValueError(f"Cubic solver did not report optimality: {identity}")
        records.append({"D": identity[0], "seed": identity[1], "method": identity[2], "kind": identity[3],
                        "absolute_differences": difference})
    return {"rows_compared": len(records), "all_primary_metrics_exact": all(value == 0. for value in maxima.values()),
            "maximum_absolute_differences": maxima, "solver_certificates_passed": True,
            "solver_certificate_maxima": diagnostics, "strict_solver_limits": limits,
            "note": "New model errors may differ; solver diagnostics must satisfy the original bounds, not match historical tails.",
            "rows": records}


def train(destination: Path, *, data_run: Path | None = None) -> dict:
    """Train cubic on four retained or independently regenerated datasets."""
    data_root, data_manifest = _data_context(data_run)
    destination = _destination(destination, data_run=data_root)
    np, torch, sweep, plan, config, protocol, args = _context()
    if data_manifest is not None:
        config = {**config, 'dataset_signature': data_manifest['training_signature']}
    cases, hashes = [], {}
    for d in TRAIN_D:
        path = _samples(d, data_run=data_root, data_manifest=data_manifest)
        hashes[str(path)] = _hash(path)
        with np.load(path, allow_pickle=False) as samples:
            prior, target = samples["train_prior"], samples["train_target"]
            if any(a.shape != (20000, 3) or not np.isfinite(a).all() for a in (prior, target)):
                raise ValueError(f"Unexpected cubic training particle arrays: D={d}")
            cases.append({"level": d, "prior": prior / 5., "target": target / 5.})
    destination.mkdir(parents=True, exist_ok=True)
    _save(destination / "plan.json", {**plan, "paper_scope": "cubic only",
          "physical_data_root": str(data_root or DATA), "physical_simulation_reused": True,
          "data_run": str(data_root) if data_root else None,
          "data_manifest_sha256": _hash(data_root / 'manifest.json') if data_root else None,
          "physical_data_source": 'independently_generated' if data_root else 'retained',
          "dataset_signature": config['dataset_signature']})
    _save(destination / "reproduction_inputs.json", {"samples_sha256": hashes,
          "original_config": str(DATA / "runs/cubic/config.json"),
          "training_kernel": "non_kpp.sweep.train", "training_reactions": ["cubic"],
          "data_run": str(data_root) if data_root else None, "dataset_signature": config['dataset_signature']})
    torch.set_num_threads(4)
    sweep.train(destination, protocol, cases, config["dataset_signature"], args, list(SEEDS))
    checkpoints = _checkpoints(destination)
    _validate_new_checkpoints(destination, protocol, config, args, torch)
    comparison = _independent_checkpoint_record(destination) if data_root else _compare_checkpoints(destination, torch)
    _save(destination / "checkpoint_comparison.json", comparison)
    if any(_hash(Path(path)) != value for path, value in hashes.items()):
        raise RuntimeError("Saved physical inputs changed during training")
    result = {"status": "complete", "destination": str(destination), "reaction": "cubic",
              "seeds": list(SEEDS), "training_D": list(TRAIN_D), "physical_simulation_reused": True,
              "saved_physical_inputs_unchanged": True, "checkpoint_sha256": checkpoints,
              "data_run": str(data_root) if data_root else None, "dataset_signature": config['dataset_signature'],
              "retained_NPZ_or_checkpoint_used": data_root is None,
              "checkpoint_comparison": "checkpoint_comparison.json",
              "historical_weights_bitwise_equal": comparison["all_bitwise_equal"],
              "all_weights_bitwise_equal": comparison["all_bitwise_equal"],
              "max_abs": comparison["max_absolute_difference"], "model_weight_comparisons": comparison["models"]}
    _save(destination / "verification.json", result)
    return result


def evaluate(destination: Path, *, model_run: Path | None = None, data_run: Path | None = None) -> dict:
    """Evaluate all nine D values using the selected model's physical data.

    The default models are the frozen published models. Passing model_run
    evaluates models from train(model_run), on the same saved references.
    """
    if data_run is not None and model_run is None:
        raise ValueError('A new cubic data_run requires a completed new model_run')
    model_root = _model_root(model_run)
    published_models = model_root == DATA.resolve()
    model_plan = _read(model_root / 'plan.json')
    recorded_data = model_plan.get('data_run')
    if data_run is None and recorded_data is not None:
        data_run = recorded_data
    data_root, data_manifest = _data_context(data_run)
    if data_root is not None:
        if published_models or recorded_data is None:
            raise ValueError('New cubic data require a model trained through the independent-data workflow')
        if (model_plan.get('data_manifest_sha256') != _hash(data_root / 'manifest.json')
                or model_plan.get('dataset_signature') != data_manifest['training_signature']):
            raise ValueError('New cubic data do not match this completed training run')
        completion = _read(model_root / 'verification.json')
        if completion.get('status') != 'complete' or completion.get('dataset_signature') != data_manifest['training_signature']:
            raise ValueError('Independent cubic training has no matching completion record')
    destination = _destination(destination, model_run=model_root, data_run=data_root)
    np, torch, sweep, plan, config, protocol, args = _context(model_root)
    if data_root is not None and config['dataset_signature'] != data_manifest['training_signature']:
        raise ValueError('Cubic training config belongs to another independent dataset')
    samples = {d: _samples(d, data_run=data_root, data_manifest=data_manifest) for d in LEVELS}
    input_hashes = {str(path): _hash(path) for path in samples.values()}
    checkpoint_hashes = _checkpoints(model_root)
    if data_root is not None and completion.get('checkpoint_sha256') != checkpoint_hashes:
        raise ValueError('Cubic checkpoint bytes changed after independent training completion')
    if data_root is None:
        _checkpoints(DATA)
    _validate_new_checkpoints(model_root, protocol, config, args, torch)
    checkpoint_comparison = _independent_checkpoint_record(model_root) if data_root else _compare_checkpoints(model_root, torch)
    from non_kpp.experiment import read_models

    # Import the established evaluator without invoking its original main().
    # Only its output root is rebound; all physical input paths above remain
    # anchored to DATA/OOD. Its exact 100k/16-bin histogram/OT kernel is reused.
    source = OOD / "run_ood.py"
    spec = importlib.util.spec_from_file_location("paper_cubic_evaluation_kernel", source)
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    kernel.OUT = destination
    kernel.SOURCE = model_root
    kernel.checkpoints = lambda: _checkpoints(model_root)
    metric = kernel.metric_module()
    models = {seed: read_models(model_root / "runs/cubic", seed, args, config["dataset_signature"])
              for seed in SEEDS}
    for seed in SEEDS:
        for method in METHODS:
            payload = torch.load(model_root / "runs/cubic" / f"seed{seed}/{method}.pt",
                                 map_location="cpu", weights_only=True)
            if payload["protocol"] != protocol.metadata() or payload["step"] != (1000 if method in ("mf_init", "mf_dp") else 2000):
                raise ValueError(f"Unexpected cubic checkpoint metadata: seed {seed}, {method}")
    si = metric.nested_indices(200000, 100000, 7000, 700000)
    ti = metric.nested_indices(200000, 100000, 8000, 800000)
    destination.mkdir(parents=True, exist_ok=True)
    _save(destination / "plan.json", {"reaction": "cubic", "evaluation_D": list(LEVELS),
          "model_run": str(model_root), "published_checkpoints_reused": published_models,
          "physical_simulation_reused": True, "particles_per_distribution": 100000,
          "data_run": str(data_root) if data_root else None,
          "data_manifest_sha256": _hash(data_root / 'manifest.json') if data_root else None,
          "dataset_signature": config['dataset_signature'],
          "bins_per_axis": 16, "normalization_scale": 5., "checkpoint_sha256": checkpoint_hashes,
          "samples_sha256": input_hashes, "metric_source_sha256": _hash(METRIC / "evaluate.py")})
    rows = []
    for d in LEVELS:
        rows.extend(kernel.evaluate(d, samples[d], protocol, models, metric, si, ti))
        inputs_path = destination / "evaluation" / _key(d) / "inputs.json"
        inputs = _read(inputs_path)
        inputs["role"] = "training" if d in TRAIN_D else "extrapolation" if d > 2 else "interpolation"
        _save(inputs_path, inputs)
    rows.sort(key=lambda r: (r["level"], r["kind"], r["seed"], r["method"]))
    summary = []
    for d in LEVELS:
        for method in ("mf", "dp", "mf_dp", "reference"):
            group = [r for r in rows if r["level"] == d and r["method"] == method]
            values = [r["w2_normalized"] for r in group]
            if len(values) != (1 if method == "reference" else 3):
                raise ValueError(f"Incomplete cubic evaluation: D={d}, {method}")
            summary.append({"D": d, "role": "training" if d in TRAIN_D else "extrapolation" if d > 2 else "interpolation",
                            "method": method, "mean": statistics.mean(values), "minimum": min(values), "maximum": max(values),
                            "sd": statistics.stdev(values) if len(values) > 1 else 0., "runs": len(values),
                            "particles_per_distribution": 100000, "bins_per_axis": 16, "normalization_scale": 5.})
    _csv(destination / "metrics_per_seed.csv", rows)
    _csv(destination / "summary.csv", summary)
    _save(destination / "summary.json", summary)
    historical_comparison = _compare_metrics(rows)
    _save(destination / "checkpoint_comparison.json", checkpoint_comparison)
    _save(destination / "historical_comparison.json", historical_comparison)
    if _checkpoints(model_root) != checkpoint_hashes or any(_hash(Path(p)) != h for p, h in input_hashes.items()):
        raise RuntimeError("Saved cubic model or physical inputs changed during evaluation")
    result = {"status": "complete", "destination": str(destination), "evaluation_D": list(LEVELS),
              "target_comparisons": 81, "reference_self_comparisons": 9,
              "model_run": str(model_root), "published_checkpoints_reused": published_models,
              "physical_simulation_reused": True, "saved_inputs_unchanged": True, "new_training": False,
              "data_run": str(data_root) if data_root else None, "dataset_signature": config['dataset_signature'],
              "retained_NPZ_or_checkpoint_used": data_root is None,
              "checkpoint_comparison": "checkpoint_comparison.json",
              "historical_weights_bitwise_equal": checkpoint_comparison["all_bitwise_equal"],
              "historical_comparison": "historical_comparison.json",
              "historical_primary_metrics_exact": historical_comparison["all_primary_metrics_exact"]}
    _save(destination / "verification.json", result)
    _save(destination / "manifest.json", {"schema": 1, "type": "cubic_evaluation", "status": "complete",
          "models": ["cubic"], "model_run": str(model_root), "evaluation_D": list(LEVELS),
          "data_run": str(data_root) if data_root else None, "dataset_signature": config['dataset_signature'],
          "summary": "summary.json", "metrics": "metrics_per_seed.csv", "verification": "verification.json",
          "checkpoint_comparison": "checkpoint_comparison.json", "historical_comparison": "historical_comparison.json"})
    return result
