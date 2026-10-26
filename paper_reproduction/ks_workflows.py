"""Generate and evaluate KS predictions using retained or newly trained weights.

By default physical references are the supplied 100,000-particle simulations.
With ``data_run`` they are read exclusively from a completed ``ks_data.prepare``
output and the supplied ``model_run`` must have trained on that new data identity.
``generate`` writes ``clouds/<model>/sigma<level>.npz`` and a completed
``generation_verification.json``. ``evaluate`` consumes that new generation (or
creates it under ``destination/generated``), then writes ``metrics_per_seed.csv``,
``summary.csv``, ``results/`` and ``verification.json``. ``multistep`` writes all
twenty three-seed two-step clouds under ``clouds/ks3d`` plus the same metric and
summary layout. All summary tables use the adopted plotting column names.

``model_run`` accepts the output of the public training CLI or its ``ks_kpp``
subdirectory. A completed run, the full physical protocol, dataset signature,
architecture, update count, finite tensors and strict state-dictionary shape are
required. New weights need not equal the historical weights. Historical metric
differences are reported as comparisons, never treated as solver failures.

Calls are sequential-process entry points because the historical import aliases
are temporarily scoped by ``experiments``. CUDA is needed for generation; metric
evaluation of an already completed generation requires only the CPU solvers.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from . import ROOT
from .experiments import _evaluation_kernel, _models, _training_kernel, KS_MODELS


SEEDS = (88, 89, 90)
METHODS = ("mf", "dp", "mf_dp")
CHECKPOINTS = ("mf", "mf_init", "dp", "mf_dp")
LEVELS = tuple(range(10, 201, 10))
PARTICLES, BINS = 100000, 16
_DATASETS = {"ks3d": "imported_current_ks3d_20261012", "ks4d": "total1k_20261013_ks4d"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _destination(destination, *read_only):
    path = Path(destination).expanduser().resolve()
    root = ROOT.resolve()
    protected = [root / name for name in ("output", "deepparticle", "paper_reproduction", "figures", ".git")]
    protected += [Path(value).resolve() for value in read_only if value is not None]
    if root.is_relative_to(path) or any(path.is_relative_to(base) or base.is_relative_to(path) for base in protected):
        raise ValueError(f"Output overlaps a protected source/input directory: {path}")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {path}")
    return path


def _workers(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("workers must be a positive integer")
    return value


def _within(path, root):
    path, root = Path(path).resolve(strict=True), Path(root).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Input path escapes its declared root: {path}")
    return path


def _training_run(model_run, selected):
    if model_run is None:
        return None, None
    supplied = Path(model_run).expanduser().resolve(strict=True)
    candidates = [path for path in (supplied, supplied / "ks_kpp")
                  if (path / "training_replay_verification.json").is_file()]
    if len(candidates) != 1:
        raise ValueError("model_run must identify one completed public training run (CLI root or ks_kpp directory)")
    run = _within(candidates[0], supplied)
    verification = _json(_within(run / "training_replay_verification.json", run))
    request = _json(_within(run / "replay_request.json", run))
    if verification.get("status") not in ("passed", "completed"):
        raise ValueError("The requested training run is not complete")
    if not set(selected).issubset(request.get("models", [])) or set(request.get("seeds", [])) != set(SEEDS):
        raise ValueError("The training run does not cover the requested KS models and seeds")
    checks = verification.get("checks", [])
    keys = [(row.get("model"), row.get("seed"), row.get("method")) for row in checks]
    expected = {(name, seed, method) for name in selected for seed in SEEDS for method in CHECKPOINTS}
    if len(keys) != len(set(keys)) or not expected.issubset(keys):
        raise ValueError("The training completion record omits or duplicates selected checkpoints")
    request["_completion_checks"] = {key: row for key, row in zip(keys, checks)}
    return run, request


def _protocol(r, name, data_run=None):
    if data_run is not None:
        from .ks_data import dataset_identity
        _, protocol, manifest = dataset_identity(name, data_run)
        return protocol, manifest["signature"]
    p = r.protocol(name)
    manifest = _json(_within(ROOT / "deepparticle/data" / _DATASETS[name] / "manifest.json", ROOT))
    config = _json(_within(r.BASE / name / "run/config.json", ROOT))
    if (config.get("status") != "complete" or p.metadata() != manifest["protocol"]
            or p.metadata() != config["protocol"] or manifest["signature"] != config["dataset_signature"]):
        raise ValueError(f"Inconsistent retained KS physical protocol or dataset signature: {name}")
    return p, manifest["signature"]


def _architecture(name, method):
    if method in ("mf", "mf_init"):
        return {"kind": "mf", "width": 64, "hidden_layers": 3, "conditioning": "log10(sigma)",
                "condition_mode": "log10_sigma", "profile": "original_tuned"}
    dimension = int(name[-2])
    return {"kind": "position_condition_dp", "dimension": dimension, "width": 64, "hidden_layers": 3,
            "activation": "SELU", "alpha": .1 if method == "dp" else 10., "condition_mode": "log10_sigma",
            "input_order": ["x", "condition"], "input_features": dimension + 1, "head_initialization": "zero",
            "internal_residual_blocks": 0, "layer_norms": 0, "periodic_position_encoding": False,
            "output": "x + alpha * MLP(x,condition)"}


def _load_models(r, name, run=None, request=None, *, device="cuda", data_run=None):
    """Validate real payloads on CPU before moving their models to the device."""
    if data_run is not None and run is None:
        raise ValueError("New physical data require a completed model_run trained on those data")
    p, signature = _protocol(r, name, data_run)
    if run is not None and request.get("datasets", {}).get(name, {}).get("signature") != signature:
        raise ValueError(f"Training request belongs to another physical dataset: {name}")
    adopted = _json(_within(r.BASE / name / "run/checkpoint_manifest.json", ROOT)) if run is None else None
    models, records = {}, []
    for seed in SEEDS:
        models[seed] = {}
        for method in CHECKPOINTS:
            path = (r.checkpoint_path(name, seed, method) if run is None else
                    run / "training_replay" / name / f"seed{seed}" / f"{method}.pt")
            path = _within(path, ROOT if run is None else run)
            digest = _sha(path)
            if run is not None:
                completed = request["_completion_checks"][name, seed, method]
                completed_hash = completed.get("checkpoint_sha256", completed.get("sha256"))
                if completed_hash is not None and completed_hash != digest:
                    raise ValueError(f"Checkpoint changed after the training completion record: {path}")
                if completed.get("file") is not None and _within(run / completed["file"].replace("\\", "/"), run) != path:
                    raise ValueError(f"Training completion record identifies another checkpoint file: {path}")
            if run is None:
                if method in ("mf", "mf_init"):
                    expected = adopted[str(seed)][method]["sha256"]
                else:
                    record = _json(_within(path.parent / "record.json", ROOT))
                    if (record["model"], record["seed"], record["method"], record["variant"]) != (name, seed, method, "dplus1"):
                        raise ValueError(f"Retained DP training record mismatch: {path}")
                    expected = record["checkpoint_sha256"]
                if digest != expected:
                    raise ValueError(f"Retained checkpoint hash mismatch: {path}")
            payload = r.torch.load(path, map_location="cpu", weights_only=True)
            step = 500 if method in ("mf_init", "mf_dp") else 1000
            architecture = _architecture(name, method)
            if (payload["protocol"] != p.metadata() or payload["dataset_signature"] != signature
                    or payload["step"] != step or payload["architecture"] != architecture):
                raise ValueError(f"Checkpoint protocol, architecture, step or data signature mismatch: {path}")
            if any(value.dtype != r.torch.float32 or not r.torch.isfinite(value).all() for value in payload["model"].values()):
                raise ValueError(f"Checkpoint must contain finite float32 weights: {path}")
            model = (r.MeanFlow(p.dimension, 64, "log10_sigma", 3) if method in ("mf", "mf_init") else
                     r.CompactCorrector(p.dimension, alpha=architecture["alpha"], condition_mode="log10_sigma"))
            model.load_state_dict(payload["model"], strict=True)
            models[seed][method] = model.to(device).eval().requires_grad_(False)
            records.append({"model": name, "seed": seed, "method": method, "file": str(path),
                            "sha256": digest, "step": step, "dataset_signature": signature,
                            "architecture": architecture, "protocol": p.metadata()})
    return p, models, records


def _kernel(r):
    if (r.N, r.BINS, tuple(r.LEVELS), tuple(r.SEEDS), tuple(r.METHODS)) != (PARTICLES, BINS, LEVELS, SEEDS, METHODS):
        raise ValueError("The adopted numerical kernel has changed its KS evaluation protocol")
    r.setup()


def _input_hashes():
    records = _json(ROOT / "paper_reproduction/retained_data.json")["files"]
    return {record["path"].replace("\\", "/"): record["sha256"] for record in records}


def _reference_source(r, name, level, p, hashes, data_run=None):
    if data_run is not None:
        from .ks_data import reference_inputs, dataset_identity
        path, record = reference_inputs(name, level, data_run)
        _, _, dataset = dataset_identity(name, data_run)
        if record["protocol"] != p.metadata() or record["dataset_signature"] != dataset["signature"]:
            raise ValueError("New KS reference belongs to another physical protocol or training dataset")
        with r.np.load(path, allow_pickle=False) as values:
            reference, source = values["reference"].copy(), values["source"].copy()
            metadata = json.loads(values["meta"].item())
        if metadata["protocol"] != p.metadata() or metadata["dataset_signature"] != dataset["signature"]:
            raise ValueError("New KS reference metadata differs from its prepared identity")
        for value in (reference, source):
            if value.shape != (PARTICLES, p.dimension) or value.dtype != r.np.float32 or not r.np.isfinite(value).all():
                raise ValueError("New KS reference/source must contain 100000 finite float32 particles")
        return reference, source, {"reference_file": str(path), "reference_sha256": record["sha256"],
                                  "source_file": str(path), "source_sha256": record["sha256"],
                                  "physical_data_signature": dataset["signature"]}
    path = _within(r.CLOUDS_ROOT / name / f"sigma{level}.npz", ROOT)
    metadata = _json(_within(path.with_suffix(".json"), ROOT))
    if _sha(path) != metadata["sha256"] or metadata["particles"] != PARTICLES:
        raise ValueError(f"Invalid retained reference cloud: {path}")
    with r.np.load(path, allow_pickle=False) as values:
        reference = values["reference"].copy()
    if reference.shape != (PARTICLES, p.dimension) or not r.np.isfinite(reference).all():
        raise ValueError(f"Unexpected retained reference shape: {path}")
    source_info = {}
    if name == "ks3d":
        source_path = _within(ROOT / f"output/particle_count100k_budget_comparison_20261009/particles/sigma{level}_inputs.npz", ROOT)
        source_digest = _sha(source_path)
        if hashes.get(source_path.relative_to(ROOT).as_posix()) != source_digest:
            raise ValueError(f"Retained KS source hash mismatch: {source_path}")
        with r.np.load(source_path, allow_pickle=False) as values:
            source = values["source"].copy()
            r.np.testing.assert_array_equal(reference, values["reference"])
        source_info = {"source_file": str(source_path), "source_sha256": source_digest}
    else:
        if metadata["initial_seed"] != 10999 or metadata["physical_protocol"] != p.metadata():
            raise ValueError("The retained KS4D reference has another source seed or physical protocol")
        source = r.unit_ball(PARTICLES, 4, r.np.random.default_rng(10999)).astype(r.np.float32)
        source_info = {"source_generator": "particle_pipeline.physics.unit_ball", "source_seed": 10999}
    if source.shape != (PARTICLES, p.dimension) or source.dtype != r.np.float32 or not r.np.isfinite(source).all():
        raise ValueError("Invalid KS inference source")
    return reference, source, {"reference_file": str(path), "reference_sha256": metadata["sha256"], **source_info}


def _save_cloud(r, destination, name, level, clouds, metadata):
    if any(value.shape != (PARTICLES, int(name[-2])) or value.dtype != r.np.float32
           or not r.np.isfinite(value).all() for value in clouds.values()):
        raise ValueError("Generation produced an invalid particle cloud")
    path = destination / "clouds" / name / f"sigma{level}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"schema": "ks_cloud_v1", "model": name, "level": level, "particles": PARTICLES,
                "dimension": int(name[-2]), "bins_per_axis": BINS, **metadata}
    r.np.savez_compressed(path, meta=r.np.asarray(json.dumps(metadata)), **clouds)
    record = {"file": path.relative_to(destination).as_posix(), "sha256": _sha(path), **metadata}
    r.save(path.with_suffix(".json"), record)
    return record


def generate(models: list[str], destination: Path, *, model_run: Path | None = None, data_run: Path | None = None) -> dict:
    """Generate all selected KS accuracy clouds from retained or new references.

    Returns the completed manifest with ``destination``, ``clouds_root`` and
    per-case file/hash records. It is written to ``generation_verification.json``.
    """
    selected = _models(models, KS_MODELS)
    run, request = _training_run(model_run, selected)
    destination = _destination(destination, run, data_run)
    with _evaluation_kernel() as r:
        _kernel(r)
        if not r.torch.cuda.is_available():
            raise RuntimeError("KS cloud generation requires CUDA")
        hashes = _input_hashes() if data_run is None else {}
        destination.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result = {"schema": "ks_generation_v1", "status": "running", "models": selected,
                  "destination": str(destination), "clouds_root": "clouds",
                  "particles": PARTICLES, "bins_per_axis": BINS, "seeds": list(SEEDS), "levels": list(LEVELS),
                  "methods": list(METHODS), "model_run": str(run) if run is not None else None,
                  "data_run": str(Path(data_run).resolve()) if data_run is not None else None,
                  "physical_data_manifest_sha256": _sha(Path(data_run) / "manifest.json") if data_run is not None else None,
                  "physical_simulation_executed": False, "checkpoints": [], "cases": []}
        r.save(destination / "generation_request.json", result)
        for name in selected:
            p, loaded, checkpoints = _load_models(r, name, run, request, data_run=data_run)
            result["checkpoints"].extend(checkpoints)
            for level in LEVELS:
                t = time.perf_counter()
                reference, source, identity = _reference_source(r, name, level, p, hashes, data_run)
                predictions = r.predict(source, level, p, loaded)
                expected = {f"seed{seed}_{method}" for seed in SEEDS for method in METHODS}
                if set(predictions) != expected:
                    raise ValueError("KS prediction omitted a model seed or method")
                record = _save_cloud(r, destination, name, level, {"reference": reference, **predictions},
                                     {"protocol": p.metadata(), **identity, "seconds": time.perf_counter() - t})
                result["cases"].append(record)
                r.save(destination / "generation_progress.json", result)
                print(f"GENERATED {name} sigma={level}: 9 x {PARTICLES} predictions", flush=True)
            del loaded
        result.update(status="passed", case_count=len(result["cases"]), elapsed_seconds=time.perf_counter() - started)
        r.save(destination / "generation_verification.json", result)
    return result


def _generation(r, generated, selected, *, run=None, request=None, data_run=None):
    folder = Path(generated).expanduser().resolve(strict=True)
    manifest = _json(_within(folder / "generation_verification.json", folder))
    if (manifest.get("schema") != "ks_generation_v1" or manifest.get("status") != "passed"
            or not set(selected).issubset(manifest.get("models", []))
            or manifest.get("particles") != PARTICLES or manifest.get("bins_per_axis") != BINS
            or manifest.get("seeds") != list(SEEDS) or manifest.get("levels") != list(LEVELS)
            or manifest.get("methods") != list(METHODS)):
        raise ValueError("generated must be a completed matching KS generation, not a saved-cloud directory")
    _models(manifest["models"], KS_MODELS)
    if (manifest.get("data_run") is None) != (data_run is None):
        raise ValueError("Pass the matching data_run when evaluating a generation made from new physical data")
    if data_run is not None and manifest.get("physical_data_manifest_sha256") != _sha(Path(data_run) / "manifest.json"):
        raise ValueError("Generated clouds were produced from another physical-data preparation")
    identities = {name: _protocol(r, name, data_run) for name in selected}
    cases = manifest.get("cases", [])
    keys = [(record.get("model"), record.get("level")) for record in cases]
    all_expected = {(name, level) for name in manifest["models"] for level in LEVELS}
    if len(keys) != len(set(keys)) or set(keys) != all_expected:
        raise ValueError("The generated manifest is missing or duplicating parameter cases")
    checkpoint_keys = [(v.get("model"), v.get("seed"), v.get("method")) for v in manifest.get("checkpoints", [])]
    if len(checkpoint_keys) != len(set(checkpoint_keys)) or set(checkpoint_keys) != {
            (name, seed, method) for name in manifest["models"] for seed in SEEDS for method in CHECKPOINTS}:
        raise ValueError("The generated manifest does not identify all source checkpoints")
    for record in manifest["checkpoints"]:
        name, method = record["model"], record["method"]
        if name not in selected:
            continue
        protocol, signature = identities[name]
        if (record.get("protocol") != protocol.metadata() or record.get("dataset_signature") != signature
                or record.get("architecture") != _architecture(name, method)
                or record.get("step") != (500 if method in ("mf_init", "mf_dp") else 1000)):
            raise ValueError("Generated checkpoint provenance has a different protocol, architecture, step or signature")
    if run is not None:
        current = []
        for name in selected:
            _, loaded, records = _load_models(r, name, run, request, device="cpu", data_run=data_run)
            current.extend(records)
            del loaded
        recorded = {(v["model"], v["seed"], v["method"]): v["sha256"] for v in manifest["checkpoints"]}
        if any(recorded[v["model"], v["seed"], v["method"]] != v["sha256"] for v in current):
            raise ValueError("The supplied generated clouds use another training run's checkpoint bytes")
    for record in cases:
        if record["model"] not in selected:
            continue
        expected_path = folder / "clouds" / record["model"] / f"sigma{record['level']}.npz"
        path = _within(folder / record["file"], folder)
        if path != expected_path.resolve() or _sha(path) != record["sha256"]:
            raise ValueError("Generated cloud path or SHA256 mismatch")
        metadata = _json(_within(path.with_suffix(".json"), folder))
        if metadata != record:
            raise ValueError("Generated cloud metadata differs from the completion manifest")
        if record.get("protocol") != identities[record["model"]][0].metadata():
            raise ValueError("Generated cloud has a different physical protocol")
        with r.np.load(path, allow_pickle=False) as values:
            expected_keys = {"reference", "meta"} | {f"seed{seed}_{method}" for seed in SEEDS for method in METHODS}
            if set(values.files) != expected_keys:
                raise ValueError("Generated cloud is missing a seed/method array")
            for key in expected_keys - {"meta"}:
                value = values[key]
                if value.shape != (PARTICLES, int(record["model"][-2])) or not r.np.isfinite(value).all():
                    raise ValueError("Generated cloud contains invalid coordinates or particle count")
    return folder, manifest


def _summaries(rows, selected, methods):
    summary = []
    for name in selected:
        for level in LEVELS:
            for method in methods:
                group = [v for v in rows if v["model"] == name and v["level"] == level and v["method"] == method]
                if len(group) != 3 or {v["seed"] for v in group} != set(SEEDS):
                    raise ValueError("A metric group does not contain all three distinct training seeds")
                values = [v["w2"] for v in group]
                summary.append({"model": name, "sigma": level, "method": method,
                                "mean_w2": statistics.mean(values), "sd_w2": statistics.stdev(values),
                                "min_w2": min(values), "max_w2": max(values), "training_seeds": 3,
                                "particles": PARTICLES, "bins_per_axis": BINS})
    return summary


def _certify(rows):
    for row in rows:
        if (row["solver_result_code"] != 1 or not math.isfinite(row["w2"]) or row["w2"] < 0
                or not math.isfinite(row["marginal_residual"]) or row["marginal_residual"] >= 1e-9
                or not math.isfinite(row["primal_dual_gap"]) or row["primal_dual_gap"] >= 1e-8):
            raise RuntimeError("Exact histogram OT failed convergence, marginal or primal-dual certification")


def _finish(r, destination, rows, selected, methods, historical, **details):
    rows.sort(key=lambda row: (row["model"], row["level"], row["seed"], row["method"]))
    expected = {(name, level, seed, method) for name in selected for level in LEVELS for seed in SEEDS for method in methods}
    keys = [(v["model"], v["level"], v["seed"], v["method"]) for v in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("The metric result is not the complete requested experiment")
    _certify(rows)
    summary = _summaries(rows, selected, methods)
    adopted = {(v["model"], int(v.get("level", v.get("sigma"))), int(v["seed"]), v["method"]): float(v["w2"])
               for v in _rows(historical)}
    differences = [{"model": v["model"], "level": v["level"], "seed": v["seed"], "method": v["method"],
                    "w2": v["w2"], "historical_w2": adopted[key], "absolute_difference": abs(v["w2"] - adopted[key])}
                   for v, key in zip(rows, keys)]
    r.table(destination / "metrics_per_seed.csv", rows)
    r.table(destination / "summary.csv", summary)
    r.table(destination / "historical_comparison.csv", differences)
    maximum = max(v["absolute_difference"] for v in differences)
    result = {"status": "passed", "models": selected, "destination": str(destination), "rows": len(rows),
              "cases": len(selected) * len(LEVELS), "particles": PARTICLES, "bins_per_axis": BINS,
              "summary": str(destination / "summary.csv"), "metrics_per_seed": str(destination / "metrics_per_seed.csv"),
              "max_historical_w2_difference": maximum, "historical_values_match": maximum < 1e-8,
              "historical_difference_is_failure": False, "all_mass_retained": True,
              "max_marginal_residual": max(v["marginal_residual"] for v in rows),
              "max_primal_dual_gap": max(v["primal_dual_gap"] for v in rows), **details}
    r.save(destination / "verification.json", result)
    return result


def evaluate(models: list[str], destination: Path, *, model_run: Path | None = None,
             generated: Path | None = None, workers: int = 1, data_run: Path | None = None) -> dict:
    """Evaluate newly generated full KS clouds; generate them first by default.

    ``generated`` must name a completed output of :func:`generate`. No fallback
    to historical predictions occurs. ``verification.json`` records its
    ``generated`` and ``clouds_root`` paths relative to the result directory.
    New-model error
    differences from the paper are informational; failed OT certificates raise.
    """
    selected, workers = _models(models, KS_MODELS), _workers(workers)
    run, request = _training_run(model_run, selected)
    destination = _destination(destination, run, generated, data_run)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if generated is None:
        generated = destination / "generated"
        generate(selected, generated, model_run=run, data_run=data_run)
    with _evaluation_kernel() as r:
        _kernel(r)
        folder, manifest = _generation(r, generated, selected, run=run, request=request, data_run=data_run)
        r.OUT, r.CLOUDS_ROOT, r.STARTED = destination, folder / "clouds", time.perf_counter()
        r.save(destination / "evaluation_request.json", {"schema": "ks_evaluation_v1", "models": selected,
               "generated": str(folder), "generation_sha256": _sha(folder / "generation_verification.json"),
               "workers": workers, "particles": PARTICLES, "bins_per_axis": BINS,
               "physical_simulation_executed": False, "model_run": str(run) if run else manifest["model_run"]})
        rows = []
        for name in selected:
            for level in LEVELS:
                r.evaluate_case(name, level, workers)
                complete = _json(destination / "results" / name / f"sigma{level}" / "complete.json")
                if complete["status"] != "complete":
                    raise RuntimeError("Incomplete exact-OT parameter case")
                rows.extend(complete["rows"])
        return _finish(r, destination, rows, selected, METHODS,
                       ROOT / "output/ks_full_histogram_100k_bins16_20261022/metrics_per_seed.csv",
                       schema="ks_evaluation_v1", generated=Path(os.path.relpath(folder, destination)).as_posix(),
                       clouds_root=Path(os.path.relpath(folder / "clouds", destination)).as_posix(),
                       model_run=str(run) if run else manifest["model_run"], workers=workers,
                       data_run=str(Path(data_run).resolve()) if data_run is not None else None,
                       elapsed_seconds=time.perf_counter() - started, physical_simulation_executed=False)


def multistep(destination: Path, *, model_run: Path | None = None, workers: int = 1, data_run: Path | None = None) -> dict:
    """Generate all 60 two-step MF predictions and recompute their exact OT.

    Uses the selected KS3D MF1000 checkpoint with intervals 1→.5→0, without
    further training. ``clouds/ks3d/sigma160.npz`` supplies all three seeds for
    representative figures; all other amplitudes are retained too. The metric
    table uses ``level`` and its 20-row plotting summary uses ``sigma``.
    """
    workers = _workers(workers)
    run, request = _training_run(model_run, ["ks3d"])
    destination = _destination(destination, run, data_run)
    started = time.perf_counter()
    with _training_kernel() as measure:
        with _evaluation_kernel() as r:
            _kernel(r)
            if not r.torch.cuda.is_available():
                raise RuntimeError("Two-step MF generation requires CUDA")
            p, loaded, checkpoints = _load_models(r, "ks3d", run, request, data_run=data_run)
            hashes = _input_hashes() if data_run is None else {}
            destination.mkdir(parents=True, exist_ok=True)
            request_record = {"schema": "ks_multistep_v1", "status": "running", "model_run": str(run) if run else None,
                              "models": ["ks3d"], "seeds": list(SEEDS), "levels": list(LEVELS), "particles": PARTICLES,
                              "bins_per_axis": BINS, "schedule": [[1., .5], [.5, 0.]], "workers": workers,
                              "checkpoints": checkpoints, "cases": [], "physical_simulation_executed": False}
            request_record["data_run"] = str(Path(data_run).resolve()) if data_run is not None else None
            r.save(destination / "generation_request.json", request_record)
            for level in LEVELS:
                reference, source, identity = _reference_source(r, "ks3d", level, p, hashes, data_run)
                tensor = r.torch.from_numpy(source)
                clouds = {"reference": reference}
                for seed in SEEDS:
                    clouds[f"seed{seed}_mf_two_step"] = measure.sample("mf_two_step", tensor, level, p, loaded[seed]).numpy()
                request_record["cases"].append(_save_cloud(r, destination, "ks3d", level, clouds,
                    {"protocol": p.metadata(), "schedule": [[1., .5], [.5, 0.]], **identity}))
                r.save(destination / "generation_progress.json", request_record)
                print(f"GENERATED ks3d sigma={level}: 3 two-step MF predictions", flush=True)
            del loaded
            request_record["status"] = "passed"
            r.save(destination / "generation_verification.json", request_record)
            rows = []
            for record in request_record["cases"]:
                level = record["level"]
                path = destination / record["file"]
                if _sha(path) != record["sha256"]:
                    raise ValueError("Generated two-step cloud changed before metric evaluation")
                with r.np.load(path, allow_pickle=False) as data:
                    clouds = {key: data[key].copy() for key in data.files if key != "meta"}
                measures, origin, widths = r.measures_for(clouds)
                folder = destination / "results/ks3d" / f"sigma{level}"
                r.save(folder / "grid.json", {"origin": origin.tolist(), "widths": widths.tolist(),
                       "base_bins_per_axis": BINS, "dimension": 3, "reference_particles": PARTICLES,
                       "cloud_sha256": record["sha256"], "occupied_bins": {key: len(value[0]) for key, value in measures.items()}})
                def one(seed):
                    score = r.solve(measures["reference"], measures[f"seed{seed}_mf_two_step"])
                    row = {"model": "ks3d", "level": level, "seed": seed, "method": "mf_two_step", "particles": PARTICLES,
                           "dimension": 3, "bins_per_axis": BINS, "metric": "full_dimensional_exact_histogram_W2", **score}
                    _certify([row])
                    r.save(folder / f"seed{seed}_mf_two_step.json", row)
                    return row
                with ThreadPoolExecutor(max_workers=min(workers, len(SEEDS))) as pool:
                    case_rows = list(pool.map(one, SEEDS))
                rows.extend(case_rows)
                r.save(folder / "complete.json", {"status": "complete", "rows": case_rows, "particles": PARTICLES, "all_mass_retained": True})
                print(f"SOLVED two-step MF sigma={level}: {[round(v['w2'], 8) for v in case_rows]}", flush=True)
            return _finish(r, destination, rows, ["ks3d"], ("mf_two_step",),
                           ROOT / "output/ks_histogram_adoption_20261023/multistep_metrics_per_seed.csv",
                           schema="ks_multistep_v1", generated=".", clouds_root="clouds",
                           model_run=str(run) if run else None, workers=workers, schedule=[[1., .5], [.5, 0.]],
                           data_run=str(Path(data_run).resolve()) if data_run is not None else None,
                           elapsed_seconds=time.perf_counter() - started, physical_simulation_executed=False)
