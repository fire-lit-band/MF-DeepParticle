"""Train from retained or independently generated data using original kernels.

New physical data runs require no retained binary data or model weights.
Legacy metric helpers below remain available for retained-result verification;
public fresh evaluation is implemented by the family workflow modules.

The numerical kernels are imported from their executed source files. Historical
manuscript-editing, packaging, preflight and report entry points are never run.
These entry points are intended for one sequential process (imports and the
historical training factory temporarily use process-global module state).
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import importlib.util
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("ks3d", "ks4d", "kpp2d", "kpp3d")
KS_MODELS = ("ks3d", "ks4d")
KPP_MODELS = ("kpp2d", "kpp3d")
BASE = ROOT / "output/mf_matched_dp_20261016"
COMPACT = ROOT / "output/dp_dplus1_four_benchmarks_20261016"
MEASURE = ROOT / "output/dp_dplus1_manuscript_20261016/measure.py"
KS_EVALUATION = ROOT / "output/ks_full_histogram_100k_bins16_20261022"
KPP_EVALUATION = ROOT / "output/other_histogram_100k_bins16_20261023"


def _models(models: list[str], allowed: tuple[str, ...]) -> list[str]:
    selected = list(models)
    if not selected or any(name not in allowed for name in selected):
        raise ValueError(f"Choose one or more of: {', '.join(allowed)}")
    if len(set(selected)) != len(selected):
        raise ValueError("Model names must be distinct")
    return selected


def _empty_destination(destination: Path) -> Path:
    from .locations import fresh_directory
    return fresh_directory(destination)


def _load(name: str, source: Path):
    source = source.resolve(strict=True)
    if not source.is_relative_to(ROOT):
        raise ValueError(f"Kernel source escaped the repository: {source}")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load numerical kernel: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _source_environment(extra_paths: list[Path]):
    """Isolate historical short module names and restore import state afterwards."""
    package = ROOT / "deepparticle"
    # Reusing an unrelated module with the same package name could silently
    # change numerical behavior. Require a fresh process in that situation.
    for name, module in list(sys.modules.items()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in ("training", "particle_pipeline", "non_kpp")):
            source = getattr(module, "__file__", None)
            if source and not Path(source).resolve().is_relative_to(package):
                raise RuntimeError(f"Conflicting loaded package {name}; run reproduction in a fresh Python process")
    aliases = (
        "experiment", "matched_architecture", "compact_architecture", "exact_sparse",
        "four_benchmark_baseline", "matched_uploaded_data", "matched_historical_eval", "bdp_five_uploaded",
        "_paper_reproduction_training", "_paper_reproduction_measure", "_paper_reproduction_ks_evaluation",
        "_paper_reproduction_kpp_evaluation",
    )
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in aliases}
    previous_path = sys.path.copy()
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(path) for path in extra_paths] + [str(package)]
        for name in aliases:
            sys.modules.pop(name, None)
        yield
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
        for name, module in saved.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@contextmanager
def _training_kernel():
    """Load the training replay implementation without executing an experiment."""
    with _source_environment([COMPACT, BASE]):
        runner = _load("_paper_reproduction_training", COMPACT / "experiment.py")
        sys.modules["experiment"] = runner
        measure = _load("_paper_reproduction_measure", MEASURE)
        if measure.r is not runner:
            raise ImportError("Historical training module resolved the wrong experiment")
        yield measure


@contextmanager
def _evaluation_kernel():
    """Load the adopted histogram and exact-OT kernels without solving OT."""
    with _source_environment([KS_EVALUATION, COMPACT]):
        _load("exact_sparse", KS_EVALUATION / "exact_sparse.py")
        yield _load("_paper_reproduction_ks_evaluation", KS_EVALUATION / "run_evaluation.py")


@contextmanager
def _kpp_evaluation_kernel():
    """Import KPP histogram kernels without running old non-KPP/report stages."""
    with _source_environment([KPP_EVALUATION, COMPACT]):
        yield _load("_paper_reproduction_kpp_evaluation", KPP_EVALUATION / "evaluate.py")


def train(models: list[str], destination: Path, *, strict_replay: bool = False, data_run=None) -> dict:
    """Train MF1000, DP1000 and MF500+DP500 on selected physical data.

    Only the four body benchmarks are accepted. Seeds 88, 89 and 90 and all
    original optimizer/model settings are fixed by the executed kernels.
    With data_run, new dataset signatures and checkpoint identities are checked
    without opening historical weights. Strict replay applies only to retained
    data, where historical tensors can meaningfully be compared.
    The destination must be new or empty; incomplete runs are preserved.
    """
    from .data_runs import bind_datasets
    if data_run is not None and strict_replay:
        raise ValueError('strict_replay compares retained-data training; omit it when using newly generated data.')
    selected = _models(models, MODELS)
    destination = _empty_destination(destination)
    with _training_kernel() as measure, bind_datasets(measure.e, data_run):
        engine = measure.e
        if not measure.torch.cuda.is_available():
            raise RuntimeError("Training replay requires CUDA and the original numerical environment")
        inputs = {}
        datasets = {}
        # Check all selected inputs before spending time training the first model.
        for name in selected:
            folder, _, (protocol, manifest, splits) = engine.dataset(name)
            datasets[name] = {"path": str(folder), "signature": manifest["signature"],
                              "manifest_sha256": engine.file_hash(folder / "manifest.json")}
            del splits
            if data_run is not None:
                continue
            checkpoint_manifest = engine.js(BASE / name / "run/checkpoint_manifest.json")
            for seed in measure.SEEDS:
                for method in ("mf", "mf_init", "dp", "mf_dp"):
                    if method in ("mf", "mf_init"):
                        path = BASE / name / "run" / f"seed{seed}" / f"{method}.pt"
                        expected = checkpoint_manifest[str(seed)][method]["sha256"]
                    else:
                        path = COMPACT / name / "dplus1" / f"seed{seed}" / method / f"{method}.pt"
                        expected = engine.js(path.parent / "record.json")["checkpoint_sha256"]
                    actual = engine.file_hash(path)
                    if actual != expected:
                        raise ValueError(f"Adopted checkpoint hash changed: {path}")
                    payload = measure.r.read_payload(path, protocol, manifest["signature"])
                    if payload["step"] != (500 if method in ("mf_init", "mf_dp") else 1000):
                        raise ValueError(f"Unexpected checkpoint update count: {path}")
                    inputs[str(path.relative_to(ROOT))] = actual
                    del payload
        engine.configure_device("cuda")
        measure.torch.set_num_threads(1)
        destination.mkdir(parents=True, exist_ok=True)
        measure.OUT = destination
        measure.NAMES = tuple(selected)
        measure.STARTED = time.perf_counter()
        engine.write_json(destination / "replay_request.json", {
            "models": selected, "seeds": list(measure.SEEDS), "datasets": datasets,
            "adopted_checkpoint_sha256": inputs, "settings": vars(engine.args()),
            "physical_data": "newly_generated" if data_run is not None else "retained",
            "data_run": str(Path(data_run).resolve()) if data_run is not None else None,
            "checkpoint_verification": "Finite parameters, protocol, architecture, step, dataset identity and file hash",
            "historical_comparison_performed": data_run is None,
            "strict_replay": strict_replay,
            "python": sys.version, "torch": str(measure.torch.__version__),
        })
        measure.training(strict_replay=strict_replay, compare_retained=data_run is None)
        verification = engine.js(destination / "training_replay_verification.json")
        checks = verification["checks"]
        expected_checks = len(selected) * len(measure.SEEDS) * 4
        if (verification["status"] != "passed" or len(checks) != expected_checks
                or not all(row["finite_parameters"] for row in checks)
                or (strict_replay and not all(row["weights_bitwise_equal"] for row in checks))):
            raise RuntimeError("Training replay did not verify every requested checkpoint")
        checkpoint_manifest = {}
        for row in checks:
            checkpoint_manifest.setdefault(row['model'], {}).setdefault(str(row['seed']), {})[row['method']] = {
                'file': row['file'], 'sha256': row['sha256'],
                'step': 500 if row['method'] in ('mf_init', 'mf_dp') else 1000}
        engine.write_json(destination / 'checkpoint_manifest.json', checkpoint_manifest)
        return {"status": "passed", "models": selected, "destination": str(destination),
                "checkpoint_checks": len(checks),
                "strict_replay": strict_replay,
                "all_weights_bitwise_equal": all(row["weights_bitwise_equal"] for row in checks) if data_run is None else None,
                "historical_comparison_performed": data_run is None,
                "data_run": str(Path(data_run).resolve()) if data_run is not None else None,
                "verification": str(destination / "training_replay_verification.json"),
                "training_per_seed": str(destination / "training_per_seed.csv")}


def evaluate_ks(models: list[str], destination: Path, workers: int = 1) -> dict:
    """Recompute current main KS histogram metrics from retained 100k clouds.

    Uses all 20 amplitudes, three model seeds and MF/direct-DP/refinement for
    each selected KS model. It does not include the separate two-step MF
    ablation. The 3D dense and 4D certified sparse exact-OT kernels are unchanged.
    CUDA inference is unnecessary, although the historical module imports Torch.
    Original cloud hashes are checked; all new metrics and logs go to a new or
    empty destination. Old manuscript hashes and protected-figure checks are
    intentionally excluded because neither is an input to this calculation.
    """
    selected = _models(models, KS_MODELS)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    destination = _empty_destination(destination)
    with _evaluation_kernel() as run:
        if run.N != 100000 or run.BINS != 16 or tuple(run.SEEDS) != (88, 89, 90):
            raise ValueError("The stored KS evaluation kernel is not the adopted protocol")
        cloud_manifest = []
        for name in selected:
            for level in run.LEVELS:
                path = run.CLOUDS_ROOT / name / f"sigma{level}.npz"
                metadata_path = path.with_suffix(".json")
                if not path.is_file() or not metadata_path.is_file():
                    raise FileNotFoundError(f"Retained KS cloud or its metadata is missing: {path}")
                metadata = run.js(metadata_path)
                cloud_manifest.append({"model": name, "sigma": level,
                                       "path": str(path.relative_to(ROOT)), "sha256": metadata["sha256"]})
        destination.mkdir(parents=True, exist_ok=True)
        run.OUT = destination
        run.STARTED = time.perf_counter()
        run.save(destination / "evaluation_request.json", {
            "models": selected, "levels": run.LEVELS, "seeds": list(run.SEEDS),
            "methods": list(run.METHODS), "particles": run.N, "bins_per_axis": run.BINS,
            "workers": workers, "clouds": cloud_manifest,
            "scope": "Recompute exact histogram OT on retained fixed point clouds; no new training or physical simulation",
            "metric_source": str(KS_EVALUATION / "run_evaluation.py"),
            "metric_source_sha256": run.sha(KS_EVALUATION / "run_evaluation.py"),
            "sparse_source_sha256": run.sha(KS_EVALUATION / "exact_sparse.py"),
        })
        for name in selected:
            for level in run.LEVELS:
                # evaluate_case verifies the cloud SHA before loading it and
                # checks retained mass and OT certificates. Its writes use OUT.
                run.evaluate_case(name, level, workers)
        rows = []
        for name in selected:
            for level in run.LEVELS:
                result = run.js(destination / "results" / name / f"sigma{level}" / "complete.json")
                if result["status"] != "complete" or len(result["rows"]) != 9:
                    raise RuntimeError(f"Incomplete histogram evaluation: {name}, sigma={level}")
                rows.extend(result["rows"])
        rows.sort(key=lambda row: (row["model"], row["level"], row["seed"], row["method"]))
        if len(rows) != len(selected) * 20 * 9 or any(
            row["solver_result_code"] != 1 or row["marginal_residual"] >= 1e-9 or row["primal_dual_gap"] >= 1e-8
            for row in rows
        ):
            raise RuntimeError("Histogram OT evaluation failed its full-mass solver checks")
        run.table(destination / "metrics_per_seed.csv", rows)
        summary = []
        for name in selected:
            for level in run.LEVELS:
                for method in run.METHODS:
                    values = [row["w2"] for row in rows if row["model"] == name and row["level"] == level and row["method"] == method]
                    summary.append({"model": name, "sigma": level, "method": method,
                                    "mean_w2": statistics.mean(values), "sd_w2": statistics.stdev(values),
                                    "min_w2": min(values), "max_w2": max(values),
                                    "training_seeds": 3, "particles": run.N, "bins_per_axis": run.BINS})
        run.table(destination / "summary.csv", summary)
        with (KS_EVALUATION / "metrics_per_seed.csv").open(encoding="utf-8-sig", newline="") as stream:
            adopted = {(row["model"], int(row["level"]), int(row["seed"]), row["method"]): float(row["w2"])
                       for row in csv.DictReader(stream)}
        difference = max(abs(row["w2"] - adopted[row["model"], row["level"], row["seed"], row["method"]]) for row in rows)
        verified = difference < 1e-8
        result = {"status": "passed" if verified else "numerical_mismatch", "models": selected,
                  "destination": str(destination), "rows": len(rows), "cases": len(selected) * 20,
                  "max_adopted_w2_difference": difference, "adopted_w2_tolerance": 1e-8,
                  "max_marginal_residual": max(row["marginal_residual"] for row in rows),
                  "max_primal_dual_gap": max(row["primal_dual_gap"] for row in rows),
                  "summary": str(destination / "summary.csv"),
                  "metrics_per_seed": str(destination / "metrics_per_seed.csv")}
        run.save(destination / "verification.json", result)
        if not verified:
            raise RuntimeError(f"Recomputed W2 differs from adopted values by {difference:g}; outputs retained for inspection")
        return result


def evaluate_kpp(models: list[str], destination: Path) -> dict:
    """Recompute adopted KPP W2 on retained 100k reference/generated clouds.

    Accepts kpp2d/kpp3d only, using all 11 levels and seeds 88, 89 and 90.
    The unchanged numerical kernel uses a fixed 16-bin-per-axis torus grid,
    exact periodic quadratic OT and three simultaneous solves. No training,
    inference, FK simulation, Arrhenius calculation or old report stage runs.
    A new or empty destination receives all results. Reference/generated input
    hashes and agreement with the adopted per-seed W2 table are checked.
    """
    selected = _models(models, KPP_MODELS)
    destination = _empty_destination(destination)
    with _kpp_evaluation_kernel() as run:
        if run.N != 100000 or run.BINS != 16 or tuple(run.SEEDS) != (88, 89, 90):
            raise ValueError("The stored KPP evaluation kernel is not the adopted protocol")
        cloud_manifest = []
        for name in selected:
            reference_manifest = run.js(KPP_EVALUATION / "references" / name / "manifest.json")
            if reference_manifest["status"] != "complete":
                raise ValueError(f"Retained KPP references are incomplete: {name}")
            for index, level in enumerate(run.KPP_LEVELS):
                records = [record for record in reference_manifest["outputs"] if float(record["level"]) == level]
                if len(records) != 1:
                    raise ValueError(f"Missing or duplicate KPP reference: {name}, level={level}")
                record = records[0]
                reference = run.reference_path(name, record)
                generated = KPP_EVALUATION / "generated" / name / f"level_{index:02d}.npz"
                level_directory = "level_" + format(level, ".6g").replace(".", "p")
                provenance = run.js(KPP_EVALUATION / "results" / name / level_directory / "inputs.json")["provenance"]
                if run.file_hash(reference) != record["sha256"] or record["sha256"] != provenance["reference_sha256"]:
                    raise ValueError(f"Retained KPP reference changed: {reference}")
                if run.file_hash(generated) != provenance["generated_sha256"]:
                    raise ValueError(f"Retained KPP generated cloud changed: {generated}")
                cloud_manifest.append({"model": name, "level": level,
                                       "reference": str(reference.relative_to(ROOT)),
                                       "reference_sha256": record["sha256"],
                                       "generated": str(generated.relative_to(ROOT)),
                                       "generated_sha256": provenance["generated_sha256"]})
        destination.mkdir(parents=True, exist_ok=True)
        request = {"models": selected, "levels": run.KPP_LEVELS, "seeds": list(run.SEEDS),
                   "methods": list(run.METHODS), "particles": run.N, "bins_per_axis": run.BINS,
                   "clouds": cloud_manifest, "metric_source": str(KPP_EVALUATION / "evaluate.py"),
                   "metric_source_sha256": run.file_hash(KPP_EVALUATION / "evaluate.py"),
                   "scope": "Exact periodic histogram OT on retained fixed clouds; no training or physical simulation"}
        run.write_json(destination / "evaluation_request.json", request)
        # evaluate_kpp uses OUT to locate retained input clouds and calls
        # evaluate_clouds for all writes. Redirect only that numerical callback.
        # Its original make_plan also checks unrelated Arrhenius inputs and old
        # manuscript hashes; the selected input checks above replace that step.
        original_evaluate_clouds = run.evaluate_clouds
        original_make_plan = run.make_plan

        def evaluate_into_destination(*args, **kwargs):
            previous = run.OUT
            try:
                run.OUT = destination
                return original_evaluate_clouds(*args, **kwargs)
            finally:
                run.OUT = previous

        run.evaluate_clouds = evaluate_into_destination
        run.make_plan = lambda: request
        try:
            run.setup()
            for name in selected:
                run.evaluate_kpp(name)
        finally:
            run.evaluate_clouds = original_evaluate_clouds
            run.make_plan = original_make_plan
        rows = []
        for name in selected:
            for level in run.KPP_LEVELS:
                level_directory = "level_" + format(level, ".6g").replace(".", "p")
                complete = run.js(destination / "results" / name / level_directory / "complete.json")
                if complete["status"] != "complete" or len(complete["rows"]) != 9:
                    raise RuntimeError(f"Incomplete KPP histogram evaluation: {name}, level={level}")
                rows.extend(complete["rows"])
        expected = {(name, level, seed, method) for name in selected for level in run.KPP_LEVELS
                    for seed in run.SEEDS for method in run.METHODS}
        actual = {(row["model"], row["level"], row["seed"], row["method"]) for row in rows}
        if actual != expected or len(rows) != len(expected) or any(
            row["kind"] != "target_error" or row["solver_result_code"] != 1
            or row["marginal_residual"] >= 1e-9 or row["primal_dual_gap"] >= 1e-8 or row["dual_violation"] >= 1e-8
            for row in rows
        ):
            raise RuntimeError("KPP histogram OT failed its completeness or solver checks")
        rows.sort(key=lambda row: (row["model"], row["level"], row["seed"], row["method"]))
        run.csv_write(destination / "metrics_per_seed.csv", rows)
        summary = []
        for name in selected:
            for level in run.KPP_LEVELS:
                for method in run.METHODS:
                    values = [row["w2"] for row in rows if row["model"] == name and row["level"] == level and row["method"] == method]
                    summary.append({"model": name, "level": level, "method": method, "kind": "target_error",
                                    "w2_mean": statistics.mean(values), "w2_sd": statistics.stdev(values),
                                    "w2_min": min(values), "w2_max": max(values),
                                    "seeds": 3, "particles": run.N, "bins": run.BINS, "normalization_scale": 1.})
        run.csv_write(destination / "summary.csv", summary)
        with (KPP_EVALUATION / "metrics_per_seed.csv").open(encoding="utf-8-sig", newline="") as stream:
            adopted = {(row["model"], float(row["level"]), int(row["seed"]), row["method"]): float(row["w2"])
                       for row in csv.DictReader(stream) if row["model"] in selected and row["kind"] == "target_error"}
        difference = max(abs(row["w2"] - adopted[row["model"], row["level"], row["seed"], row["method"]]) for row in rows)
        verified = difference < 1e-8
        result = {"status": "passed" if verified else "numerical_mismatch", "models": selected,
                  "destination": str(destination), "rows": len(rows), "cases": len(selected) * 11,
                  "max_adopted_w2_difference": difference, "adopted_w2_tolerance": 1e-8,
                  "max_marginal_residual": max(row["marginal_residual"] for row in rows),
                  "max_primal_dual_gap": max(row["primal_dual_gap"] for row in rows),
                  "max_dual_violation": max(row["dual_violation"] for row in rows),
                  "summary": str(destination / "summary.csv"),
                  "metrics_per_seed": str(destination / "metrics_per_seed.csv")}
        run.write_json(destination / "verification.json", result)
        if not verified:
            raise RuntimeError(f"Recomputed KPP W2 differs from adopted values by {difference:g}; outputs retained for inspection")
        return result
