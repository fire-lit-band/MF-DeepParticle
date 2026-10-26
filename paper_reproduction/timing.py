"""Remeasure GPU-resident inference for the current manuscript methods only.

Sampling reuses ``output/dp_dplus1_manuscript_20261016/measure.py:sample``.
Retained weights use its ``models_for`` loader; completed training runs use the
workflow metadata/hash checks and the same network classes. Direct DP uses
alpha=0.1, refinement uses alpha=10, and ``sample`` wraps periodic direct-DP
inputs before calling the MLP. No adoption script is loaded.

Loading checkpoints, preparing inputs, and checking outputs occur outside the
timed region.  Results depend on the device and current load; this command
produces new measurements rather than replacing the paper's saved timings.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics
import time

from .experiments import ROOT, MODELS, _evaluation_kernel, _models, _training_kernel
from .ks_workflows import _destination, _load_models, _training_run, _within
from .kpp_workflows import _checkpoint_inputs
from .portability import resolve_recorded_path
from .data_runs import bind_datasets, resolve_data_run, stage as data_stage


PARTICLES = 100000
CHUNK = 10000
WARMUPS = 10
REPEATS = 40
SEEDS = (88, 89, 90)
METHODS = ("mf", "dp", "mf_dp")
SELECTED = ROOT / "output/dp_alpha01_refinement_alpha10_20261017"


def _native_source(name, level, case, np, data_run=None):
    """Load exactly the original timing source without unused baseline clouds."""
    if name == "ks3d":
        if data_run is not None:
            from .ks_data import reference_inputs
            path, _ = reference_inputs(name, level, data_stage(data_run, 'ks', [name]))
        else:
            path = ROOT / "output/particle_count100k_budget_comparison_20261009/particles" / f"sigma{int(level)}_inputs.npz"
        with np.load(path, allow_pickle=False) as saved:
            return saved["source"].copy()
    return case["prior"].copy()


def _check_selected_checkpoints(measure, selected, name, seed):
    """Bind every loaded model to the manifest used by the current paper."""
    records = []
    for method in (*METHODS, "mf_init"):
        record = selected[name][str(seed)][method]
        expected = (measure.BASE / name / "run" / f"seed{seed}" / f"{method}.pt"
                    if method in ("mf", "mf_init") else
                    measure.ABLATION / name / "dplus1" / f"seed{seed}" / method / f"{method}.pt")
        supplied = resolve_recorded_path(record["file"], root=ROOT)
        if supplied != expected.resolve(strict=True):
            raise ValueError(f"Selected checkpoint path changed: {name}, {seed}, {method}")
        if measure.e.file_hash(supplied) != record["sha256"]:
            raise ValueError(f"Selected checkpoint bytes changed: {name}, {seed}, {method}")
        records.append({"model": name, "seed": seed, "method": method,
                        "file": supplied.relative_to(ROOT).as_posix(), "sha256": record["sha256"]})
    return records


def _aggregate(rows, models=None):
    names = _models(list(MODELS) if models is None else models, MODELS)
    expected = {(name, method, "gpu_resident", seed) for name in names for seed in SEEDS
                for method in (*METHODS, "mf_two_step") if method != "mf_two_step" or name == "ks3d"}
    grouped = defaultdict(list)
    repeated = defaultdict(list)
    for row in rows:
        key = row["model"], row["method"], row["scope"], int(row["seed"])
        value = float(row["wall_ms"])
        if int(row["particles"]) != PARTICLES or not math.isfinite(value) or value <= 0:
            raise ValueError("Timing rows must contain positive finite durations for 100,000 particles")
        grouped[key].append(value)
        repeated[key].append(int(row["repeat"]))
    if (set(grouped) != expected or len(rows) != len(expected) * REPEATS
            or any(len(values) != REPEATS for values in grouped.values())
            or any(set(values) != set(range(REPEATS)) for values in repeated.values())):
        raise ValueError(f"Expected 40 distinct measurements for each of {len(expected)} selected model/method/seed cases")
    per_seed = [dict(zip(("model", "method", "scope", "seed"), key)) |
                {"median_ms": statistics.median(values), "repeats": len(values)}
                for key, values in sorted(grouped.items())]
    groups = defaultdict(list)
    for row in per_seed:
        groups[row["model"], row["method"]].append(row)
    summary = []
    for (model, method), records in sorted(groups.items()):
        if {row["seed"] for row in records} != set(SEEDS):
            raise ValueError(f"Missing training seed in inference measurements: {model}, {method}")
        values = [row["median_ms"] for row in records]
        summary.append({"model": model, "method": method, "scope": "gpu_resident",
                        "particles": PARTICLES, "seeds": len(values), "repeats_per_seed": REPEATS,
                        "gpu_resident_mean": statistics.mean(values),
                        "gpu_resident_sd": statistics.stdev(values)})
    if len(summary) != len(expected) // len(SEEDS):
        raise ValueError("Incomplete selected model/method timing summaries")
    return per_seed, summary


def _training_statistics(path, names):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        records = [row for row in csv.DictReader(stream) if row["model"] in names and row["method"] in METHODS]
    keys = [(row["model"], row["method"], int(row["seed"])) for row in records]
    expected = {(name, method, seed) for name in names for method in METHODS for seed in SEEDS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Training times must cover the three distinct seeds for every selected model/method")
    stats = {}
    for name in names:
        for method in METHODS:
            values = [float(row["training_seconds"]) for row in records if row["model"] == name and row["method"] == method]
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("Invalid completed-training duration")
            stats[name, method] = statistics.mean(values), statistics.stdev(values)
    return stats, records


def _run_models(measure, name, protocol, signature, run, request, *, device="cuda", data_run=None):
    if name.startswith("ks"):
        with _evaluation_kernel() as kernel:
            loaded_protocol, networks, records = _load_models(kernel, name, run, request, device=device,
                                                              data_run=data_stage(data_run, 'ks', [name]))
        if loaded_protocol.metadata() != protocol.metadata():
            raise ValueError("Timing and checkpoint loaders resolved different KS protocols")
    else:
        networks, records = _checkpoint_inputs(measure, name, protocol, signature, run, device=device)
    # Also cover completed older runs that have per-check hashes but no separate
    # checkpoint_manifest.json. Different weights are valid; changed bytes after
    # completion are not.
    for record in records:
        completed = request["_completion_checks"][name, record["seed"], record["method"]]
        digest = completed.get("checkpoint_sha256", completed.get("sha256"))
        if digest is not None and digest != record["sha256"]:
            raise ValueError("Timing checkpoint changed after training completion")
        if completed.get("file") is not None:
            expected = _within(run / completed["file"].replace("\\", "/"), run)
            if expected != Path(record["file"]).resolve(strict=True):
                raise ValueError("Training completion identifies another timing checkpoint")
    return networks, records


def benchmark(destination: Path, *, model_run: Path | None = None, models: list[str] | None = None, data_run=None) -> dict:
    """Write fresh raw, per-seed, summary and verification files to an empty folder.

    Uses 100,000 GPU-resident particles in chunks of 10,000; ten warmups per
    method, forty synchronized wall-time measurements in shuffled method order,
    and each of the original training seeds 88, 89 and 90.  No model training,
    physical simulation, metric calculation, or manuscript editing is run.

    ``models`` selects any subset of the four KS/KPP benchmarks. ``model_run``
    selects a completed public training root or its ``ks_kpp`` directory;
    otherwise the retained selected weights are used. New weights need not
    reproduce historical KS3D predictions. The additional ``runtime_summary.csv``
    combines these fresh inference measurements with that run's training times
    (or the retained training times when model_run is omitted).
    """
    names = _models(list(MODELS) if models is None else models, MODELS)
    data_run = resolve_data_run(data_run, model_run)
    if data_run is not None and model_run is None:
        raise ValueError('Timing on new data requires its completed training run.')
    run, request = _training_run(model_run, names)
    destination = _destination(destination, run)
    training_path = _within((run if run is not None else SELECTED) / "training_per_seed.csv", run if run is not None else ROOT)
    training_statistics, training_records = _training_statistics(training_path, names)
    with _training_kernel() as measure, bind_datasets(measure.e, data_run):
        e, torch, np = measure.e, measure.torch, measure.np
        if not torch.cuda.is_available():
            raise RuntimeError("GPU-resident inference timing requires CUDA")
        e.configure_device("cuda")
        torch.set_num_threads(1)
        selected = json.loads((SELECTED / "checkpoint_manifest.json").read_text(encoding="utf-8")) if run is None else None
        destination.mkdir(parents=True, exist_ok=True)
        rows, identities, checkpoint_records = [], [], []
        order_rng = random.Random(20261016)
        for name in names:
            _, _, (protocol, manifest, splits) = e.dataset(name)
            level = 4.5 if protocol.periodic else 160.
            index, case = next((i, case) for i, case in enumerate(splits["eval"]) if case["level"] == level)
            native = _native_source(name, level, case, np, data_run)
            source = torch.from_numpy(np.tile(native, (math.ceil(PARTICLES / len(native)), 1))[:PARTICLES].copy())
            if source.shape != (PARTICLES, protocol.dimension):
                raise ValueError(f"Unexpected inference source shape for {name}")
            methods = list(METHODS) + (["mf_two_step"] if name == "ks3d" else [])
            new_networks = None
            if run is not None:
                new_networks, new_records = _run_models(measure, name, protocol, manifest["signature"], run, request, data_run=data_run)
                checkpoint_records.extend(new_records)
            for seed in SEEDS:
                if run is None:
                    checkpoint_records.extend(_check_selected_checkpoints(measure, selected, name, seed))
                    networks = measure.models_for(name, seed, protocol, manifest["signature"])
                else:
                    networks = new_networks[seed]
                if networks["dp"].alpha != .1 or networks["mf_dp"].alpha != 10.:
                    raise ValueError(f"Selected DP scales changed: {name}, seed {seed}")
                device_sources = {method: source.cuda() for method in methods}
                for method in methods:
                    cpu_output = measure.sample(method, source, level, protocol, networks)
                    resident_output = measure.sample(method, device_sources[method], level, protocol, networks,
                                                     resident=True).cpu()
                    if not torch.isfinite(cpu_output).all() or not torch.equal(cpu_output, resident_output):
                        raise ValueError(f"CPU-input and GPU-input inference differ: {name}, seed {seed}, {method}")
                    if name == "ks3d" and run is None:
                        if method == "mf_two_step":
                            expected_path = SELECTED / "multistep_particles/sigma160.npz"
                        else:
                            expected_path = SELECTED / "benchmarks" / name / "evaluation" / f"particles_{index:02d}.npz"
                        arrays, _ = e.read_npz(expected_path)
                        np.testing.assert_array_equal(cpu_output.numpy(), arrays[f"seed{seed}_{method}"])
                    identities.append({"model": name, "seed": seed, "method": method,
                                       "resident_cpu_bitwise_equal": True,
                                       "saved_ks3d_prediction_checked": name == "ks3d" and run is None})
                del cpu_output, resident_output

                def call(method):
                    return measure.sample(method, device_sources[method], level, protocol, networks, resident=True)

                for _ in range(WARMUPS):
                    for method in methods:
                        call(method)
                e.engine.sync("cuda")
                for repeat in range(REPEATS):
                    order = list(methods)
                    order_rng.shuffle(order)
                    for position, method in enumerate(order):
                        e.engine.sync("cuda")
                        started = time.perf_counter()
                        value = call(method)
                        e.engine.sync("cuda")
                        elapsed = (time.perf_counter() - started) * 1000
                        if not math.isfinite(elapsed) or elapsed <= 0:
                            raise ValueError("Invalid synchronized inference duration")
                        rows.append({"model": name, "level": level, "seed": seed, "particles": PARTICLES,
                                     "scope": "gpu_resident", "method": method, "repeat": repeat,
                                     "order": position, "wall_ms": elapsed})
                e.CSV(destination / "inference_raw.csv", rows)
                del value, networks, device_sources
                print(f"Inference timing complete: {name}, seed {seed}", flush=True)
            del new_networks
        per_seed, summary = _aggregate(rows, names)
        e.CSV(destination / "inference_per_seed.csv", per_seed)
        e.CSV(destination / "inference_summary.csv", summary)
        runtime_summary = []
        for row in summary:
            trained_method = "mf" if row["method"] == "mf_two_step" else row["method"]
            mean, sd = training_statistics[row["model"], trained_method]
            runtime_summary.append({"model": row["model"], "method": row["method"],
                                    "training_mean": mean, "training_sd": sd,
                                    "gpu_resident_mean": row["gpu_resident_mean"], "gpu_resident_sd": row["gpu_resident_sd"]})
        e.CSV(destination / "runtime_summary.csv", runtime_summary)
        e.CSV(destination / "training_per_seed.csv", training_records)
        verification = {
            "status": "passed", "model_method_cases": len(summary), "per_seed_rows": len(per_seed),
            "raw_rows": len(rows), "identities": identities, "checkpoints": checkpoint_records,
            "particles": PARTICLES, "chunk": CHUNK, "warmups_per_method": WARMUPS,
            "repeats_per_seed": REPEATS, "method_order_seed": 20261016, "training_seeds": list(SEEDS),
            "scope": "GPU-resident synchronized wall time; loading, RNG, metrics, OT and FK excluded",
            "periodic_direct_DP": "measure.sample wraps spatial source before the direct-DP MLP",
            "dp_alpha": .1, "refinement_alpha": 10., "models": names,
            "model_run": str(run) if run is not None else None,
            "data_run": str(data_run) if data_run is not None else None,
            "weight_source": "completed_training_run" if run is not None else "retained_selected_checkpoints",
            "training_times_source": str(training_path), "training_times_sha256": e.file_hash(training_path),
            "training_times_remeasured_here": False, "runtime_summary": "runtime_summary.csv",
            "source_protocol": "KS3D uses its generated 100k source; other models repeat their generated native source to 100k" if data_run else "KS3D uses its retained 100k source; other models repeat their retained source to 100k for timing",
            "summary_statistic": "mean and sample SD over the three per-seed medians of forty measurements",
            "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "numpy": str(np.__version__),
            "source_code": "output/dp_dplus1_manuscript_20261016/measure.py:sample",
            "checkpoint_loader": "KS/KPP workflow validated loaders" if run is not None else "measure.models_for",
            "source_sha256": e.file_hash(ROOT / "output/dp_dplus1_manuscript_20261016/measure.py"),
            "saved_paper_timings_modified": False,
        }
        e.write_json(destination / "inference_verification.json", verification)
    return {"status": "completed", "output": str(destination), "models": names,
            "raw_rows": len(rows), "per_seed_rows": len(per_seed), "summary_rows": len(summary),
            "runtime_summary": str(destination / "runtime_summary.csv"), "scope": "gpu_resident"}
