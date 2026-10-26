"""Verify saved timing statistics against raw measurements and a numerical specification."""
from pathlib import Path
import csv
import math
import json
import statistics


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output/dp_alpha01_refinement_alpha10_20261017"
PROBLEMS = {"3D KS": "ks3d", "4D KS": "ks4d", "2D KPP": "kpp2d", "3D KPP": "kpp3d"}
METHODS = {"MF": "mf", "DP": "dp", "DP refinement": "mf_dp", "MF (2 steps)": "mf_two_step"}
FIELDS = ("training_mean", "training_sd", "gpu_resident_mean", "gpu_resident_sd")


def _read(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit() -> dict:
    """Check thirteen selected timing results without any document files."""
    source = _read("runtime_summary.csv")
    summary = {(r["model"], r["method"]): r for r in source if r["method"] != "bdp"}
    expected = {(model, method) for model in PROBLEMS.values() for method in ("mf", "dp", "mf_dp")}
    expected.add(("ks3d", "mf_two_step"))
    if len(summary) != len(expected) or set(summary) != expected:
        raise ValueError("Runtime summaries do not cover the selected experiment methods")
    training, inference = _read("training_per_seed.csv"), _read("inference_per_seed.csv")
    raw = _read("inference_raw.csv")
    for key, row in summary.items():
        model, method = key
        trained_method = "mf" if method == "mf_two_step" else method
        train_rows = [r for r in training if r["model"] == model and r["method"] == trained_method]
        infer_rows = [r for r in inference if r["model"] == model and r["method"] == method and r["scope"] == "gpu_resident"]
        for group in (train_rows, infer_rows):
            if len(group) != 3 or {int(r["seed"]) for r in group} != {88, 89, 90}:
                raise ValueError(f"Runtime summary requires three original seeds: {key}")
        for record in infer_rows:
            repeated = [r for r in raw if r["model"] == model and r["method"] == method
                        and r["scope"] == "gpu_resident" and r["seed"] == record["seed"]]
            if len(repeated) != 40 or {int(r["repeat"]) for r in repeated} != set(range(40)):
                raise ValueError(f"Inference median requires 40 measurements: {key}, seed {record['seed']}")
            median = statistics.median(float(r["wall_ms"]) for r in repeated)
            if not math.isclose(median, float(record["median_ms"]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Inference median differs from raw measurements: {key}")
        train_values = [float(r["training_seconds"]) for r in train_rows]
        infer_values = [float(r["median_ms"]) for r in infer_rows]
        values = (statistics.mean(train_values), statistics.stdev(train_values),
                  statistics.mean(infer_values), statistics.stdev(infer_values))
        for field, actual in zip(FIELDS, values):
            if not math.isclose(float(row[field]), actual, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Runtime aggregate differs from per-seed records: {key}, {field}")
    spec = json.loads((ROOT / 'paper_reproduction/reproduction_spec.json').read_text(encoding='utf-8'))
    reference = {(row['model'], row['method']): row for row in spec['runtime_reference']}
    if len(reference) != len(spec['runtime_reference']) or set(reference) != expected:
        raise ValueError('Expected thirteen unique numerical runtime references')
    checked = []
    for key in sorted(reference):
        row = reference[key]
        wanted = (*row['training_seconds'], *row['inference_ms'])
        actual = tuple(f"{float(summary[key][field]):.2f}" for field in FIELDS)
        if actual != wanted:
            raise ValueError(f"Numerical timing reference differs from CSV: {key}")
        checked.append({'model': key[0], 'method': key[1],
                        'training_seconds': list(actual[:2]), 'inference_ms': list(actual[2:])})
    return {"status": "passed", "reference_rows": len(checked), "distinct_results": len(summary),
            "source_summary_rows": len(source), "excluded_method": "bdp", "rows": checked,
            "training_and_inference_aggregates_verified": True, "raw_inference_medians_verified": True,
            "source": str(DATA / "runtime_summary.csv"),
            "training_replay_code": "output/dp_dplus1_manuscript_20261016/measure.py:training",
            "selected_inference_code": "paper_reproduction/timing.py:benchmark"}
