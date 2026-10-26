"""Fixed-budget BDP extensions using the exact existing five benchmark datasets."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
_dll_handles = []
if os.name == "nt" and (Path(sys.prefix) / "Library/bin").is_dir():
    _dll_handles.append(os.add_dll_directory(str(Path(sys.prefix) / "Library/bin")))

import numpy as np
import torch
import ot
from scipy.spatial.distance import cdist

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PACKAGE = ROOT / "deepparticle"
sys.path.insert(0, str(PACKAGE))
from particle_pipeline import bdp
from particle_pipeline.data import load_dataset, read_npz, write_npz, write_json
from particle_pipeline.engine import csv_write
from particle_pipeline.geometry import cloud_metric, histogram_metrics
from particle_pipeline.protocol import file_hash
from training.core import configure_device

BASE = ROOT / "output/kpp_original_training_total1k_20261014"
HYBRID = ROOT / "output/kpp_original_ks_alpha10_lambda01_total1k_20261014"
KS = ROOT / "output/particle_pipeline_total1k_20261013/aligned"
KS3_INPUTS = ROOT / "output/particle_count100k_budget_comparison_20261009/particles"
KS3_CLOUDS = ROOT / "output/particle_count100k_refresh500_all_budgets_20261009/particles"
EXISTING_BDP = ROOT / "output/bdp_matched_kpp2d_20261015"
NAMES = ("ks2d", "ks3d", "ks4d", "kpp2d", "kpp3d")
SEEDS = (88, 89, 90)
METHODS = ("mf", "dp", "bdp", "mf_dp")
PRIMARY = {"ks2d": "euclidean_histogram", "ks3d": "euclidean_histogram",
           "ks4d": "euclidean_full_cloud", "kpp2d": "periodic_histogram",
           "kpp3d": "periodic_full_cloud"}
spec = importlib.util.spec_from_file_location("bdp_five_uploaded", BASE / "experiment.py")
uploaded = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = uploaded
spec.loader.exec_module(uploaded)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def dataset(name):
    if name.startswith("kpp"):
        path, loader = BASE / "datasets" / name, uploaded.load_dataset
    else:
        path = PACKAGE / "data" / ("imported_current_ks3d_20261012" if name == "ks3d" else f"total1k_20261013_{name}")
        loader = load_dataset
    return path, loader, loader(path)


def baseline_path(name):
    return ((ROOT / "output/kpp_original_ks_total2k_20261014/comparison_metrics_per_seed.csv")
            if name.startswith("kpp") else KS / name / "run/evaluation/metrics_per_seed.csv")


def baseline_rows(name):
    rows = read_csv(baseline_path(name))
    if name.startswith("kpp"):
        rows = [r for r in rows if r["model"] == name and int(r["budget"]) == 1000]
    return [{"protocol": name, "level": float(r["level"]), "parameter": float(r.get("parameter", r.get("kappa", r["level"]))),
             "model_seed": int(r["model_seed"]), "method": r["method"], "metric": r["metric"],
             "w2": float(r["w2"]), "w2_squared": float(r["w2_squared"]),
             "origin": "preserved_existing", "source_csv": str(baseline_path(name))}
            for r in rows if r["method"] in ("mf", "dp", "mf_dp")]


def preflight():
    OUT.mkdir(exist_ok=True)
    inputs = {str(baseline_path(name)): file_hash(baseline_path(name)) for name in NAMES}
    inputs[str(ROOT / "mainV2.tex")] = file_hash(ROOT / "mainV2.tex")
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared",
        "models": list(NAMES), "new_training_models": [n for n in NAMES if n != "kpp2d"],
        "seeds": list(SEEDS), "steps": 1000, "batch_size": 2000, "width": 40,
        "learning_rate": .001, "cycle_weight": .001, "batch_refresh": 500, "ot_refresh": 50,
        "network": "two independent 3-hidden-layer width-40 tanh MLPs; linear outputs",
        "objective": "forward squared OT + reverse squared OT + 0.001 * squared cycle error",
        "source": "physical initial arrays; KS uniform unit ball, KPP uniform torus",
        "geometry": "Euclidean KS; periodic KPP",
        "selection": "Settings fixed before runs; no test-driven tuning or checkpoint selection",
        "budgets": {"mf": 1000, "dp": 1000, "bdp": 1000, "mf_dp": [500, 500]},
        "comparison_scope": "equal optimizer updates; architectures, OT refreshes and source distributions may differ; not equal wall time",
        "primary_metrics": PRIMARY, "input_sha256": inputs,
        "primary_source": "https://arxiv.org/html/2504.11851v1#S2.SS2",
        "datasets": {}}
    for name in NAMES:
        path, _, (p, manifest, splits) = dataset(name)
        plan["datasets"][name] = {"path": str(path), "signature": manifest["signature"],
            "manifest_sha256": file_hash(path / "manifest.json"), "protocol": p.metadata(),
            "train_levels": [c["level"] for c in splits["train"]], "eval_levels": [c["level"] for c in splits["eval"]],
            "primary_metric": PRIMARY[name], "primary_eval_particles": 100000 if name == "ks3d" else p.n_eval}
        assert len(splits["train"]) == 8 and len(splits["eval"]) == (11 if p.periodic else 20)
        if name != "ks3d":
            config_root = BASE / name / "run" if p.periodic else KS / name / "run"
            assert read_json(config_root / "config.json")["dataset_signature"] == manifest["signature"]
        if p.periodic:
            assert read_json(HYBRID / name / "run/config.json")["dataset_signature"] == manifest["signature"]
    path = OUT / "plan.json"
    if path.exists():
        previous = read_json(path)
        assert previous["input_sha256"] == inputs, "Baseline or manuscript changed since preflight"
        assert previous["datasets"] == plan["datasets"]
    else:
        write_json(path, plan)
    print("PREFLIGHT: five datasets and baseline metrics identified; fixed settings recorded", flush=True)


def train(name):
    path, loader, (p, manifest, _) = dataset(name)
    if name == "kpp2d":
        bdp.read_bdp(EXISTING_BDP / "bdp_run", p, manifest["signature"], "cuda")
        print("kpp2d: retained three verified existing BDP checkpoints", flush=True)
        return
    run = OUT / name / "bdp_run"
    if (run / "config.json").exists() and read_json(run / "config.json")["status"] == "complete":
        bdp.read_bdp(run, p, manifest["signature"], "cuda")
        print(f"{name}: retained three verified completed checkpoints", flush=True)
        return
    bdp.load_dataset = loader
    started = time.perf_counter()
    bdp.train_bdp(path, run, seeds=SEEDS, steps=1000, batch_size=2000, width=40,
                  lr=.001, cycle_weight=.001, batch_refresh=500, ot_refresh=50, device="cuda")
    write_json(OUT / name / "training_complete.json", {"status": "complete", "seconds": time.perf_counter()-started,
        "dataset_signature": manifest["signature"], "new_updates": 3000})


def infer(models, source, level, p):
    result = {}
    for seed, model in models.items():
        chunks = []
        for start in range(0, len(source), 10000):
            x = torch.from_numpy(source[start:start+10000]).to("cuda")
            condition = torch.full((len(x), 1), p.condition(level), device="cuda")
            with torch.no_grad():
                chunks.append(model(x, condition).cpu().numpy())
        cloud = np.concatenate(chunks)
        assert cloud.shape == source.shape and np.isfinite(cloud).all()
        result[f"seed{seed}_bdp"] = cloud
    return result


def anchored_measures(reference, baseline, generated, old_edges):
    """Extend empty boundary bins, preserving every existing edge and bin center."""
    all_clouds = {"reference": reference, **baseline, **generated}
    edges = []
    for axis, edge in enumerate(old_edges):
        width = (edge[-1]-edge[0])/(len(edge)-1)
        low = min(c[:, axis].min() for c in all_clouds.values())
        high = max(c[:, axis].max() for c in all_clouds.values())
        left = max(0, math.ceil((edge[0]-low)/width))
        right = max(0, math.ceil((high-edge[-1])/width))
        # NumPy 2 preserves float32 linspace endpoints in the original KS2D
        # metric. Keep that dtype so inherited centers remain bitwise equal.
        extended = np.concatenate((edge[0]-width*np.arange(left, 0, -1), edge, edge[-1]+width*np.arange(1, right+1))).astype(edge.dtype)
        assert np.array_equal(extended[left:left+len(edge)], edge)
        edges.append(extended)

    def measure(cloud, grid):
        counts = np.histogramdd(cloud, bins=grid)[0]
        assert counts.sum() == len(cloud), "Histogram discarded particles"
        index = np.argwhere(counts > 0)
        centers = [(e[:-1]+e[1:])*.5 for e in grid]
        positions = np.column_stack([centers[d][index[:, d]] for d in range(cloud.shape[1])])
        return positions, counts[tuple(index.T)] / len(cloud)

    measures = {key: measure(cloud, edges) for key, cloud in all_clouds.items()}
    for key, cloud in {"reference": reference, **baseline}.items():
        old = measure(cloud, old_edges)
        assert all(np.array_equal(a, b) for a, b in zip(old, measures[key])), "Inherited histogram measure changed"
    return measures, edges


def exact_histogram_score(first, second):
    value, log = ot.emd2(first[1], second[1], cdist(first[0], second[0], "sqeuclidean"),
                         numItermax=5_000_000, log=True)
    assert log["result_code"] == 1 and math.isfinite(value) and value >= -1e-10
    return {"w2": math.sqrt(max(0., value)), "w2_squared": max(0., float(value))}


def evaluate(name):
    destination = OUT / name / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    if name == "kpp2d":
        rows = [{**r, "protocol": name, "level": float(r["level"]), "parameter": float(r["parameter"]),
                 "model_seed": int(r["model_seed"]), "w2": float(r["w2"]), "w2_squared": float(r["w2_squared"]),
                 "origin": "reused_verified_BDP"} for r in read_csv(EXISTING_BDP / "bdp_metrics_per_seed.csv")]
        csv_write(destination / "bdp_metrics_per_seed.csv", rows)
        write_json(destination / "verification.json", {"status": "complete", "rows": len(rows),
            "source": str(EXISTING_BDP / "verification.json"), "new_training": False})
        return
    _, _, (p, manifest, splits) = dataset(name)
    _, models = bdp.read_bdp(OUT / name / "bdp_run", p, manifest["signature"], "cuda")
    existing = baseline_rows(name)
    lookup = {(r["level"], r["model_seed"], r["method"], r["metric"]): r["w2"] for r in existing}
    rows, checks = [], []
    for index, case in enumerate(splits["eval"]):
        level = float(case["level"])
        print(f"{name}: evaluating {level:g} ({index+1}/{len(splits['eval'])})", flush=True)
        edges = None
        if name == "ks3d":
            with np.load(KS3_INPUTS / f"sigma{int(level)}_inputs.npz", allow_pickle=False) as saved:
                source, reference = saved["source"].copy(), saved["reference"].copy()
            with np.load(KS3_CLOUDS / f"total1000_sigma{int(level)}.npz", allow_pickle=False) as saved:
                baseline = {f"seed{s}_{new}": saved[f"seed{s}_{old}"].copy()
                    for s in SEEDS for old, new in (("MF", "mf"), ("DP", "dp"), ("hybrid2000", "mf_dp"))}
                edges = [saved[f"edges_{a}"].copy() for a in "xyz"]
        else:
            source, reference = case["initial"], case["target"]
            base_root = BASE / name / "run/evaluation" if p.periodic else KS / name / "run/evaluation"
            baseline, meta = read_npz(base_root / f"particles_{index:02d}.npz")
            assert meta["dataset_signature"] == manifest["signature"]
            np.testing.assert_array_equal(baseline.pop("reference"), reference)
            if p.periodic:
                hybrid, hmeta = read_npz(HYBRID / name / f"run/evaluation/particles_{index:02d}.npz")
                assert hmeta["dataset_signature"] == manifest["signature"]
                np.testing.assert_array_equal(hybrid.pop("reference"), reference)
                for seed in SEEDS:
                    baseline[f"seed{seed}_mf_dp"] = hybrid[f"seed{seed}_mf_dp"]
            if name == "ks2d":
                union = np.concatenate([reference, *baseline.values()])
                low, high = union.min(0), union.max(0)
                padding = np.maximum(.05*(high-low), 1e-5)
                edges = [np.linspace(low[d]-padding[d], high[d]+padding[d], 25) for d in range(p.dimension)]
        generated = infer(models, source, level, p)
        scores = {}
        recheck = level in ((10., 150., 200.) if not p.periodic else (2., 3.75, 4.5))
        if edges is not None:
            measures, edges = anchored_measures(reference, baseline, generated, edges)
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda key: (key, exact_histogram_score(measures["reference"], measures[key])), generated))
            scores[PRIMARY[name]] = dict(results)
            if recheck:
                for method in ("mf", "dp", "mf_dp"):
                    key = f"seed88_{method}"
                    value = exact_histogram_score(measures["reference"], measures[key])["w2"]
                    expected = lookup[level, 88, method, PRIMARY[name]]
                    assert math.isclose(value, expected, abs_tol=1e-10, rel_tol=1e-10), (name, level, method, value, expected)
                    checks.append({"level": level, "seed": 88, "method": method, "metric": PRIMARY[name],
                                   "saved_w2": expected, "recomputed_w2": value, "difference": abs(value-expected)})
        if p.periodic:
            projection = (0, 2) if p.dimension == 3 else None
            metric = "periodic_histogram_xz" if projection else "periodic_histogram"
            scores[metric] = histogram_metrics(reference, generated, 72, True, projection, workers=2)
        if name in ("ks4d", "kpp3d"):
            scores[PRIMARY[name]] = {key: cloud_metric(reference, cloud, p.periodic, 2000, 999) for key, cloud in generated.items()}
            if recheck:
                for method in ("mf", "dp", "mf_dp"):
                    value = cloud_metric(reference, baseline[f"seed88_{method}"], p.periodic, 2000, 999)["w2"]
                    expected = lookup[level, 88, method, PRIMARY[name]]
                    assert math.isclose(value, expected, abs_tol=1e-11, rel_tol=1e-11)
                    checks.append({"level": level, "seed": 88, "method": method, "metric": PRIMARY[name],
                                   "saved_w2": expected, "recomputed_w2": value, "difference": abs(value-expected)})
        for metric, values in scores.items():
            for seed in SEEDS:
                value = values[f"seed{seed}_bdp"]
                rows.append({"protocol": name, "level": level, "parameter": p.parameter(level), "model_seed": seed,
                    "method": "bdp", "metric": metric, "w2": value["w2"], "w2_squared": value["w2_squared"],
                    "particles": len(source), "dataset_signature": manifest["signature"], "origin": "new_BDP_training"})
        edge_arrays = {f"edges_{axis}": e for axis, e in enumerate(edges)} if edges is not None else {}
        write_npz(destination / f"particles_{index:02d}.npz", meta={"protocol": p.metadata(), "level": level,
            "dataset_signature": manifest["signature"], "all_mass_retained": True}, reference=reference, **generated, **edge_arrays)
        csv_write(destination / "bdp_metrics_per_seed.csv", rows)
        csv_write(destination / "baseline_recomputed_checks.csv", checks)
        write_json(destination / "progress.json", {"completed": index+1, "total": len(splits["eval"]), "rows": len(rows)})
    expected = len(splits["eval"])*3*(2 if name == "kpp3d" else 1)
    assert len(rows) == expected
    write_json(destination / "verification.json", {"status": "complete", "rows": len(rows), "new_training": True,
        "dataset_signature": manifest["signature"], "baseline_recomputed": len(checks),
        "max_baseline_difference": max(r["difference"] for r in checks), "all_mass_retained": True,
        "fixed_histogram_measures_preserved": edges is not None})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "train", "evaluate", "all"))
    parser.add_argument("--models", nargs="+", choices=NAMES, default=list(NAMES))
    args = parser.parse_args()
    configure_device("cuda")
    torch.set_num_threads(1)
    started = time.perf_counter()
    preflight()
    if args.stage in ("train", "all"):
        for name in args.models:
            train(name)
            write_json(OUT / "progress.json", {"phase": "training", "last_completed": name, "elapsed_seconds": time.perf_counter()-started})
    if args.stage in ("evaluate", "all"):
        for name in args.models:
            evaluate(name)
            write_json(OUT / "progress.json", {"phase": "evaluation", "last_completed": name, "elapsed_seconds": time.perf_counter()-started})


if __name__ == "__main__":
    main()
