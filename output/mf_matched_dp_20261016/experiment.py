"""Retrain the four manuscript benchmarks with a capacity-matched DP backbone.

All inputs are read-only. Results, checkpoints and exact measurement records
are written into this new experiment directory. No manuscript edits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import zipfile

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
_dll_handles = []
if os.name == "nt" and (Path(sys.prefix) / "Library/bin").is_dir():
    _dll_handles.append(os.add_dll_directory(str(Path(sys.prefix) / "Library/bin")))

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PACKAGE = ROOT / "deepparticle"
sys.path.insert(0, str(PACKAGE))

import numpy as np
import torch
from particle_pipeline import engine
from particle_pipeline.data import load_dataset, read_npz, write_npz, write_json
from particle_pipeline.geometry import cloud_metric, histogram_metrics
from particle_pipeline.models import MeanFlow, Corrector, generate
from particle_pipeline.protocol import file_hash
from training.core import configure_device
from matched_architecture import MatchedCorrector, describe
from paper_reproduction.portability import protocols_match, resolve_recorded_path


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


uploaded = import_path("matched_uploaded_data", ROOT / "output/kpp_original_training_total1k_20261014/experiment.py")
historical = import_path("matched_historical_eval", ROOT / "output/bdp_five_datasets_20261015/experiment.py")
NAMES = ("ks3d", "ks4d", "kpp2d", "kpp3d")
SEEDS = (88, 89, 90)
METHODS = ("mf", "dp", "mf_dp")
PRIMARY = {"ks3d": "euclidean_histogram", "ks4d": "euclidean_full_cloud",
           "kpp2d": "periodic_histogram", "kpp3d": "periodic_full_cloud"}
CSV = engine.csv_write
STARTED = time.perf_counter()
TIMEOUT = 10800


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tick(**values):
    if time.perf_counter() - STARTED > TIMEOUT:
        raise TimeoutError(f"Hard experiment-stage timeout: {TIMEOUT} seconds")
    write_json(OUT / "progress.json", {"utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.perf_counter()-STARTED, **values})


def dataset(name):
    if name.startswith("kpp"):
        path, loader = ROOT / "output/kpp_original_training_total1k_20261014/datasets" / name, uploaded.load_dataset
    else:
        path = PACKAGE / "data" / ("imported_current_ks3d_20261012" if name == "ks3d" else "total1k_20261013_ks4d")
        loader = load_dataset
    return path, loader, loader(path)


def args():
    return argparse.Namespace(mf_steps=500, dp_steps=500, mf_lr=.001, dp_lr=.0001,
        width=64, mf_width=64, mf_hidden_layers=3, dp_blocks=4, alpha=.1,
        hybrid_alpha=10., residual_weight=.1, profile="original_tuned", batch_size=2000,
        batch_refresh=500, ot_refresh=500, seeds=list(SEEDS), device="cuda")


def preflight():
    plan = {"material_passport": {"origin_skill": "academic-research-suite / experiment-agent",
        "origin_mode": "run + validate", "date": "2026-10-16", "verification_status": "PLANNED",
        "version": "mf-matched-dp-v1"}, "models": list(NAMES), "seeds": list(SEEDS),
        "budgets": {"mf": 1000, "dp": 1000, "mf_dp": [500, 500]},
        "hyperparameters": vars(args()), "architecture": {name: describe(int(name[-2]), .1,
            "protocol" if name.startswith("kpp") else "log10_sigma") for name in NAMES},
        "differences_from_MF": ["independent weights and objective", "DP endpoint inputs t=1,r=0",
            "DP head zero initialized; MF uses default Linear initialization", "DP output adds alpha*MLP",
            "DP retains historical geometric loss, including periodic KPP"],
        "randomness": "same original-tuned seed/profile; DP batches preserve the historical four-block constructor RNG consumption",
        "selection": "fixed final update; no hyperparameter search or test-driven checkpoint selection",
        "evaluation": {"KS3D": "100000 stored source/reference particles; original anchored histogram grid",
            "KS4D": "2000-point full-coordinate exact OT, metric seed999",
            "KPP2D": "20000 particles; 72x72 periodic histogram",
            "KPP3D": "2000-point periodic full-coordinate exact OT, plus 72x72 xz projection",
            "grids": "all 20 KS or 11 KPP test levels; three fixed training seeds"},
        "inference": {"particles": [2000, 10000, 100000], "chunk": 10000,
            "warmups": 10, "repeats": 40, "representative_levels": {"ks3d": 160, "ks4d": 160, "kpp2d": 4.5, "kpp3d": 4.5},
            "scopes": ["gpu_resident", "cpu_to_cpu"], "statistic": "per-seed repeat median then mean and sample SD across seeds",
            "100k_note": "KS3D uses its stored 100k source; other datasets deterministically repeat the fixed stored source solely for workload scaling"},
        "timing": "GPU synchronized; one training run at a time; includes model/optimizer setup, source pools, OT and checkpoint I/O; excludes imports, dataset verification, physical simulation, metrics and plotting",
        "physical_conventions": "preserve the current manuscript benchmark data; uploaded KPP signs, missing reaction and offset are unchanged",
        "input_sha256": {"mainV2.tex": file_hash(ROOT / "mainV2.tex")}, "datasets": {}}
    for name in NAMES:
        path, _, (p, manifest, splits) = dataset(name)
        plan["datasets"][name] = {"path": str(path), "signature": manifest["signature"],
            "manifest_sha256": file_hash(path / "manifest.json"), "train_levels": [c["level"] for c in splits["train"]],
            "eval_levels": [c["level"] for c in splits["eval"]], "protocol": p.metadata()}
        baseline = historical.baseline_path(name)
        plan["input_sha256"][str(baseline.relative_to(ROOT))] = file_hash(baseline)
    if (OUT / "plan.json").exists():
        previous = js(OUT / "plan.json")
        assert previous == plan, "Preregistered settings or inputs changed"
    else:
        write_json(OUT / "plan.json", plan)
    checks = []
    for dimension in (2, 3, 4):
        mf = MeanFlow(dimension)
        dp = MatchedCorrector(dimension)
        mf_shapes = [(m.in_features, m.out_features) for m in mf.modules() if isinstance(m, torch.nn.Linear)]
        dp_shapes = [(m.in_features, m.out_features) for m in dp.modules() if isinstance(m, torch.nn.Linear)]
        assert mf_shapes == dp_shapes
        assert sum(p.numel() for p in mf.parameters()) == sum(p.numel() for p in dp.parameters())
        x, cond = torch.randn(16, dimension), torch.randn(16, 1)
        assert torch.equal(dp(x, cond), x)
        with torch.no_grad():
            dp.net[-1].weight.normal_(std=.1)
        prediction = dp(x, cond)
        loss = prediction.square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in dp.parameters())
        restored = MatchedCorrector(dimension)
        restored.load_state_dict(dp.state_dict(), strict=True)
        assert torch.equal(prediction, restored(x, cond))
        # A common parameter tensor must produce exactly the same MLP output.
        mf.net.load_state_dict(dp.net.state_dict())
        expected = x + dp.alpha * mf(x, torch.zeros_like(cond), torch.ones_like(cond), cond)
        assert torch.equal(prediction, expected)
        checks.append({"dimension": dimension, "parameters_per_network": sum(p.numel() for p in dp.parameters()),
            "identical_linear_shapes": mf_shapes, "identity_at_initialization": True,
            "finite_training_gradients": True, "checkpoint_roundtrip_bitwise": True,
            "same_weights_same_backbone_bitwise": True})
    write_json(OUT / "preflight_verification.json", {"status": "passed", "architecture_checks": checks})
    print("PREFLIGHT passed: four fixed datasets, identical MF/DP MLPs, gradients and save/load checked", flush=True)


def snapshot():
    sources = [OUT / "experiment.py", OUT / "matched_architecture.py",
        *sorted((PACKAGE / "particle_pipeline").glob("*.py")), PACKAGE / "training/core.py",
        PACKAGE / "training/protocol.py", ROOT / "output/kpp_original_training_total1k_20261014/experiment.py"]
    write_json(OUT / "source_manifest.json", {str(p.relative_to(ROOT)): file_hash(p) for p in sources})
    if not (OUT / "source.zip").exists():
        with zipfile.ZipFile(OUT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sources:
                archive.write(path, path.relative_to(ROOT).as_posix())


def train(name):
    path, _, (p, manifest, splits) = dataset(name)
    run = OUT / name / "run"
    if (run / "training_complete.json").exists():
        assert js(run / "training_complete.json")["dataset_signature"] == manifest["signature"]
        print(f"REUSE completed immutable training {name}", flush=True)
        return
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Partial run preserved; inspect before proceeding: {run}")
    run.mkdir(parents=True, exist_ok=True)
    hp, mode = args(), "protocol" if p.periodic else "log10_sigma"
    config = {"status": "training", "dataset": str(path), "dataset_signature": manifest["signature"],
        "protocol": p.metadata(), "training": vars(hp), "dp_architecture": describe(p.dimension, .1, mode),
        "budgets": {"mf": 1000, "mf_init": 500, "dp": 1000, "mf_dp": 1000}}
    write_json(run / "config.json", config)
    saved, timing_rows = {}, []
    old_corrector, old_checkpoint = engine.Corrector, engine.checkpoint
    for seed in SEEDS:
        tick(phase="training", model=name, seed=seed)
        folder = run / f"seed{seed}"
        folder.mkdir()
        prefix = {}
        def checked_checkpoint(file, model, optimizer, **kw):
            if isinstance(model, MatchedCorrector):
                kw["architecture"] = describe(p.dimension, model.alpha, mode)
            old_checkpoint(file, model, optimizer, **kw)
            if Path(file).name == "mf_init.pt":
                engine.sync("cuda")
                prefix["seconds"] = time.perf_counter()-mf_start
        def factory(dimension, periodic, width, blocks, alpha):
            # Preserve the batch RNG offset while changing the actual network.
            state = torch.get_rng_state()
            legacy = old_corrector(dimension, periodic, width, blocks, alpha)
            after = torch.get_rng_state()
            del legacy
            torch.set_rng_state(state)
            model = MatchedCorrector(dimension, width=width, alpha=alpha, condition_mode=mode)
            torch.set_rng_state(after)
            return model
        engine.checkpoint, engine.Corrector = checked_checkpoint, factory
        try:
            engine.sync("cuda")
            mf_start = time.perf_counter()
            mf, init_state, _ = engine.train_mf(splits["train"], p, hp, folder, manifest["signature"], seed)
            engine.sync("cuda")
            mf_seconds = time.perf_counter()-mf_start
            del mf
            engine.sync("cuda")
            direct_start = time.perf_counter()
            dp, direct = engine.train_dp(splits["train"], p, hp, folder, manifest["signature"], seed)
            engine.sync("cuda")
            direct_seconds = time.perf_counter()-direct_start
            del dp
            engine.sync("cuda")
            refine_start = time.perf_counter()
            mf_init = MeanFlow(p.dimension, 64, mode).cuda().eval().requires_grad_(False)
            mf_init.load_state_dict(init_state, strict=True)
            before = {k: v.detach().cpu().clone() for k, v in mf_init.state_dict().items()}
            corrector, refine = engine.train_dp(splits["train"], p, hp, folder, manifest["signature"], seed, mf=mf_init)
            engine.sync("cuda")
            refinement_seconds = time.perf_counter()-refine_start
            assert all(torch.equal(before[k], value.cpu()) for k, value in mf_init.state_dict().items())
            del corrector, mf_init
        finally:
            engine.checkpoint, engine.Corrector = old_checkpoint, old_corrector
        saved[str(seed)] = {method: {"file": f"seed{seed}/{method}.pt", "sha256": file_hash(folder / f"{method}.pt")}
                           for method in (*METHODS, "mf_init")}
        for method, seconds, detail in [("mf", mf_seconds, {}), ("dp", direct_seconds, direct),
            ("mf_dp", prefix["seconds"]+refinement_seconds, refine)]:
            timing_rows.append({"model": name, "seed": seed, "method": method,
                "training_seconds": seconds, "mf500_seconds": prefix["seconds"] if method=="mf_dp" else 0.,
                "refinement_complete_seconds": refinement_seconds if method=="mf_dp" else 0.,
                "ot_seconds": detail.get("ot_seconds", 0.), "ot_solves": detail.get("ot_solves", 0),
                "total_updates": 1000, "frozen_MF_unchanged": True})
        write_json(run / "checkpoint_manifest.json", saved)
        CSV(run / "training_per_seed.csv", timing_rows)
        print(f"TRAINED {name} seed{seed}: MF={mf_seconds:.3f}s DP={direct_seconds:.3f}s MF+DP={prefix['seconds']+refinement_seconds:.3f}s", flush=True)
    config["status"] = "complete"
    write_json(run / "config.json", config)
    write_json(run / "training_complete.json", {"status": "complete", "dataset_signature": manifest["signature"],
        "new_updates": 3000*len(SEEDS), "new_checkpoints": len(SEEDS)*4,
        "training_timing": "synchronized external boundaries; source-pool generation and OT included; shared MF500 prefix charged to hybrid"})


def models_for(name, seed, p, signature):
    run = OUT / name / "run"
    saved = js(run / "checkpoint_manifest.json")[str(seed)]
    result = {}
    for kind, record in saved.items():
        path = resolve_recorded_path(record["file"], root=ROOT, base=run, allowed_root=run)
        assert file_hash(path) == record["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["dataset_signature"] == signature and protocols_match(payload["protocol"], p.metadata(), root=ROOT)
        assert payload["step"] == (500 if kind in ("mf_init", "mf_dp") else 1000)
        arch = payload["architecture"]
        model = (MeanFlow(p.dimension, 64, arch["condition_mode"]) if kind in ("mf", "mf_init") else
            MatchedCorrector(p.dimension, alpha=arch["alpha"], condition_mode=arch["condition_mode"]))
        model.load_state_dict(payload["model"], strict=True)
        result[kind] = model.cuda().eval().requires_grad_(False)
    return result


@torch.no_grad()
def sample(method, source, level, p, models, *, resident=False):
    chunks = []
    for begin in range(0, len(source), 10000):
        x = source[begin:begin+10000].to("cuda")
        condition = torch.full((len(x), 1), p.condition(level), device="cuda")
        mf = models["mf"] if method == "mf" else models["mf_init"]
        selected = "mf" if method == "mf_init" else method
        x = generate(selected, x, condition, p, mf=mf,
                     dp=models["mf_dp"] if method == "mf_dp" else models["dp"],
                     mf_condition=engine.mf_condition(mf, level, p, len(x), "cuda"))
        chunks.append(x if resident else x.cpu())
    return torch.cat(chunks)


def evaluation_case(name, index, case):
    if name == "ks3d":
        with np.load(historical.KS3_INPUTS / f"sigma{int(case['level'])}_inputs.npz", allow_pickle=False) as data:
            source, ref = data["source"].copy(), data["reference"].copy()
        with np.load(historical.KS3_CLOUDS / f"total1000_sigma{int(case['level'])}.npz", allow_pickle=False) as data:
            baseline = {f"old_seed{s}_{new}": data[f"seed{s}_{old}"].copy() for s in SEEDS
                        for old, new in (("MF", "mf"), ("DP", "dp"), ("hybrid2000", "mf_dp"))}
            edges = [data[f"edges_{a}"].copy() for a in "xyz"]
        return source, ref, baseline, edges
    return case["prior"].copy(), case["target"].copy(), {}, None


def evaluate(name):
    _, _, (p, manifest, splits) = dataset(name)
    destination = OUT / name / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "verification.json").exists():
        assert js(destination / "verification.json")["status"] == "complete"
        print(f"REUSE complete evaluation {name}", flush=True)
        return
    models = {seed: models_for(name, seed, p, manifest["signature"]) for seed in SEEDS}
    previous = historical.baseline_rows(name)
    old_lookup = {(r["level"], r["model_seed"], r["method"], r["metric"]): r["w2"] for r in previous}
    rows, checks = [], []
    for index, case in enumerate(splits["eval"]):
        tick(phase="evaluation", model=name, level=case["level"], completed=index, total=len(splits["eval"]))
        level = float(case["level"])
        source, reference, old_clouds, edges = evaluation_case(name, index, case)
        clouds = {f"seed{s}_{method}": sample(method, torch.from_numpy(source), level, p, models[s]).numpy()
                  for s in SEEDS for method in (*METHODS, "mf_init")}
        assert all(cloud.shape == source.shape and np.isfinite(cloud).all() for cloud in clouds.values())
        scores = {}
        if name == "ks3d":
            measures, edges = historical.anchored_measures(reference, old_clouds, clouds, edges)
            with ThreadPoolExecutor(max_workers=2) as executor:
                scores[PRIMARY[name]] = dict(executor.map(lambda key: (key, historical.exact_histogram_score(measures["reference"], measures[key])), clouds))
            if level in (10., 160., 200.):
                for method in METHODS:
                    measured = historical.exact_histogram_score(measures["reference"], measures[f"old_seed88_{method}"])["w2"]
                    expected = old_lookup[level, 88, method, PRIMARY[name]]
                    assert math.isclose(measured, expected, rel_tol=1e-10, abs_tol=1e-10)
                    checks.append({"level": level, "method": method, "old_metric_reproduced": True,
                                   "difference": abs(measured-expected)})
        if p.periodic:
            metric = "periodic_histogram_xz" if p.dimension == 3 else "periodic_histogram"
            scores[metric] = histogram_metrics(reference, clouds, 72, True, (0, 2) if p.dimension == 3 else None, workers=2)
        if name in ("ks4d", "kpp3d"):
            with ThreadPoolExecutor(max_workers=2) as executor:
                scores[PRIMARY[name]] = dict(executor.map(lambda key: (key, cloud_metric(reference, clouds[key], p.periodic, 2000, 999)), clouds))
        for metric, results in scores.items():
            for seed in SEEDS:
                for method in (*METHODS, "mf_init"):
                    score = results[f"seed{seed}_{method}"]
                    rows.append({"model": name, "level": level, "parameter": p.parameter(level), "seed": seed,
                        "method": method, "metric": metric, "w2": score["w2"], "w2_squared": score["w2_squared"],
                        "particles": len(source), "origin": "retrained_matched_backbone"})
        write_npz(destination / f"particles_{index:02d}.npz", meta={"protocol": p.metadata(), "level": level,
            "dataset_signature": manifest["signature"]}, reference=reference, **clouds,
            **({f"edges_{i}": edge for i, edge in enumerate(edges)} if edges is not None else {}))
        CSV(destination / "metrics_per_seed.csv", rows)
        print(f"EVALUATED {name} {index+1}/{len(splits['eval'])}, level={level:g}", flush=True)
    write_json(destination / "verification.json", {"status": "complete", "rows": len(rows),
        "dataset_signature": manifest["signature"], "all_mass_retained": True, "inherited_metric_checks": checks,
        "uncertainty": "three training seeds on fixed references; SD not CI; no reference uncertainty estimate"})


def benchmark(name):
    _, _, (p, manifest, splits) = dataset(name)
    folder = OUT / name / "inference"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "verification.json").exists():
        return
    level = 4.5 if p.periodic else 160.
    case = next(c for c in splits["eval"] if c["level"] == level)
    native, _, _, _ = evaluation_case(name, splits["eval"].index(case), case)
    rng = random.Random(20261016)
    rows, identities = [], []
    for seed in SEEDS:
        models = models_for(name, seed, p, manifest["signature"])
        for n in (2000, 10000, 100000):
            source = torch.from_numpy(np.tile(native, (math.ceil(n/len(native)), 1))[:n].copy())
            gpu_source = source.cuda()
            for method in (*METHODS, "mf_init"):
                a = sample(method, source, level, p, models)
                b = sample(method, gpu_source, level, p, models, resident=True).cpu()
                assert torch.equal(a, b)
            identities.append({"seed": seed, "particles": n, "resident_cpu_bitwise_equal": True})
            for scope in ("gpu_resident", "cpu_to_cpu"):
                resident = scope == "gpu_resident"
                call = lambda method: sample(method, gpu_source if resident else source, level, p, models, resident=resident)
                for _ in range(10):
                    for method in (*METHODS, "mf_init"):
                        call(method)
                engine.sync("cuda")
                for repeat in range(40):
                    methods = list((*METHODS, "mf_init"))
                    rng.shuffle(methods)
                    for position, method in enumerate(methods):
                        engine.sync("cuda")
                        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        started = time.perf_counter()
                        if resident:
                            start_event.record()
                        result = call(method)
                        if resident:
                            end_event.record()
                        engine.sync("cuda")
                        elapsed = (time.perf_counter()-started)*1000
                        rows.append({"model": name, "level": level, "seed": seed, "particles": n, "scope": scope,
                            "method": method, "repeat": repeat, "order": position, "wall_ms": elapsed,
                            "cuda_ms": start_event.elapsed_time(end_event) if resident else ""})
                CSV(folder / "raw_timings.csv", rows)
                tick(phase="inference_timing", model=name, seed=seed, particles=n, scope=scope)
        print(f"TIMED {name} seed{seed}", flush=True)
    CSV(folder / "per_seed_timings.csv", aggregate_timing(rows))
    write_json(folder / "verification.json", {"status": "complete", "technical_rows": len(rows), "identities": identities,
        "warmups": 10, "repeats": 40, "scope": "synchronized eager forward passes; condition construction included; no checkpoint loading, RNG, metrics or physical simulation",
        "cpu_to_cpu": "one H2D and D2H per chunk; MF+DP intermediate stays on GPU",
        "100k_source": "native KS3D; repeated stored sources in other models for scaling only"})


def aggregate_timing(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["seed"], row["particles"], row["scope"], row["method"])].append(float(row["wall_ms"]))
    return [{"model": k[0], "seed": k[1], "particles": k[2], "scope": k[3], "method": k[4],
             "median_ms": statistics.median(v), "mean_ms": statistics.mean(v), "sd_ms": statistics.stdev(v),
             "p25_ms": float(np.percentile(v, 25)), "p75_ms": float(np.percentile(v, 75)), "repeats": len(v)}
            for k,v in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "train", "evaluate", "benchmark", "all"))
    parser.add_argument("--models", nargs="+", choices=NAMES, default=list(NAMES))
    selected = parser.parse_args()
    configure_device("cuda")
    torch.set_num_threads(1)
    preflight()
    snapshot()
    write_json(OUT / "runtime.json", {"python": sys.version, "torch": str(torch.__version__),
        "numpy": np.__version__, "gpu": torch.cuda.get_device_name(), "threads": 1,
        "precision": "float32; TF32 off; deterministic algorithms", "new_physical_simulation": False})
    for name in selected.models:
        if selected.stage in ("train", "all"):
            train(name)
        if selected.stage in ("benchmark", "all"):
            benchmark(name)
        if selected.stage in ("evaluate", "all"):
            evaluate(name)
    tick(phase="stage_complete", stage=selected.stage, models=selected.models)


if __name__ == "__main__":
    with torch.no_grad():
        # Training explicitly re-enables gradients; generation stays forward-only.
        pass
    main()
