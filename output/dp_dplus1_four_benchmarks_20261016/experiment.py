"""Four fixed manuscript benchmarks: paired DP d+3 versus d+1 inputs."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT / "output/mf_matched_dp_20261016"
REUSE = ROOT / "output/mfdp_dplus1_input_ablation_20261016"
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location("four_benchmark_baseline", BASE / "experiment.py")
e = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e
spec.loader.exec_module(e)
from compact_architecture import CompactCorrector, describe as compact_description
from paper_reproduction.portability import protocols_match

torch, np = e.torch, e.np
NAMES = ("kpp2d", "ks3d", "ks4d", "kpp3d")
SEEDS = (88, 89, 90)
METHODS = ("dp", "mf_dp")
VARIANTS = ("dplus3", "dplus1")
START = time.perf_counter()
TIMEOUT = 7200


def tick(**values):
    elapsed = time.perf_counter() - START
    if elapsed > TIMEOUT:
        raise TimeoutError("Two-hour stage timeout")
    e.write_json(OUT / "progress.json", {"pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat(),
                 "elapsed_seconds": elapsed, **values})


def mode(p):
    return "protocol" if p.periodic else "log10_sigma"


def make_model(variant, p, alpha):
    constructor = CompactCorrector if variant == "dplus1" else e.MatchedCorrector
    return constructor(p.dimension, alpha=alpha, condition_mode=mode(p))


def description(variant, p, alpha):
    if variant == "dplus3":
        return e.describe(p.dimension, alpha, mode(p))
    result = compact_description(p.dimension, alpha)
    result["condition_mode"] = mode(p)
    return result


def original_path(name, seed, method):
    return BASE / name / "run" / f"seed{seed}/{method}.pt"


def read_payload(path, p, signature):
    item = torch.load(path, map_location="cpu", weights_only=True)
    assert protocols_match(item["protocol"], p.metadata(), root=ROOT) and item["dataset_signature"] == signature
    return item


def frozen_mf(name, seed, p, signature):
    item = read_payload(original_path(name, seed, "mf_init"), p, signature)
    assert item["step"] == 500
    mf = e.MeanFlow(p.dimension, 64, mode(p), 3)
    mf.load_state_dict(item["model"], strict=True)
    return mf.cuda().eval().requires_grad_(False)


def setup():
    e.configure_device("cuda")
    torch.set_num_threads(1)


def preflight():
    setup()
    inputs, datasets, checks = {}, {}, []
    sources = [Path(__file__), OUT / "compact_architecture.py", BASE / "experiment.py",
               BASE / "matched_architecture.py", ROOT / "deepparticle/particle_pipeline/engine.py",
               ROOT / "deepparticle/particle_pipeline/models.py", ROOT / "deepparticle/particle_pipeline/geometry.py"]
    for name in NAMES:
        path, _, (p, manifest, splits) = e.dataset(name)
        datasets[name] = {"path": str(path), "signature": manifest["signature"], "protocol": p.metadata(),
                          "train_levels": [c["level"] for c in splits["train"]],
                          "eval_levels": [c["level"] for c in splits["eval"]], "primary_metric": e.PRIMARY[name]}
        paths = [path / "manifest.json", BASE / name / "evaluation/metrics_per_seed.csv",
                 *[original_path(name, seed, method) for seed in SEEDS for method in (*METHODS, "mf_init", "mf")],
                 *[BASE / name / "run" / f"seed{seed}/{method}_loss.csv" for seed in SEEDS for method in METHODS]]
        inputs.update({str(path.relative_to(ROOT)): e.file_hash(path) for path in paths})
        for variant in VARIANTS:
            model = make_model(variant, p, 10.)
            x, c = torch.randn(32, p.dimension), torch.randn(32, 1)
            assert model.net[0].in_features == p.dimension + (1 if variant == "dplus1" else 3)
            assert torch.equal(model(x, c), x)
            with torch.no_grad():
                model.net[-1].weight.normal_(std=.01)
            model(x, c).square().mean().backward()
            assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
            restored = make_model(variant, p, 10.)
            restored.load_state_dict(model.state_dict(), strict=True)
            assert torch.equal(model(x, c), restored(x, c))
            checks.append({"model": name, "variant": variant, "inputs": model.net[0].in_features,
                           "parameters": sum(q.numel() for q in model.parameters()), "identity_and_gradients": True,
                           "checkpoint_roundtrip_bitwise": True})
        for method in METHODS:
            item = read_payload(original_path(name, 88, method), p, manifest["signature"])
            alpha = 10. if method == "mf_dp" else .1
            old, compact = make_model("dplus3", p, alpha), make_model("dplus1", p, alpha)
            old.load_state_dict(item["model"], strict=True)
            folded = {k: v.clone() for k, v in item["model"].items()}
            folded["net.0.bias"] += folded["net.0.weight"][:, p.dimension]
            folded["net.0.weight"] = folded["net.0.weight"][:, [*range(p.dimension), p.dimension + 2]].clone()
            compact.load_state_dict(folded, strict=True)
            error = (old(x, c) - compact(x, c)).abs().max().item()
            assert error < 1e-4, (name, method, error)
    plan = {"material_passport": {"origin_skill": "academic-research-suite", "origin_mode": "experiment-agent run + validate",
            "date": "2026-10-16", "verification_status": "PLANNED"}, "question": "Remove DP constant endpoint channels in all four manuscript benchmarks",
            "models": list(NAMES), "seeds": list(SEEDS), "datasets": datasets,
            "budgets": {"dp": 1000, "mf_dp": {"frozen_MF": 500, "new_DP": 500}},
            "hyperparameters": vars(e.args()), "variants": {"dplus3": "[x,t=1,r=0,condition]", "dplus1": "[x,condition]"},
            "hidden_layers": "three width64 SELU hidden layers, linear zero-initialized output", "DP_alpha": .1,
            "MF_DP_alpha": 10., "MF_DP_displacement_penalty": .1,
            "pairing": "same data, physical conditions, seeds, source pools and batches; preserve historical legacy constructor RNG offset",
            "initialization": "fresh default Linear initialization per shape; fan-in and random parameter assignment change",
            "controls": "retrain every d+3 control; require original weights and full loss logs bitwise; reuse audited KPP2D hybrid pairs",
            "reuse": "KPP2D MF+DP three-seed d+3/d+1 experiment reused with hash, model, log and schedule checks",
            "selection": "fixed final updates; no tuning or test-based selection", "physical_simulation": "existing audited datasets only",
            "metrics": "same references/grids and exact OT as manuscript; all20 KS/all11 KPP levels; KPP3D full cloud primary and xz histogram secondary",
            "uncertainty": "three training seeds on fixed physical references; sample SD and min-max; parameter levels are not independent seeds",
            "input_sha256": inputs, "source_sha256": {str(p.relative_to(ROOT)): e.file_hash(p) for p in sources}}
    if (OUT / "plan.json").exists():
        assert e.js(OUT / "plan.json") == plan, "Inputs or preregistered plan changed"
    else:
        e.write_json(OUT / "plan.json", plan)
        e.write_json(OUT / "preflight_verification.json", {"status": "passed", "architectures": checks,
                     "constant_channel_folding_checked": True, "folding_tolerance": 1e-4})
        e.write_json(OUT / "runtime.json", {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                     "gpu": torch.cuda.get_device_name(), "threads": 1, "precision": "float32; TF32 off; deterministic"})
        with e.zipfile.ZipFile(OUT / "source.zip", "x", e.zipfile.ZIP_DEFLATED) as archive:
            for path in sources:
                archive.write(path, path.relative_to(ROOT).as_posix())
    print("PREFLIGHT passed: all four datasets, shapes, gradients, checkpoint loading and constant-input folding", flush=True)


def train_one(name, variant, seed, method, p, manifest, splits):
    folder = OUT / name / variant / f"seed{seed}" / method
    assert not folder.exists(), f"Existing run preserved: {folder}"
    folder.mkdir(parents=True)
    if name == "kpp2d" and method == "mf_dp":
        origin = REUSE / variant / f"seed{seed}"
        record = e.js(origin / "record.json")
        assert e.file_hash(origin / "mf_dp.pt") == record["checkpoint_sha256"]
        for filename in ("mf_dp.pt", "mf_dp_loss.csv", "batch_schedule.json"):
            shutil.copy2(origin / filename, folder / filename)
        record = {"model": name, "variant": variant, "seed": seed, "method": method, **record,
                  "reused_verified_run": True, "origin": str(origin.relative_to(ROOT))}
        e.write_json(folder / "record.json", record)
        return record
    mf = frozen_mf(name, seed, p, manifest["signature"]) if method == "mf_dp" else None
    before = {k: v.detach().cpu().clone() for k, v in mf.state_dict().items()} if mf else {}
    old_factory, old_checkpoint, old_randperm = e.engine.Corrector, e.engine.checkpoint, torch.randperm
    batches, initial_batches, assignment_count = [], [], 0

    def factory(dimension, periodic, width, blocks, alpha):
        initial_rng = torch.get_rng_state()
        legacy = old_factory(dimension, periodic, width, blocks, alpha)
        historical_rng = torch.get_rng_state()
        del legacy
        torch.set_rng_state(initial_rng)
        model = make_model(variant, p, alpha)
        torch.set_rng_state(historical_rng)
        return model

    def save(file, model, optimizer, **kw):
        kw["architecture"] = description(variant, p, model.alpha)
        old_checkpoint(file, model, optimizer, **kw)

    def randperm(*args, **kwargs):
        values = old_randperm(*args, **kwargs)
        batches.append({"pool_length": len(values), "selected_indices_sha256": hashlib.sha256(values[:2000].cpu().numpy().tobytes()).hexdigest()})
        return values

    def solver(first, second, periodic):
        nonlocal assignment_count
        if assignment_count < len(splits["train"]):
            initial_batches.append({"source_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
                                    "target_sha256": hashlib.sha256(second.tobytes()).hexdigest()})
        assignment_count += 1
        return e.engine.exact_assignment(first, second, periodic)

    e.engine.Corrector, e.engine.checkpoint, torch.randperm = factory, save, randperm
    e.engine.sync("cuda")
    start = time.perf_counter()
    try:
        model, details = e.engine.train_dp(splits["train"], p, e.args(), folder, manifest["signature"], seed, mf=mf, assignment_solver=solver)
    finally:
        e.engine.Corrector, e.engine.checkpoint, torch.randperm = old_factory, old_checkpoint, old_randperm
    e.engine.sync("cuda")
    seconds = time.perf_counter() - start
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    if mf:
        assert all(torch.equal(before[k], v.detach().cpu()) for k, v in mf.state_dict().items())
    control_equal = loss_equal = None
    if variant == "dplus3":
        original = read_payload(original_path(name, seed, method), p, manifest["signature"])
        control_equal = all(torch.equal(v.detach().cpu(), original["model"][k]) for k, v in model.state_dict().items())
        loss_equal = (folder / f"{method}_loss.csv").read_bytes() == (BASE / name / "run" / f"seed{seed}/{method}_loss.csv").read_bytes()
        assert control_equal and loss_equal, f"Original control differs: {name}/{seed}/{method}"
    record = {"model": name, "variant": variant, "seed": seed, "method": method, "DP_stage_seconds": seconds,
              "DP_updates": details["steps"], "parameters": sum(q.numel() for q in model.parameters()),
              "MF_frozen_unchanged": True, "control_parameters_bitwise_equal": control_equal,
              "control_loss_trace_byte_identical": loss_equal, "ot_solves": details["ot_solves"],
              "checkpoint_sha256": e.file_hash(folder / f"{method}.pt"), "reused_verified_run": False}
    e.write_json(folder / "batch_schedule.json", batches)
    e.write_json(folder / "initial_OT_batches.json", initial_batches)
    e.write_json(folder / "record.json", record)
    print(f"TRAINED {name} {method} {variant} seed{seed}: {seconds:.2f}s; control={control_equal}", flush=True)
    return record


def train(names):
    preflight()
    for name in names:
        if (OUT / name / "training_verification.json").exists():
            assert e.js(OUT / name / "training_verification.json")["status"] == "passed"
            continue
        _, _, (p, manifest, splits) = e.dataset(name)
        records, paired = [], []
        for seed in SEEDS:
            for method in METHODS:
                variants = list(VARIANTS)
                random.Random(20261016 + seed).shuffle(variants)
                for variant in variants:
                    tick(stage="training", model=name, variant=variant, method=method, seed=seed)
                    records.append(train_one(name, variant, seed, method, p, manifest, splits))
                    e.CSV(OUT / name / "training_per_seed.csv", records)
                folders = [OUT / name / v / f"seed{seed}" / method for v in VARIANTS]
                assert e.js(folders[0] / "batch_schedule.json") == e.js(folders[1] / "batch_schedule.json")
                if all((f / "initial_OT_batches.json").exists() for f in folders):
                    assert e.js(folders[0] / "initial_OT_batches.json") == e.js(folders[1] / "initial_OT_batches.json")
                paired.append({"seed": seed, "method": method, "identical_batches_and_initial_clouds": True})
        e.write_json(OUT / name / "training_verification.json", {"status": "passed", "records": len(records),
                     "all_original_controls_bitwise": True, "MF_frozen_unchanged": True, "paired_schedules": paired})
    tick(stage="training_complete", models=list(names))


def read_models(name, seed, p, signature):
    result = {"mf_init": frozen_mf(name, seed, p, signature)}
    for method in METHODS:
        folder = OUT / name / "dplus1" / f"seed{seed}" / method
        record = e.js(folder / "record.json")
        assert e.file_hash(folder / f"{method}.pt") == record["checkpoint_sha256"]
        item = read_payload(folder / f"{method}.pt", p, signature)
        assert item["step"] == (500 if method == "mf_dp" else 1000)
        model = make_model("dplus1", p, 10. if method == "mf_dp" else .1)
        model.load_state_dict(item["model"], strict=True)
        result[method] = model.cuda().eval().requires_grad_(False)
    return result


def evaluate(names):
    setup()
    for name in names:
        destination = OUT / name / "evaluation"
        if (destination / "verification.json").exists():
            assert e.js(destination / "verification.json")["status"] == "passed"
            continue
        assert e.js(OUT / name / "training_verification.json")["status"] == "passed"
        _, _, (p, manifest, splits) = e.dataset(name)
        models = {seed: read_models(name, seed, p, manifest["signature"]) for seed in SEEDS}
        previous = e.historical.read_csv(BASE / name / "evaluation/metrics_per_seed.csv")
        rows, checks = [], []
        destination.mkdir(exist_ok=True, parents=True)
        for index, case in enumerate(splits["eval"]):
            level = float(case["level"])
            tick(stage="evaluation", model=name, level=level, completed=index, total=len(splits["eval"]))
            source, reference, _, _ = e.evaluation_case(name, index, case)
            clouds = {f"seed{s}_{m}": e.sample(m, torch.from_numpy(source), level, p, models[s]).numpy() for s in SEEDS for m in METHODS}
            assert all(c.shape == source.shape and np.isfinite(c).all() for c in clouds.values())
            if p.periodic:
                assert all(np.all((c >= 0) & (c < 2 * math.pi)) for c in clouds.values())
            saved, meta = e.read_npz(BASE / name / "evaluation" / f"particles_{index:02d}.npz")
            assert meta["dataset_signature"] == manifest["signature"]
            np.testing.assert_array_equal(reference, saved["reference"])
            recheck = index in (0, len(splits["eval"]) - 1)
            scored_clouds = dict(clouds)
            if recheck:
                for method in METHODS:
                    scored_clouds[f"old_seed88_{method}"] = saved[f"seed88_{method}"]
            scores = {}
            edges = None
            if name == "ks3d":
                old_clouds = {f"old_{key}": value for key, value in saved.items() if key.startswith("seed")}
                old_edges = [saved[f"edges_{i}"] for i in range(3)]
                measures, edges = e.historical.anchored_measures(reference, old_clouds, clouds, old_edges)
                with ThreadPoolExecutor(max_workers=2) as executor:
                    scores[e.PRIMARY[name]] = dict(executor.map(lambda k: (k, e.historical.exact_histogram_score(measures["reference"], measures[k])), scored_clouds))
            if p.periodic:
                metric = "periodic_histogram_xz" if p.dimension == 3 else "periodic_histogram"
                scores[metric] = e.histogram_metrics(reference, scored_clouds, 72, True, (0, 2) if p.dimension == 3 else None, workers=2)
            if name in ("ks4d", "kpp3d"):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    scores[e.PRIMARY[name]] = dict(executor.map(lambda k: (k, e.cloud_metric(reference, scored_clouds[k], p.periodic, 2000, 999)), scored_clouds))
            for metric, measured in scores.items():
                if recheck:
                    for method in METHODS:
                        expected = next(float(r["w2"]) for r in previous if int(r["seed"]) == 88 and float(r["level"]) == level and r["method"] == method and r["metric"] == metric)
                        delta = abs(measured[f"old_seed88_{method}"]["w2"] - expected)
                        assert delta < 1e-10, (name, level, metric, delta)
                        checks.append({"level": level, "method": method, "metric": metric, "old_metric_difference": delta})
                for seed in SEEDS:
                    for method in METHODS:
                        score = measured[f"seed{seed}_{method}"]
                        rows.append({"model": name, "variant": "dplus1", "level": level, "parameter": p.parameter(level),
                            "seed": seed, "method": method, "metric": metric, "w2": score["w2"], "w2_squared": score["w2_squared"],
                            "particles": len(source), "metric_particles": 2000 if metric.endswith("full_cloud") else len(source)})
            e.write_npz(destination / f"particles_{index:02d}.npz", meta={"protocol": p.metadata(), "level": level,
                        "dataset_signature": manifest["signature"], "variant": "dplus1"}, reference=reference, **clouds,
                        **({f"edges_{i}": edge for i, edge in enumerate(edges)} if edges is not None else {}))
            e.CSV(destination / "metrics_per_seed.csv", rows)
            print(f"EVALUATED {name} {index + 1}/{len(splits['eval'])}: level={level:g}", flush=True)
        e.write_json(destination / "verification.json", {"status": "passed", "rows": len(rows), "all_mass_retained": True,
                     "dataset_signature": manifest["signature"], "old_metric_checks": checks})
    tick(stage="evaluation_complete", models=list(names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "train", "evaluate"))
    parser.add_argument("--models", nargs="+", choices=NAMES, default=list(NAMES))
    selected = parser.parse_args()
    if selected.stage == "preflight":
        preflight()
    else:
        globals()[selected.stage](selected.models)


if __name__ == "__main__":
    main()
