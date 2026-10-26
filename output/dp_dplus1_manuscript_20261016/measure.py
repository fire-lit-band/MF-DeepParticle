"""Bind verified d+1 models to manuscript data and remeasure affected quantities."""
from pathlib import Path
from collections import defaultdict
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
ABLATION = ROOT / "output/dp_dplus1_four_benchmarks_20261016"
BASE = ROOT / "output/mf_matched_dp_20261016"
BDP = ROOT / "output/bdp_dplus1_input_ablation_20261016"
sys.path.insert(0, str(ABLATION))
import experiment as r
e, torch, np = r.e, r.torch, r.np
NAMES, SEEDS, METHODS = e.NAMES, e.SEEDS, e.METHODS
MASTER = OUT / "benchmarks"
STARTED = time.perf_counter()


def tick(**values):
    elapsed = time.perf_counter() - STARTED
    if elapsed > 10800:
        raise TimeoutError("Measurement stage exceeded three hours")
    e.write_json(OUT / "progress.json", {"pid": os.getpid(), "elapsed_seconds": elapsed, **values})


def read(path):
    return e.historical.read_csv(path)


def final_evaluation(name):
    return ABLATION / "validated_periodic" / name / "evaluation" if name.startswith("kpp") else ABLATION / name / "evaluation"


def prepare():
    assert e.js(ABLATION / "delivery_verification.json")["status"] == "passed"
    assert e.js(BDP / "verification.json")["status"] == "passed"
    rows, checkpoints, inputs = [], {}, {}
    for name in NAMES:
        _, _, (p, manifest, splits) = e.dataset(name)
        old_path, new_path = BASE / name / "evaluation/metrics_per_seed.csv", final_evaluation(name) / "metrics_per_seed.csv"
        inputs.update({str(path.relative_to(ROOT)): e.file_hash(path) for path in (old_path, new_path)})
        for method, path in (("mf", old_path), ("dp", new_path), ("mf_dp", new_path)):
            rows += [{"model": name, "level": float(v["level"]), "seed": int(v["seed"]), "method": method,
                      "metric": v["metric"], "w2": float(v["w2"]), "w2_squared": float(v["w2_squared"]),
                      "particles": int(v["particles"]), "origin": "reused_MF" if method == "mf" else "verified_dplus1"}
                     for v in read(path) if v["method"] == method]
        checkpoints[name] = {}
        for seed in SEEDS:
            checkpoints[name][str(seed)] = {}
            for method in ("mf", "mf_init", "dp", "mf_dp"):
                path = (BASE / name / "run" / f"seed{seed}/{method}.pt" if method in ("mf", "mf_init") else
                        ABLATION / name / "dplus1" / f"seed{seed}" / method / f"{method}.pt")
                item = r.read_payload(path, p, manifest["signature"])
                assert item["step"] == (500 if method in ("mf_init", "mf_dp") else 1000)
                inputs[str(path.relative_to(ROOT))] = e.file_hash(path)
                checkpoints[name][str(seed)][method] = {"file": str(path.relative_to(ROOT)),
                    "sha256": e.file_hash(path), "step": item["step"], "inputs": item["model"]["net.0.weight"].shape[1]}
            assert checkpoints[name][str(seed)]["dp"]["inputs"] == p.dimension + 1
            assert checkpoints[name][str(seed)]["mf_dp"]["inputs"] == p.dimension + 1
        selected = {"ks3d": [15], "ks4d": [], "kpp2d": [7, 8, 10], "kpp3d": [10]}[name]
        for index in selected:
            old, old_meta = e.read_npz(BASE / name / "evaluation" / f"particles_{index:02d}.npz")
            new, meta = e.read_npz(final_evaluation(name) / f"particles_{index:02d}.npz")
            assert old_meta["dataset_signature"] == meta["dataset_signature"] == manifest["signature"]
            np.testing.assert_array_equal(old["reference"], new["reference"])
            for seed in SEEDS:
                new[f"seed{seed}_mf"] = old[f"seed{seed}_mf"]
            e.write_npz(MASTER / name / "evaluation" / f"particles_{index:02d}.npz", meta=meta, **new)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"], row["level"], row["method"], row["metric"]].append(row["w2"])
    curves = []
    for key, values in sorted(grouped.items()):
        assert len(values) == 3
        curves.append(dict(zip(("model", "level", "method", "metric"), key)) |
                      {"w2_mean": statistics.mean(values), "w2_sd": statistics.stdev(values),
                       "w2_min": min(values), "w2_max": max(values), "origin": "current_dplus1_paper"})
    e.CSV(OUT / "metrics_per_seed.csv", rows)
    e.CSV(OUT / "curve_summary.csv", curves)
    e.write_json(OUT / "checkpoint_manifest.json", checkpoints)
    for filename in ("multistep_summary.csv", "multistep_verification.json", "multistep_metrics_per_seed.csv"):
        shutil.copy2(ROOT / "output/mf_matched_manuscript_20261016" / filename, OUT / filename)
    (OUT / "multistep_particles").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "output/mf_matched_manuscript_20261016/multistep_particles/sigma160.npz",
                 OUT / "multistep_particles/sigma160.npz")
    e.write_json(OUT / "data_binding_verification.json", {"status": "passed", "metric_rows": len(rows),
                 "model_seeds": SEEDS, "checkpoint_records": sum(len(v) for d in checkpoints.values() for v in d.values()),
                 "input_sha256": inputs, "DP_inputs": "d+1 for both direct DP and MF+DP", "MF_inputs": "d+3 unchanged"})
    print("PREPARED: verified d+1 checkpoints, unchanged MF, merged figure particles and retained two-step MF", flush=True)


def models_for(name, seed, p, signature):
    result = r.read_models(name, seed, p, signature)
    item = r.read_payload(BASE / name / "run" / f"seed{seed}/mf.pt", p, signature)
    mf = e.MeanFlow(p.dimension, 64, r.mode(p), 3)
    mf.load_state_dict(item["model"], strict=True)
    result["mf"] = mf.cuda().eval().requires_grad_(False)
    return result


class CompactBDP(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.forward_map = r.make_model("dplus1", p, .1)
        self.reverse_map = r.make_model("dplus1", p, .1)

    def forward(self, x, condition):
        return torch.remainder(self.forward_map(x, condition), 2 * math.pi)


@torch.no_grad()
def sample(method, source, level, p, models, *, resident=False):
    if method not in ("mf_two_step", "bdp"):
        if method == "dp" and p.periodic:
            source = torch.remainder(source, 2 * math.pi)
        return e.sample(method, source, level, p, models, resident=resident)
    chunks = []
    for begin in range(0, len(source), 10000):
        x = source[begin:begin + 10000].cuda()
        if method == "bdp":
            cond = torch.full((len(x), 1), p.condition(level), device="cuda")
            x = models["bdp"](x, cond)
        else:
            mf = models["mf"]
            cond = e.engine.mf_condition(mf, level, p, len(x), "cuda")
            for t, interval_r in ((1., .5), (.5, 0.)):
                x = x - (t - interval_r) * mf(x, torch.full_like(cond, interval_r), torch.full_like(cond, t), cond)
        chunks.append(x if resident else x.cpu())
    return torch.cat(chunks)


def training(strict_replay=True, compare_retained=True):
    if strict_replay and not compare_retained:
        raise ValueError('Strict replay requires the retained physical data and checkpoints.')
    folder = OUT / "training_replay"
    rows, checks = [], []
    old_factory, old_checkpoint = e.engine.Corrector, e.engine.checkpoint
    for name in NAMES:
        _, _, (p, manifest, splits) = e.dataset(name)
        for seed in SEEDS:
            root = folder / name / f"seed{seed}"
            assert not root.exists(), f"Preserve partial timing run: {root}"
            root.mkdir(parents=True)
            prefix = {}
            def factory(dimension, periodic, width, blocks, alpha):
                before = torch.get_rng_state()
                legacy = old_factory(dimension, periodic, width, blocks, alpha)
                after = torch.get_rng_state()
                del legacy
                torch.set_rng_state(before)
                model = r.make_model("dplus1", p, alpha)
                torch.set_rng_state(after)
                return model
            def checkpoint(path, model, optimizer, **kw):
                if not isinstance(model, e.MeanFlow):
                    kw["architecture"] = r.description("dplus1", p, model.alpha)
                old_checkpoint(path, model, optimizer, **kw)
                if Path(path).name == "mf_init.pt":
                    e.engine.sync("cuda")
                    prefix["seconds"] = time.perf_counter() - mf_start
            e.engine.Corrector, e.engine.checkpoint = factory, checkpoint
            try:
                e.engine.sync("cuda")
                mf_start = time.perf_counter()
                mf, init_state, _ = e.engine.train_mf(splits["train"], p, e.args(), root, manifest["signature"], seed)
                e.engine.sync("cuda")
                mf_seconds = time.perf_counter() - mf_start
                del mf
                e.engine.sync("cuda")
                started = time.perf_counter()
                dp, direct = e.engine.train_dp(splits["train"], p, e.args(), root, manifest["signature"], seed)
                e.engine.sync("cuda")
                dp_seconds = time.perf_counter() - started
                del dp
                e.engine.sync("cuda")
                started = time.perf_counter()
                init = e.MeanFlow(p.dimension, 64, r.mode(p), 3).cuda().eval().requires_grad_(False)
                init.load_state_dict(init_state, strict=True)
                dp, hybrid = e.engine.train_dp(splits["train"], p, e.args(), root, manifest["signature"], seed, mf=init)
                e.engine.sync("cuda")
                refinement_seconds = time.perf_counter() - started
                del dp, init
            finally:
                e.engine.Corrector, e.engine.checkpoint = old_factory, old_checkpoint
            for method in ("mf", "mf_init", "dp", "mf_dp"):
                replay = torch.load(root / f"{method}.pt", map_location="cpu", weights_only=True)
                if (replay['step'] != (500 if method in ('mf_init', 'mf_dp') else 1000)
                        or replay['dataset_signature'] != manifest['signature']
                        or replay['protocol'] != p.metadata() or not replay['model']):
                    raise ValueError(f'Invalid new checkpoint identity: {name}, {seed}, {method}')
                if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in replay['model'].values()):
                    raise ValueError(f'Invalid new parameters: {name}, {seed}, {method}')
                equal, maximum = None, None
                if compare_retained:
                    path = (BASE / name / "run" / f"seed{seed}/{method}.pt" if method in ("mf", "mf_init") else
                            ABLATION / name / "dplus1" / f"seed{seed}" / method / f"{method}.pt")
                    target = r.read_payload(path, p, manifest["signature"])
                    assert replay["step"] == target["step"]
                    if set(replay["model"]) != set(target["model"]):
                        raise ValueError(f"Checkpoint parameter names changed: {name}, {seed}, {method}")
                    differences = []
                    for key, value in replay["model"].items():
                        expected = target["model"][key]
                        if value.shape != expected.shape:
                            raise ValueError(f"Invalid trained parameter shape: {name}, {seed}, {method}, {key}")
                        differences.append(float((value.double() - expected.double()).abs().max()))
                    equal = all(torch.equal(value, target["model"][key]) for key, value in replay["model"].items())
                    maximum = max(differences)
                checks.append({"model": name, "seed": seed, "method": method,
                               "weights_bitwise_equal": equal, "max_absolute_difference": maximum,
                               "historical_comparison_performed": compare_retained,
                               "finite_parameters": True,
                               "file": (root / f"{method}.pt").relative_to(OUT).as_posix(),
                               "sha256": e.file_hash(root / f"{method}.pt")})
                if strict_replay and not equal:
                    e.write_json(OUT / "training_replay_verification.json", {
                        "status": "failed", "strict_replay": True, "checks": checks,
                        "reason": f"Trained tensors differ: {name}, {seed}, {method}"})
                    raise RuntimeError(f"Strict replay tensors differ: {name}, {seed}, {method}")
            for method, seconds, ot_seconds in (("mf", mf_seconds, 0.), ("dp", dp_seconds, direct["ot_seconds"]),
                    ("mf_dp", prefix["seconds"] + refinement_seconds, hybrid["ot_seconds"])):
                rows.append({"model": name, "seed": seed, "method": method, "training_seconds": seconds,
                             "mf500_seconds": prefix["seconds"] if method == "mf_dp" else 0.,
                             "ot_seconds": ot_seconds, "total_updates": 1000})
            e.CSV(OUT / "training_per_seed.csv", rows)
            tick(stage="training_replay", model=name, seed=seed)
            equal = all(row["weights_bitwise_equal"] for row in checks if row["model"] == name and row["seed"] == seed) if compare_retained else None
            print(f"TRAINED {name} seed{seed}: complete; exact historical weights={equal}", flush=True)
    e.write_json(OUT / "training_replay_verification.json", {"status": "passed", "checks": checks,
                 "strict_replay": strict_replay,
                 "historical_comparison_performed": compare_retained,
                 "all_weights_bitwise_equal": all(row["weights_bitwise_equal"] for row in checks) if compare_retained else None,
                 "one_sequential_training_process": True, "threads": 1,
                 "timing_scope": "complete setup, sources, OT, updates, checkpoint I/O; no imports/data checks/reference generation"})


def latency():
    rows, identities = [], []
    rng = random.Random(20261016)
    for name in NAMES:
        _, _, (p, manifest, splits) = e.dataset(name)
        level = 4.5 if p.periodic else 160.
        index, case = next((i, c) for i, c in enumerate(splits["eval"]) if c["level"] == level)
        native, _, _, _ = e.evaluation_case(name, index, case)
        source = torch.from_numpy(np.tile(native, (math.ceil(100000 / len(native)), 1))[:100000].copy())
        methods = list(METHODS) + (["mf_two_step"] if name == "ks3d" else []) + (["bdp"] if name == "kpp2d" else [])
        for seed in SEEDS:
            models = models_for(name, seed, p, manifest["signature"])
            if name == "kpp2d":
                path = BDP / "dplus1" / f"seed{seed}/bdp.pt"
                item = torch.load(path, map_location="cpu", weights_only=True)
                bdp = CompactBDP(p)
                bdp.load_state_dict(item["model"], strict=True)
                models["bdp"] = bdp.cuda().eval().requires_grad_(False)
            sources = {method: source for method in methods}
            if "bdp" in methods:
                sources["bdp"] = torch.from_numpy(np.tile(case["initial"], (5, 1)).copy())
            device_sources = {method: values.cuda() for method, values in sources.items()}
            for method in methods:
                a = sample(method, sources[method], level, p, models)
                b = sample(method, device_sources[method], level, p, models, resident=True).cpu()
                assert torch.equal(a, b)
                if name == "ks3d":
                    expected_path = (ROOT / "output/mf_matched_manuscript_20261016/multistep_particles/sigma160.npz" if method == "mf_two_step" else
                                     (BASE / name / "evaluation" / f"particles_{index:02d}.npz" if method == "mf" else
                                      final_evaluation(name) / f"particles_{index:02d}.npz"))
                    arrays, _ = e.read_npz(expected_path)
                    np.testing.assert_array_equal(a.numpy(), arrays[f"seed{seed}_{method}"])
                identities.append({"model": name, "seed": seed, "method": method, "resident_cpu_bitwise_equal": True})
            for scope in ("gpu_resident", "cpu_to_cpu"):
                resident = scope == "gpu_resident"
                def call(method):
                    return sample(method, device_sources[method] if resident else sources[method], level, p, models, resident=resident)
                for _ in range(10):
                    for method in methods:
                        call(method)
                e.engine.sync("cuda")
                for repeat in range(40):
                    order = list(methods)
                    rng.shuffle(order)
                    for position, method in enumerate(order):
                        e.engine.sync("cuda")
                        started = time.perf_counter()
                        value = call(method)
                        e.engine.sync("cuda")
                        rows.append({"model": name, "level": level, "seed": seed, "particles": 100000,
                                     "scope": scope, "method": method, "repeat": repeat,
                                     "order": position, "wall_ms": (time.perf_counter() - started) * 1000})
                e.CSV(OUT / "inference_raw.csv", rows)
            del models, device_sources
            print(f"LATENCY {name} seed{seed} complete", flush=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"], row["method"], row["scope"], row["seed"]].append(row["wall_ms"])
    per_seed = [dict(zip(("model", "method", "scope", "seed"), key)) |
                {"median_ms": statistics.median(values), "repeats": len(values)} for key, values in sorted(grouped.items())]
    e.CSV(OUT / "inference_per_seed.csv", per_seed)
    e.write_json(OUT / "inference_verification.json", {"status": "passed", "identities": identities,
                 "warmups": 10, "repeats": 40, "particles": 100000, "chunk": 10000,
                 "scope": "synchronized wall time; GPU-resident or CPU-to-CPU; no loading/RNG/metric OT/FK",
                 "periodic_direct_DP": "wrap source before MLP, identical to manuscript evaluation"})


def warm():
    current = sys.modules["experiment"]
    sys.modules["experiment"] = e.uploaded
    try:
        helper = e.import_path("dplus1_paper_warm_helper", ROOT / "output/kpp_original_training_total1k_20261014/warm_starts.py")
    finally:
        sys.modules["experiment"] = current
    gpu = e.uploaded.load_accelerator()
    for name in ("kpp2d", "kpp3d"):
        _, _, (p, manifest, splits) = e.dataset(name)
        level = 4.5
        case = next(c for c in splits["eval"] if c["level"] == level)
        arrays, meta = e.read_npz(MASTER / name / "evaluation/particles_10.npz")
        assert meta["dataset_signature"] == manifest["signature"]
        previous = [v for v in read(BASE / name / "warm_starts/trace.csv") if float(v["level"]) == level]
        traces = [{"model": name, "level": level, "method": "mf", "model_seed": int(v["model_seed"]),
                   "fk_seed": int(v["fk_seed"]), "generation": int(v["generation"]), "c_hat": float(v["c_hat"]),
                   "reference_c": float(v["reference_c"]), "absolute_error": float(v["absolute_error"])}
                  for v in previous if v["method"] == "mf"]
        old_arrays, _ = e.read_npz(BASE / name / "evaluation/particles_10.npz")
        for seed in SEEDS:
            np.testing.assert_array_equal(arrays[f"seed{seed}_mf"], old_arrays[f"seed{seed}_mf"])
        jobs = []
        for physical_seed in (1234, 1235, 1236):
            entries = [("cold", None, None)] + [(method, seed, arrays[f"seed{seed}_{method}"])
                       for seed in SEEDS for method in ("dp", "mf_dp")]
            for method, seed, cloud in entries:
                jobs.append({"dimension": p.dimension, "level": level, "kappa": p.parameter(level), "seed": physical_seed,
                             "particles": 20000, "method": method, "model_seed": seed, "cloud": cloud})
        simulation = helper.make_simulation(gpu, p.dimension, jobs)
        identities, cold_errors = [], []
        for job, initial in zip(jobs, simulation.initial):
            if job["cloud"] is not None:
                np.testing.assert_array_equal(initial, job["cloud"].astype(np.float64))
            identities.append({"method": job["method"], "model_seed": job["model_seed"], "fk_seed": job["seed"],
                               "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest()})
        for generation in range(1, 129):
            simulation.graph.replay()
            if generation & (generation - 1) == 0:
                e.engine.sync("cuda")
                values = ((simulation.kappa + simulation.growth / generation) / simulation.lam).cpu().numpy()
                for job, value in zip(jobs, values):
                    if job["method"] == "cold":
                        old = next(v for v in previous if v["method"] == "cold" and int(v["fk_seed"]) == job["seed"] and int(v["generation"]) == generation)
                        delta = abs(float(value) - float(old["c_hat"]))
                        assert delta < 1e-12
                        cold_errors.append(delta)
                    traces.append({"model": name, "level": level, "method": job["method"], "model_seed": job["model_seed"],
                                   "fk_seed": job["seed"], "generation": generation, "c_hat": float(value),
                                   "reference_c": case["meta"]["c_hat"], "absolute_error": abs(float(value) - case["meta"]["c_hat"])})
                e.CSV(MASTER / name / "warm_starts/trace.csv", traces)
            if generation == 1 or generation % 32 == 0:
                tick(stage="FK_evolution", model=name, generation=generation)
                print(f"FK d+1 {name}: {generation}/128", flush=True)
        assert len(traces) == 240
        e.write_json(MASTER / name / "warm_starts/verification.json", {"status": "passed", "trace_rows": len(traces),
                     "levels": [4.5], "model_seeds": SEEDS, "physical_seeds": [1234, 1235, 1236],
                     "new_DP_and_MFDP_evolution": True, "unchanged_MF_trace_reused_after_bitwise_cloud_check": True,
                     "cold_reproduction_checks": len(cold_errors), "maximum_cold_difference": max(cold_errors), "initial_clouds": identities})
        del simulation
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "training", "latency", "warm", "all"))
    stage = parser.parse_args().stage
    e.configure_device("cuda")
    torch.set_num_threads(1)
    if stage in ("prepare", "all"):
        prepare()
    if stage in ("training", "all"):
        training()
    if stage in ("latency", "all"):
        latency()
    if stage in ("warm", "all"):
        warm()
    tick(stage="complete", selected=stage)


if __name__ == "__main__":
    main()
