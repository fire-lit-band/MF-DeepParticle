"""100k generated and 100k physical-reference particles, frozen 50/50 models.

No training. Exact inherited KS simulator on CUDA float64; common anchored
histogram grid and exact weighted EMD. Preserve original checkpoints/data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
_dll_handles = []
if os.name == "nt":
    _dll_handles.append(os.add_dll_directory(str(Path(sys.prefix) / "Library/bin")))
import torch
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREVIOUS = ROOT / "output/total_budget_refresh50_20261003"
BASE = ROOT / "deepparticle/results/original3_three_seeds"
FIVE_K = ROOT / "output/dp_alpha_lambda_tuning_20261001/budget10k_extension"
sys.path.insert(0, str(PREVIOUS / "runtime"))
from training import core as c, protocol as p

N = 100_000
OLD_N = 10_000
BUDGETS = [10_000, 20_000, 40_000]
SEEDS = [88, 89, 90]
SIGMAS = list(range(10, 201, 10))
METHODS = ["MF", "DP", "hybrid2000", "hybrid4000", "hybrid8000"]
PHYSICS_CHUNK = 256
INFERENCE_CHUNK = 10_000
OT_WORKERS = 3
HARD_TIMEOUT = 7200
SOURCE_SEED = 2026100901
REFERENCE_SEED = 2026100902
NOISE_SEED = 2026100903


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mf_path(seed, step):
    if step == 5000:
        return FIVE_K / f"seed{seed}/mf5k/meanflow_step5000.pt"
    return BASE / f"seed{seed}/mf{step}.pt"


def dp_path(seed, total, method):
    if method == "DP":
        return PREVIOUS / f"pure_dp/seed{seed}/dp_step{total}.pt"
    return PREVIOUS / f"total{total}/batch{method[6:]}/seed{seed}/dp_step{total // 2}.pt"


def unit_ball_seeded(n, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    radius = torch.rand(n, 1, generator=generator).pow(1 / 3)
    direction = torch.randn(n, 3, generator=generator)
    result = radius * direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    assert torch.isfinite(result).all() and torch.linalg.vector_norm(result, dim=1).max() <= 1.000001
    return result


def load_models():
    models, paths = {}, set()
    for seed in SEEDS:
        for step in [5000, 10000, 20000, 40000]:
            path = mf_path(seed, step)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            assert saved["step"] == step
            model = c.Meanflow().cuda().eval().requires_grad_(False)
            model.load_state_dict(saved["model"], strict=True)
            models[seed, "MF", step] = model
            paths.add(path)
        for total in BUDGETS:
            for method in METHODS[1:]:
                path = dp_path(seed, total, method)
                saved = torch.load(path, map_location="cpu", weights_only=True)
                alpha, residual = (.1, 0.) if method == "DP" else (10., .1)
                assert saved["step"] == (total if method == "DP" else total // 2)
                assert saved["alpha"] == alpha and saved["residual_weight"] == residual
                assert saved["ot_refresh"] == saved["batch_refresh"] == 50
                model = c.DPCorrector(width=64, alpha=alpha, blocks=4).cuda().eval().requires_grad_(False)
                model.load_state_dict(saved["model"], strict=True)
                models[seed, method, total] = model
                paths.add(path)
    return models, paths


def preflight():
    c.configure_device("cuda")
    torch.set_num_threads(4)
    checks = []
    for sigma in [10, 100, 200]:
        initial = unit_ball_seeded(1000, REFERENCE_SEED + sigma)
        np.random.seed(NOISE_SEED + sigma)
        cpu = p.simulate_original(initial, sigma, chunk=128, device="cpu")
        np.random.seed(NOISE_SEED + sigma)
        gpu = p.simulate_original(initial, sigma, chunk=128, device="cuda")
        difference = float((cpu - gpu).abs().max())
        assert torch.equal(cpu, gpu), (sigma, difference)
        checks.append(dict(sigma=sigma, particles=1000, cpu_gpu_float32_bitwise_equal=True,
                           max_absolute_difference=difference))
    models, paths = load_models()
    del models
    sources = paths | {BASE / "evaluation_data.pt", BASE / "config.json",
                       PREVIOUS / "metrics_per_seed.csv", PREVIOUS / "config.json",
                       PREVIOUS / "runtime/training/core.py", PREVIOUS / "runtime/training/protocol.py"}
    for total in BUDGETS:
        for sigma in SIGMAS:
            sources.add(PREVIOUS / f"total{total}/evaluation/sigma{sigma}.npz")
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in sorted(sources)}
    physics = json.loads((BASE / "config.json").read_text("utf-8-sig"))["physics"]
    config = dict(status="prepared", generated_particles_per_method_seed_sigma=N,
                  physical_reference_particles_per_sigma=N, previous_particles_per_arm=OLD_N,
                  budgets=BUDGETS, methods=METHODS, training_seeds=SEEDS, sigmas=SIGMAS,
                  new_training_updates=0, new_physical_simulation=True, physics=physics,
                  source_seed_base=SOURCE_SEED, reference_initial_seed_base=REFERENCE_SEED,
                  reference_noise_seed_base=NOISE_SEED, preserve_original_source_10k_prefix=True,
                  simulation_device="cuda", simulation_internal_dtype="float64",
                  output_particle_dtype="float32", physical_simulation_chunk=PHYSICS_CHUNK,
                  inference_chunk=INFERENCE_CHUNK, OT_workers=OT_WORKERS,
                  histogram_protocol="common inherited grid per sigma across all budgets/methods/seeds; extend only, keep spacing",
                  metric="full-mass 3D histogram W2; exact weighted POT emd2 then sqrt",
                  metric_solver_max_iterations=5_000_000, hybrid_alpha=10., hybrid_lambda=.1,
                  pure_dp_alpha=.1, pure_dp_lambda=0., batch_and_OT_training_refresh=50,
                  hard_timeout_seconds=HARD_TIMEOUT,
                  reference_uncertainty="one physical realization shared across model seeds; no confidence interval",
                  size_comparison_caveat="N also changes the finite-particle interacting reference and its realization",
                  user_reference_choice="Generate a new 100000-particle physical reference for every sigma",
                  input_sha256=hashes)
    write_json(OUT / "config.json", config)
    write_json(OUT / "preflight_verification.json", dict(status="passed", cpu_gpu_checks=checks,
               checkpoints_loaded_strictly=len(paths), cuda_device=torch.cuda.get_device_name(),
               torch_version=torch.__version__, numpy_version=np.__version__, POT_version=c.ot.__version__))
    print(f"PREFLIGHT passed; {len(paths)} frozen checkpoints, {len(sources)} immutable inputs", flush=True)


@torch.no_grad()
def infer(model, source, sigma, kind):
    sampler = c.mf_sample if kind == "MF" else c.dp_sample
    outputs = [sampler(model, source[i:i + INFERENCE_CHUNK][None], [sigma])[0]
               for i in range(0, len(source), INFERENCE_CHUNK)]
    result = torch.cat(outputs)
    assert result.shape == (N, 3) and torch.isfinite(result).all()
    return result


def extend_edges(old, low, high):
    result = []
    for dim, edges in enumerate(old):
        width = (edges[-1] - edges[0]) / (len(edges) - 1)
        left = max(0, math.ceil((edges[0] - low[dim]) / width))
        right = max(0, math.ceil((high[dim] - edges[-1]) / width))
        extended = np.concatenate((edges[0] - width * np.arange(left, 0, -1), edges,
                                   edges[-1] + width * np.arange(1, right + 1)))
        np.testing.assert_array_equal(extended[left:left + len(edges)], edges)
        result.append(extended)
    return result


def score(generated, reference, total, method, seed, sigma):
    started = time.perf_counter()
    squared = c.histogram_ot(generated, reference)
    return dict(total_steps=total, method=method, seed=seed, sigma=sigma,
                generated_particles=N, reference_particles=N, w2=math.sqrt(squared),
                w2_squared=squared, generated_nonempty_bins=len(generated[1]),
                reference_nonempty_bins=len(reference[1]), OT_seconds=time.perf_counter() - started)


def summarize(raw):
    curves = []
    for total in BUDGETS:
        for method in METHODS:
            for sigma in SIGMAS:
                selected = [r for r in raw if r["total_steps"] == total and r["method"] == method and r["sigma"] == sigma]
                assert len(selected) == 3 and {r["seed"] for r in selected} == set(SEEDS)
                values = [r["w2"] for r in selected]
                curves.append(dict(total_steps=total, method=method, sigma=sigma,
                                   w2_mean=statistics.mean(values), w2_min=min(values), w2_max=max(values),
                                   w2_sd=statistics.stdev(values),
                                   w2_squared_mean=statistics.mean(r["w2_squared"] for r in selected)))
    aggregate = []
    for total in BUDGETS:
        for method in METHODS:
            data = [r for r in curves if r["total_steps"] == total and r["method"] == method]
            aggregate.append(dict(total_steps=total, method=method,
                mean_w2=statistics.mean(r["w2_mean"] for r in data),
                train_range_mean=statistics.mean(r["w2_mean"] for r in data if 15 <= r["sigma"] <= 150),
                high_sigma_mean=statistics.mean(r["w2_mean"] for r in data if r["sigma"] > 150),
                sigma200=next(r["w2_mean"] for r in data if r["sigma"] == 200)))
    write_csv(OUT / "curve_data.csv", curves)
    write_csv(OUT / "aggregate_summary.csv", aggregate)
    return curves, aggregate


def plot(curves, old):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#365D85", "#62666D", "#D66A32", "#149789", "#855DB3"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    old_curves = []
    for total in BUDGETS:
        for method in METHODS:
            for sigma in SIGMAS:
                values = [float(r["w2"]) for r in old if int(r["total_steps"]) == total and r["method"] == method and int(r["sigma"]) == sigma]
                assert len(values) == 3
                old_curves.append(dict(total_steps=total, method=method, sigma=sigma,
                    w2_mean=statistics.mean(values), w2_min=min(values), w2_max=max(values)))
    ymax = max(.425, max(r["w2_max"] for r in curves + old_curves) * 1.06)

    def draw(ax, total, data, subtitle):
        for i, method in enumerate(METHODS):
            selected = [r for r in data if r["total_steps"] == total and r["method"] == method]
            label = {"MF": f"MF {total // 1000}k", "DP": f"DP {total // 1000}k"}.get(method,
                     f"MF {total // 2000}k + DP {total // 2000}k | B={method[6:]}")
            x = [r["sigma"] for r in selected]
            ax.plot(x, [r["w2_mean"] for r in selected], color=colors[i], linewidth=1.9,
                    marker=["o", "^", "s", "D", "v"][i], markersize=3, markevery=2,
                    linestyle="--" if method == "DP" else "-", label=label)
            ax.fill_between(x, [r["w2_min"] for r in selected], [r["w2_max"] for r in selected],
                            color=colors[i], alpha=.09, linewidth=0)
        ax.axvspan(15, 150, color="#64748B", alpha=.035)
        ax.axvline(150, color="#8B929A", linestyle=":", linewidth=.8)
        ax.set(xlabel=r"$\sigma$", ylabel=r"$W_2$", xlim=(5, 205), ylim=(0, ymax), xticks=[10, 50, 100, 150, 200])
        ax.set_title(f"Total training: {total // 1000}k | {subtitle}", loc="left", pad=12, fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=.17)

    note = ("100,000 generated + 100,000 newly simulated reference particles per sigma. Frozen models; no retraining.\n"
            "Same inherited grid spacing; full 3D mass. Lines: seeds 88/89/90 mean; bands: min-max, not confidence intervals.\n"
            "Hybrid alpha=10, lambda=0.1; pure DP alpha=0.1, lambda=0; training OT/batch refresh=50.")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.subplots_adjust(left=.05, right=.985, bottom=.25, top=.83, wspace=.19)
    for ax, total in zip(axes, BUDGETS):
        draw(ax, total, curves, "100k particles")
    fig.text(.05, .945, "MF / DP / MF+DP: 100k-particle evaluation", fontsize=18, fontweight="bold")
    fig.text(.05, .035, note, fontsize=8.8, color="#4B5563", linespacing=1.6)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"sigma_w2_three_budgets_100k.{ext}", dpi=190, facecolor="white")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    fig.subplots_adjust(left=.05, right=.985, bottom=.16, top=.90, wspace=.19, hspace=.30)
    for j, total in enumerate(BUDGETS):
        draw(axes[0, j], total, old_curves, "original 10k")
        draw(axes[1, j], total, curves, "new 100k")
    fig.text(.05, .96, "10k vs 100k particles: same frozen training checkpoints", fontsize=18, fontweight="bold")
    fig.text(.05, .028, note + "\nThe new reference also changes the finite-particle physical realization; differences are not solely sampling noise.",
             fontsize=8.8, color="#4B5563", linespacing=1.6)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"sigma_w2_10k_vs100k.{ext}", dpi=190, facecolor="white")
    plt.close(fig)
    for total in BUDGETS:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        fig.subplots_adjust(left=.11, right=.97, bottom=.27, top=.90)
        draw(ax, total, curves, "100k particles")
        fig.text(.11, .035, note, fontsize=7.7, color="#4B5563", linespacing=1.5)
        for ext in ["png", "pdf"]:
            fig.savefig(OUT / f"sigma_w2_total{total // 1000}k_100k.{ext}", dpi=190, facecolor="white")
        plt.close(fig)


def run():
    assert json.loads((OUT / "preflight_verification.json").read_text())["status"] == "passed"
    config = json.loads((OUT / "config.json").read_text())
    assert config["status"] == "prepared", "Do not restart or overwrite a partial experiment."
    c.configure_device("cuda")
    torch.set_num_threads(4)
    started = time.perf_counter()
    models, _ = load_models()
    cases = {int(x["sigma"]): x for x in torch.load(BASE / "evaluation_data.pt", map_location="cpu", weights_only=True)["cases"]}
    old_metrics = read_csv(PREVIOUS / "metrics_per_seed.csv")
    (OUT / "particles").mkdir(exist_ok=False)
    config["status"] = "running"
    write_json(OUT / "config.json", config)
    metrics, futures, sigma_checks = [], {}, []
    completed_references = completed_generations = 0

    def progress(phase, sigma=None):
        elapsed = time.perf_counter() - started
        if elapsed > HARD_TIMEOUT:
            raise TimeoutError(f"Declared hard timeout {HARD_TIMEOUT}s reached")
        write_json(OUT / "progress.json", dict(status="running", phase=phase, sigma=sigma,
            pid=os.getpid(), physical_references_completed=completed_references,
            generated_sigma_sets_completed=completed_generations, sigmas_total=len(SIGMAS),
            metric_rows_completed=len(metrics), metric_rows_total=900, elapsed_seconds=elapsed))

    def collect():
        completed = [f for f in futures if f.done()]
        for f in completed:
            row = f.result()
            metrics.append(row)
            del futures[f]
            if len(metrics) == 1 or len(metrics) % 15 == 0:
                print(f"METRICS {len(metrics)}/900; latest sigma={row['sigma']} budget={row['total_steps']} method={row['method']} OT={row['OT_seconds']:.2f}s", flush=True)
        if completed:
            write_csv(OUT / "metrics_per_seed.csv", sorted(metrics, key=lambda r: (r["total_steps"], r["method"], r["sigma"], r["seed"])))

    with ThreadPoolExecutor(max_workers=OT_WORKERS) as executor:
        for sigma in SIGMAS:
            collect()
            progress("physical_reference", sigma)
            print(f"REFERENCE sigma={sigma}: simulate {N} interacting particles", flush=True)
            initial = unit_ball_seeded(N, REFERENCE_SEED + sigma)
            np.random.seed(NOISE_SEED + sigma)
            simulation_started = time.perf_counter()
            reference = p.simulate_original(initial, sigma, mu=config["physics"]["mu"], chi=config["physics"]["chi"],
                delta=config["physics"]["delta"], t=config["physics"]["final_time"], dt=config["physics"]["dt"],
                chunk=PHYSICS_CHUNK, device="cuda")
            simulation_seconds = time.perf_counter() - simulation_started
            completed_references += 1
            source = torch.cat((cases[sigma]["source"], unit_ball_seeded(N - OLD_N, SOURCE_SEED + sigma)))
            np.testing.assert_array_equal(source[:OLD_N].numpy(), cases[sigma]["source"].numpy())
            np.savez(OUT / f"particles/sigma{sigma}_inputs.npz", source=source.numpy(), reference_initial=initial.numpy(), reference=reference.numpy())
            progress("model_generation", sigma)
            clouds = {}
            for seed in SEEDS:
                mf_clouds = {step: infer(models[seed, "MF", step], source, sigma, "MF") for step in [5000, 10000, 20000, 40000]}
                for total in BUDGETS:
                    clouds[total, "MF", seed] = mf_clouds[total].numpy()
                    for method in METHODS[1:]:
                        input_cloud = source if method == "DP" else mf_clouds[total // 2]
                        clouds[total, method, seed] = infer(models[seed, method, total], input_cloud, sigma, "DP").numpy()
            saved_old = {}
            prefix_checks = []
            for total in BUDGETS:
                with np.load(PREVIOUS / f"total{total}/evaluation/sigma{sigma}.npz", allow_pickle=False) as cached:
                    saved_old[total] = {key: cached[key].copy() for key in cached.files}
                for method in METHODS:
                    for seed in SEEDS:
                        old = saved_old[total][f"seed{seed}_{method}"]
                        fresh = clouds[total, method, seed][:OLD_N]
                        difference = float(np.max(np.abs(fresh - old)))
                        np.testing.assert_allclose(fresh, old, rtol=1e-6, atol=1e-6)
                        prefix_checks.append(dict(total=total, method=method, seed=seed,
                                                  bitwise_equal=bool(np.array_equal(fresh, old)), max_absolute_difference=difference))
            existing_reference = saved_old[BUDGETS[0]]["reference"]
            all_clouds = list(clouds.values()) + [reference.numpy(), existing_reference]
            low = np.min([a.min(0) for a in all_clouds], axis=0)
            high = np.max([a.max(0) for a in all_clouds], axis=0)
            anchor = [saved_old[BUDGETS[0]][f"edges_{axis}"] for axis in "xyz"]
            edges = extend_edges(anchor, low, high)
            reference_measure = c.histogram(reference.numpy(), edges)
            assert reference_measure[2] == N
            for total in BUDGETS:
                payload = {f"seed{seed}_{method}": clouds[total, method, seed] for seed in SEEDS for method in METHODS}
                np.savez(OUT / f"particles/total{total}_sigma{sigma}.npz", edges_x=edges[0], edges_y=edges[1], edges_z=edges[2], **payload)
                for method in METHODS:
                    for seed in SEEDS:
                        generated = c.histogram(clouds[total, method, seed], edges)
                        assert generated[2] == N
                        f = executor.submit(score, generated, reference_measure, total, method, seed, sigma)
                        futures[f] = (total, method, seed, sigma)
            completed_generations += 1
            check = dict(sigma=sigma, reference_particles=len(reference), source_particles=len(source),
                         model_clouds=len(clouds), model_particles_per_cloud=N,
                         reference_seconds=simulation_seconds, grid_shape=[len(e) - 1 for e in edges],
                         grid_spacing=[float(e[1] - e[0]) for e in anchor],
                         all_particles_retained=True, common_grid_across_budgets=True,
                         max_old_prefix_difference=max(x["max_absolute_difference"] for x in prefix_checks),
                         exact_old_prefixes=sum(x["bitwise_equal"] for x in prefix_checks), prefix_checks=prefix_checks)
            sigma_checks.append(check)
            write_json(OUT / "particle_verification.json", dict(status="running", checks=sigma_checks))
            collect()
            progress("reference_and_generation_done", sigma)
            print(f"GENERATED sigma={sigma}: reference={N}, 45 model clouds x {N}; physics={simulation_seconds:.2f}s; grid={check['grid_shape']}", flush=True)
        while futures:
            collect()
            progress("OT_evaluation")
            if futures:
                time.sleep(2)
    assert len(metrics) == 900
    assert len({(r["total_steps"], r["method"], r["seed"], r["sigma"]) for r in metrics}) == 900
    assert all(math.isfinite(r["w2"]) and math.isclose(r["w2"] ** 2, r["w2_squared"], rel_tol=1e-12, abs_tol=1e-14) for r in metrics)
    curves, aggregate = summarize(metrics)
    plot(curves, old_metrics)
    immutable = {rel: sha(ROOT / rel) == digest for rel, digest in config["input_sha256"].items()}
    assert all(immutable.values()), "Original source/model/data changed during evaluation"
    elapsed = time.perf_counter() - started
    write_json(OUT / "particle_verification.json", dict(status="passed", checks=sigma_checks,
               all_model_particles_generated=90_000_000, physical_reference_particles=2_000_000))
    write_json(OUT / "verification.json", dict(status="passed", new_training_updates=0,
        new_physical_simulation=True, particles_per_arm=N, sigma_count=20, metric_rows=900,
        curve_points=300, completed_budget_method_groups=15, all_particles_retained=True,
        common_grid_across_methods_seeds_budgets=True, grid_spacing_preserved=True,
        exact_weighted_OT=True, metric_relation_verified=True, immutable_inputs_unchanged=True,
        checked_input_files=len(immutable), generated_cloud_10k_prefixes_checked=900,
        generated_cloud_10k_prefixes_bitwise_equal=sum(x["exact_old_prefixes"] for x in sigma_checks),
        maximum_prefix_difference=max(x["max_old_prefix_difference"] for x in sigma_checks),
        elapsed_seconds=elapsed, output_particle_files=80,
        caveats=["Three model seeds share one new physical reference realization per sigma.",
                 "Finite-particle physics uses J=100000; old J=10000 reference is a different realization.",
                 "Grid distance is exact for the binned measures, approximate for raw point clouds.",
                 "Bands are model-seed min/max, not confidence intervals."]))
    config["status"] = "completed"
    config["elapsed_seconds"] = elapsed
    write_json(OUT / "config.json", config)
    write_json(OUT / "progress.json", dict(status="completed", pid=os.getpid(),
        physical_references_completed=20, generated_sigma_sets_completed=20, metric_rows_completed=900,
        metric_rows_total=900, elapsed_seconds=elapsed))
    print(f"COMPLETED: 900 metrics; {elapsed:.1f}s; all original inputs unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    try:
        preflight() if args.prepare else run()
    except BaseException as exc:
        write_json(OUT / "failure.json", dict(status="failed", error=str(exc), traceback=traceback.format_exc(), pid=os.getpid()))
        print(traceback.format_exc(), flush=True)
        raise
