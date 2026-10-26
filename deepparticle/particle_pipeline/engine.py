"""Fixed-budget MF/DP training and genuine cold/learned-start FK evaluation."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import statistics
import sys
import time
import zipfile

import numpy as np
import torch
from torch.func import jvp

from .data import load_dataset, prior, write_json, write_npz
from .geometry import difference, exact_assignment, cloud_metric, histogram_metrics
from .models import MeanFlow, Corrector, generate
from .physics import kpp_fk
from .protocol import file_hash


def csv_write(path, rows):
    path = Path(path)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))


def mf_condition(model, level, protocol, count, device, *, training=False):
    if model.condition_mode == "log10_sigma":
        value = float(torch.log10(torch.tensor(level, dtype=torch.float32))) if training else math.log10(level)
    else:
        value = protocol.condition(level)
    return torch.full((count, 1), value, device=device)


def checkpoint(path, model, optimizer, *, step, protocol, dataset_signature, architecture):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "step": step,
                "protocol": protocol.metadata(), "dataset_signature": dataset_signature,
                "architecture": architecture, "torch_version": str(torch.__version__)}, temporary)
    temporary.replace(path)


def mf_dimensions(args):
    """Keep the historical shared width unless an MF-only width is supplied."""
    width = getattr(args, "mf_width", None)
    return args.width if width is None else width, getattr(args, "mf_hidden_layers", 3)


def train_mf(cases, protocol, args, folder, dataset_signature, seed, *, sampling_rng_state=None, sampling_trace=None):
    original = args.profile == "original_tuned"
    if original:
        from training import core as original_core
        original_core.set_seed(seed)
    else:
        set_seed(seed)
    mode = "log10_sigma" if original and protocol.family == "ks" else "protocol"
    width, hidden_layers = mf_dimensions(args)
    model = MeanFlow(protocol.dimension, width, mode, hidden_layers).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mf_lr)
    if sampling_rng_state is not None or sampling_trace is not None:
        if not original:
            raise ValueError("Paired architecture sampling requires the original_tuned profile")
        if sampling_rng_state is not None:
            torch.set_rng_state(sampling_rng_state)
    rng = np.random.default_rng(seed + 100003)
    total = args.mf_steps + args.dp_steps
    architecture = {"kind": "mf", "width": width, "hidden_layers": hidden_layers,
                    "conditioning": "log10(sigma)" if mode == "log10_sigma" else "protocol.condition(level)",
                    "condition_mode": mode, "profile": args.profile}
    rows, initialized, initializer_seconds = [], None, None
    sync(args.device)
    start = time.perf_counter()
    tensors = [{k: torch.from_numpy(case[k]).to(args.device) for k in ("prior", "target")} for case in cases]
    for step in range(1, total + 1):
        i = int(torch.randint(len(cases), (1,)).item()) if original else int(rng.integers(len(cases)))
        indices = torch.randint(len(cases[i]["target"]), (args.batch_size,)) if original else torch.from_numpy(rng.integers(len(cases[i]["target"]), size=args.batch_size))
        idx = indices.to(args.device)
        source, target = tensors[i]["prior"][idx], tensors[i]["target"][idx]
        if protocol.periodic:
            target = source + difference(target, source, True)
        if original:
            from training.protocol import sample_times_original
            r, t = sample_times_original(args.batch_size)
            if sampling_trace is not None:
                # The sweep hashes actual condition/batch/time draws, not just seeds.
                for values in (np.asarray([i], dtype=np.int64), indices.numpy(), r.numpy(), t.numpy()):
                    sampling_trace.update(values.tobytes())
            r, t = r.to(args.device), t.to(args.device)
        else:
            samples = rng.normal(-.4, 1., (args.batch_size, 2))
            samples = 1 / (1 + np.exp(-samples))
            r = torch.tensor(samples.min(1)[:, None], dtype=torch.float32, device=args.device)
            t = torch.tensor(samples.max(1)[:, None], dtype=torch.float32, device=args.device)
            diagonal = torch.randperm(args.batch_size, device=args.device)[:args.batch_size // 2]
            r[diagonal] = t[diagonal]
        condition = mf_condition(model, cases[i]["level"], protocol, args.batch_size, args.device, training=True)
        velocity = source - target
        z = (1 - t) * target + t * source
        u, derivative = jvp(lambda zz, rr, tt: model(zz, rr, tt, condition),
                            (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)))
        desired = (velocity - (t - r) * derivative).detach()
        loss = (u - desired).square().sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite MF loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        rows.append({"step": step, "loss": loss.item(), "level": cases[i]["level"]})
        if step == args.mf_steps:
            sync(args.device)
            initializer_seconds = time.perf_counter() - start
            initialized = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            checkpoint(folder / "mf_init.pt", model, optimizer, step=step, protocol=protocol,
                       dataset_signature=dataset_signature, architecture=architecture)
        if step == 1 or step % 250 == 0 or step == total:
            print(f"{protocol.name} seed={seed} MF {step}/{total} loss={loss.item():.5g}", flush=True)
            csv_write(folder / "mf_loss.csv", rows)
        if step % 500 == 0 or step == total:
            checkpoint(folder / "mf.pt", model, optimizer, step=step, protocol=protocol,
                       dataset_signature=dataset_signature, architecture=architecture)
    sync(args.device)
    return model, initialized, {"trajectory_seconds": time.perf_counter() - start,
                                "initializer_seconds": initializer_seconds}


def train_dp(cases, protocol, args, folder, dataset_signature, seed, mf=None, assignment_solver=None,
             corrector_factory=None):
    original = args.profile == "original_tuned"
    if original:
        from training import core as original_core
        original_core.set_seed(seed)
    else:
        set_seed(seed + 101)
    alpha = args.hybrid_alpha if mf is not None and args.hybrid_alpha is not None else args.alpha
    rng = np.random.default_rng(seed + 200003)
    total = args.dp_steps if mf is not None else args.mf_steps + args.dp_steps
    method = "mf_dp" if mf is not None else "dp"
    solve_assignment = exact_assignment if assignment_solver is None else assignment_solver
    residual_weight = args.residual_weight if mf is not None else 0.
    architecture = {"kind": "dp", "width": args.width, "blocks": args.dp_blocks, "alpha": alpha,
                    "periodic_position_encoding": protocol.periodic,
                    "conditioning": "protocol.condition(level)", "profile": args.profile}
    pools = []
    for case in cases:
        if original and not protocol.periodic:
            radius = torch.rand(len(case["prior"]), 1).pow(1 / protocol.dimension)
            directions = torch.randn(len(case["prior"]), protocol.dimension)
            source = (radius * directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)).to(args.device)
        else:
            source = torch.from_numpy(case["prior"]).to(args.device)
        condition = torch.full((len(source), 1), protocol.condition(case["level"]), device=args.device)
        if mf is not None:
            source = generate("mf", source, condition, protocol, mf=mf,
                              mf_condition=mf_condition(mf, case["level"], protocol, len(source), args.device))
        elif protocol.periodic:
            source = torch.remainder(source, 2 * math.pi)
        pools.append((source.detach(), torch.from_numpy(case["target"]).to(args.device)))
    factory = Corrector if corrector_factory is None else corrector_factory
    model = factory(protocol.dimension, protocol.periodic, args.width, args.dp_blocks, alpha).to(args.device)
    if corrector_factory is not None:
        architecture.update(getattr(model, 'architecture', {}))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.dp_lr)
    batches, rows, solve_seconds, solves = [], [], 0., 0
    sync(args.device)
    start = time.perf_counter()
    for step in range(1, total + 1):
        changed = not batches or (step - 1) % args.batch_refresh == 0
        if changed:
            batches = []
            if original:
                if len({(len(source), len(target)) for source, target in pools}) != 1:
                    raise ValueError("Original profile requires equal-size condition pools")
                shared_n = min(args.batch_size, len(pools[0][0]), len(pools[0][1]))
                shared_si = torch.randperm(len(pools[0][0]))[:shared_n].to(args.device)
                shared_ti = torch.randperm(len(pools[0][1]))[:shared_n].to(args.device)
            for case, (source_pool, target_pool) in zip(cases, pools):
                n = min(args.batch_size, len(source_pool), len(target_pool))
                si = shared_si if original else torch.as_tensor(rng.choice(len(source_pool), n, replace=False), device=args.device)
                ti = shared_ti if original else torch.as_tensor(rng.choice(len(target_pool), n, replace=False), device=args.device)
                condition = torch.full((n, 1), protocol.condition(case["level"]), device=args.device)
                batches.append({"source": source_pool[si], "target": target_pool[ti], "condition": condition})
        if changed or (step - 1) % args.ot_refresh == 0:
            for batch in batches:
                with torch.no_grad():
                    current = model(batch["source"], batch["condition"]).cpu().numpy()
                solve_start = time.perf_counter()
                permutation = solve_assignment(current, batch["target"].cpu().numpy(), protocol.periodic)
                solve_seconds += time.perf_counter() - solve_start
                solves += 1
                batch["matched"] = batch["target"][torch.as_tensor(permutation, device=args.device)]
        transport, residual = [], []
        for batch in batches:
            refined = model(batch["source"], batch["condition"])
            transport.append(difference(refined, batch["matched"], protocol.periodic).square().sum(-1).mean())
            residual.append(difference(refined, batch["source"], protocol.periodic).square().sum(-1).mean())
        ot_loss, penalty = torch.stack(transport).mean(), torch.stack(residual).mean()
        loss = ot_loss + residual_weight * penalty
        if original:
            loss = loss * sum(len(batch["source"]) for batch in batches)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite DP loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        rows.append({"step": step, "loss": loss.item(), "transport": ot_loss.item(), "displacement": penalty.item()})
        if step == 1 or step % 250 == 0 or step == total:
            print(f"{protocol.name} seed={seed} {method} {step}/{total} loss={loss.item():.5g}", flush=True)
            csv_write(folder / f"{method}_loss.csv", rows)
        if step % 500 == 0 or step == total:
            checkpoint(folder / f"{method}.pt", model, optimizer, step=step, protocol=protocol,
                       dataset_signature=dataset_signature, architecture=architecture)
    sync(args.device)
    return model, {"seconds": time.perf_counter() - start, "ot_seconds": solve_seconds,
                   "ot_solves": solves, "steps": total, "residual_weight": residual_weight}


def train(dataset, out, args):
    protocol, manifest, splits = load_dataset(dataset)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Training requires a new run directory")
    if min(args.mf_steps, args.dp_steps, args.batch_size, args.width, args.dp_blocks, args.batch_refresh, args.ot_refresh, *mf_dimensions(args)) < 1:
        raise ValueError("Training budgets, sizes and refresh periods must be positive")
    if args.alpha <= 0 or args.residual_weight < 0 or min(args.mf_lr, args.dp_lr) <= 0:
        raise ValueError("Invalid training hyperparameters")
    if args.hybrid_alpha is not None and args.hybrid_alpha <= 0:
        raise ValueError("Hybrid alpha must be positive")
    if args.profile == "original_tuned":
        from training.core import configure_device
        configure_device(args.device)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Model seeds must be distinct")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Model seeds must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose cpu explicitly")
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset)
    dataset_path = dataset_path.parent if dataset_path.name == "manifest.json" else dataset_path
    config = {"dataset": str(dataset_path.resolve()), "dataset_signature": manifest["signature"],
              "protocol": protocol.metadata(), "evidence": protocol.evidence(), "purpose": manifest["purpose"],
              "training": {key: getattr(args, key) for key in ("mf_steps", "dp_steps", "mf_lr", "dp_lr", "width", "dp_blocks", "alpha", "hybrid_alpha", "profile", "batch_size", "batch_refresh", "ot_refresh", "residual_weight", "seeds", "device")},
              "budgets": {"mf": args.mf_steps + args.dp_steps, "mf_init": args.mf_steps,
                          "dp": args.mf_steps + args.dp_steps, "mf_dp": args.mf_steps + args.dp_steps},
              "network_evaluations": {"mf": 1, "mf_init": 1, "dp": 1, "mf_dp": 2},
              "status": "training"}
    config["training"].update(mf_width=mf_dimensions(args)[0], mf_hidden_layers=mf_dimensions(args)[1])
    write_json(out / "config.json", config)
    import ot
    write_json(out / "runtime.json", {"python": sys.version.split()[0], "torch": str(torch.__version__),
               "numpy": np.__version__, "POT": ot.__version__, "device": args.device,
               "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None})
    package = Path(__file__).parent
    source_files = list(package.glob("*.py")) + [package.parent / "run_particles.py", package.parent / "training" / "core.py"]
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
        for source in source_files:
            snapshot.write(source, source.relative_to(package.parent).as_posix())
    write_json(out / "source_manifest.json", {source.relative_to(package.parent).as_posix(): file_hash(source) for source in source_files})
    model_manifest, timings = {}, []
    for seed in args.seeds:
        folder = out / f"seed{seed}"
        folder.mkdir()
        _, initializer_state, mf_seconds = train_mf(splits["train"], protocol, args, folder, manifest["signature"], seed)
        mode = "log10_sigma" if args.profile == "original_tuned" and protocol.family == "ks" else "protocol"
        mf_width, mf_layers = mf_dimensions(args)
        mf_init = MeanFlow(protocol.dimension, mf_width, mode, mf_layers).to(args.device)
        mf_init.load_state_dict(initializer_state)
        mf_init.requires_grad_(False)
        _, dp_timing = train_dp(splits["train"], protocol, args, folder, manifest["signature"], seed)
        _, mixed_timing = train_dp(splits["train"], protocol, args, folder, manifest["signature"], seed, mf=mf_init)
        model_manifest[str(seed)] = {method: {"file": (folder / f"{method}.pt").relative_to(out).as_posix(),
                                              "sha256": file_hash(folder / f"{method}.pt")}
                                    for method in ("mf", "mf_init", "dp", "mf_dp")}
        timings.append({"seed": seed, "mf": mf_seconds, "pure_dp": dp_timing, "mf_dp_refinement": mixed_timing,
                        "mf_dp_training_seconds": mf_seconds["initializer_seconds"] + mixed_timing["seconds"]})
        write_json(out / "checkpoint_manifest.json", model_manifest)
        write_json(out / "training_timings.json", timings)
    config["status"] = "complete"
    write_json(out / "config.json", config)
    return out


def read_models(run, seed, config, protocol, device):
    manifest = json.loads((run / "checkpoint_manifest.json").read_text(encoding="utf-8"))[str(seed)]
    models = {}
    for kind, item in manifest.items():
        path = (run / item["file"]).resolve()
        if not path.is_relative_to(run.resolve()) or file_hash(path) != item["sha256"]:
            raise ValueError("Model checkpoint checksum/path differs")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["protocol"] != protocol.metadata() or payload["dataset_signature"] != config["dataset_signature"]:
            raise ValueError("Checkpoint belongs to a different physical model or dataset")
        hp = config["training"]
        architecture = payload["architecture"]
        model = (MeanFlow(protocol.dimension, architecture.get("width", hp.get("mf_width", hp["width"])),
                          architecture.get("condition_mode", "protocol"), architecture.get("hidden_layers", 3))
                 if kind.startswith("mf") and kind != "mf_dp" else
                 Corrector(protocol.dimension, protocol.periodic, architecture.get("width", hp["width"]),
                           architecture.get("blocks", hp["dp_blocks"]), architecture["alpha"]))
        model.load_state_dict(payload["model"], strict=True)
        model.to(device).eval().requires_grad_(False)
        models[kind] = model
    return models


def infer(method, source, case, protocol, models, device):
    sync(device)
    start = time.perf_counter()
    source = torch.from_numpy(source).to(device)
    condition = torch.full((len(source), 1), protocol.condition(case["level"]), device=device)
    mf = models["mf"] if method == "mf" else models["mf_init"]
    dp = models["mf_dp"] if method == "mf_dp" else models["dp"]
    particles = generate("mf" if method == "mf_init" else method, source, condition, protocol, mf=mf, dp=dp,
                         mf_condition=mf_condition(mf, case["level"], protocol, len(source), device))
    if not torch.isfinite(particles).all():
        raise FloatingPointError("Nonfinite generated particles")
    particles = particles.cpu().numpy()
    sync(device)
    return particles, time.perf_counter() - start


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["level"], row["method"], row["metric"])
        groups.setdefault(key, []).append(row)
    output = []
    for (level, method, metric), items in sorted(groups.items()):
        first = items[0]
        row = {"protocol": first["protocol"], "level": level, "parameter": first["parameter"],
               "method": method, "metric": metric, "model_runs": len(items)}
        for field in ("w2", "w2_squared"):
            values = [item[field] for item in items]
            row[field + "_mean"] = statistics.mean(values)
            row[field + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        output.append(row)
    return output


def _fk_job(job):
    result = kpp_fk(job["level"], job["protocol"], seed=job["fk_seed"],
                    particles=job["particles"], initial=job["cloud"], generations=job["generations"])
    return job, result


def evaluate(run, args):
    run = Path(run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if config["status"] != "complete":
        raise ValueError("Models are incomplete")
    protocol, manifest, splits = load_dataset(config["dataset"])
    if manifest["signature"] != config["dataset_signature"]:
        raise ValueError("Evaluation dataset changed")
    if args.fk and protocol.periodic and min(args.fk_generations, args.fk_repeats) < 1:
        raise ValueError("FK budgets and repeats must be positive")
    if args.fk_workers < 1:
        raise ValueError("FK worker count must be positive")
    if min(args.metric_particles, args.metric_workers) < 1 or (args.bins is not None and args.bins < 2) or args.fk_tolerance < 0:
        raise ValueError("Metric sizes/bins and FK tolerance are invalid")
    if args.fk and protocol.periodic and args.fk_levels and not any(
            any(math.isclose(case["level"], level) for level in args.fk_levels) for case in splits["eval"]):
        raise ValueError("No requested FK level is present in the evaluation dataset")
    output = run / "evaluation"
    output.mkdir(exist_ok=True)
    all_models = {seed: read_models(run, seed, config, protocol, args.device) for seed in config["training"]["seeds"]}
    rows, fk_rows, fk_runs, fk_jobs = [], [], [], []
    methods = ("mf", "mf_init", "dp", "mf_dp")
    for i, case in enumerate(splits["eval"]):
        clouds, inference = {}, {}
        for seed, models in all_models.items():
            for method in methods:
                key = f"seed{seed}_{method}"
                clouds[key], inference[key] = infer(method, case["prior"], case, protocol, models, args.device)
        bins = args.bins or (72 if protocol.periodic else 24)
        measures = {}
        if protocol.dimension <= 3:
            projection = (0, 2) if protocol.periodic and protocol.dimension == 3 else None
            metric = "periodic_histogram_xz" if projection else "periodic_histogram" if protocol.periodic else "euclidean_histogram"
            measures[metric] = histogram_metrics(case["target"], clouds, bins, protocol.periodic, projection, workers=args.metric_workers)
        if protocol.dimension >= 3:
            measures["periodic_full_cloud" if protocol.periodic else "euclidean_full_cloud"] = {
                key: cloud_metric(case["target"], value, protocol.periodic, args.metric_particles, args.metric_seed)
                for key, value in clouds.items()}
        for metric, results in measures.items():
            for seed in all_models:
                for method in methods:
                    key = f"seed{seed}_{method}"
                    values = results[key]
                    rows.append({"protocol": protocol.name, "level": case["level"], "parameter": case["parameter"],
                                 "model_seed": seed, "method": method, "metric": metric,
                                 "w2": values["w2"], "w2_squared": values["w2_squared"],
                                 "particles": len(clouds[key]), "inference_seconds": inference[key],
                                 "network_evaluations": config["network_evaluations"][method]})
        write_npz(output / f"particles_{i:02d}.npz", meta={"protocol": protocol.metadata(), "level": case["level"],
                  "dataset_signature": manifest["signature"], "inference_seconds": inference}, reference=case["target"], **clouds)
        print(f"{protocol.name} evaluation level={case['level']:g}: actual generated clouds and OT metrics saved", flush=True)
        if args.fk and protocol.periodic and (not args.fk_levels or any(math.isclose(case["level"], l) for l in args.fk_levels)):
            reference_c = case["meta"].get("c_hat")
            if reference_c is None:
                raise ValueError("KPP reference lacks a canonical eigenvalue estimate")
            for repeat in range(args.fk_repeats):
                evolution_seed = args.fk_seed + repeat
                initializations = [(None, "cold", None, 0.)] + [
                    (seed, method, clouds[f"seed{seed}_{method}"], inference[f"seed{seed}_{method}"])
                    for seed in all_models for method in ("mf", "mf_init", "mf_dp", "dp")]
                for model_seed, method, cloud, sampling_seconds in initializations:
                    fk_jobs.append({"protocol": protocol, "level": case["level"], "model_seed": model_seed,
                                    "method": method, "cloud": cloud, "particles": len(case["target"]),
                                    "fk_seed": evolution_seed, "generations": args.fk_generations,
                                    "sampling_seconds": sampling_seconds, "reference_c": reference_c})
    summary = summarize(rows)
    csv_write(output / "metrics_per_seed.csv", rows)
    csv_write(output / "summary.csv", summary)
    def record_fk(item):
        job, result = item
        first_hit = None
        for record in result["trace"]:
            error = abs(record["c"] - job["reference_c"])
            if first_hit is None and error <= args.fk_tolerance:
                first_hit = record["generation"]
            fk_rows.append({"protocol": protocol.name, "level": job["level"], "model_seed": job["model_seed"],
                            "method": job["method"], "fk_seed": job["fk_seed"], "generation": record["generation"],
                            "mu_hat": record["mu"], "c_hat": record["c"], "reference_c": job["reference_c"],
                            "absolute_error": error, "fk_seconds": record["seconds"],
                            "sampling_plus_fk_seconds": job["sampling_seconds"] + record["seconds"],
                            "backend": result.get("backend", "numpy_cpu"), "evolution_seed": job["fk_seed"] + 7919})
        fk_runs.append({"level": job["level"], "model_seed": job["model_seed"], "method": job["method"],
                        "fk_seed": job["fk_seed"], "generations": args.fk_generations,
                        "first_recorded_tolerance_hit": first_hit, "tolerance": args.fk_tolerance,
                        "meaning": "first recorded hit, not a proof of sustained convergence",
                        "sampling_seconds": job["sampling_seconds"], "fk_seconds": result["seconds"],
                        "backend": result.get("backend", "numpy_cpu"), "evolution_seed": job["fk_seed"] + 7919})
        write_json(output / "progress.json", {"protocol": protocol.name, "stage": "fk",
                   "completed_runs": len(fk_runs), "total_runs": len(fk_jobs), "workers": args.fk_workers})
        if len(fk_runs) == 1 or len(fk_runs) % 10 == 0 or len(fk_runs) == len(fk_jobs):
            print(f"{protocol.name} actual FK warm/cold runs {len(fk_runs)}/{len(fk_jobs)}", flush=True)
    fk_backend = args.device if args.fk_backend == "auto" else args.fk_backend
    if fk_jobs and fk_backend == "cuda":
        from .fk_batch import kpp_fk_batch
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        # Leave ample headroom for the OS and concurrently running applications.
        bytes_per_job = len(splits["eval"][0]["target"]) * protocol.dimension * 8 * 20
        ensemble_size = max(1, min(64, int(free * .25 / bytes_per_job)))
        for begin in range(0, len(fk_jobs), ensemble_size):
            selected_jobs = fk_jobs[begin:begin + ensemble_size]
            def progress(g, total):
                write_json(output / "progress.json", {"protocol": protocol.name, "stage": "fk_cuda_ensemble",
                           "completed_runs": len(fk_runs), "total_runs": len(fk_jobs),
                           "ensemble_runs": len(selected_jobs), "generation": g, "total_generations": total})
                print(f"{protocol.name} FK GPU ensemble {begin + 1}-{begin + len(selected_jobs)}/{len(fk_jobs)}: {g}/{total}", flush=True)
            for result in kpp_fk_batch(selected_jobs, device="cuda", progress=progress):
                record_fk(result)
    elif args.fk_workers == 1:
        for job in fk_jobs:
            record_fk(_fk_job(job))
    elif fk_jobs:
        with ThreadPoolExecutor(max_workers=args.fk_workers) as executor:
            futures = [executor.submit(_fk_job, job) for job in fk_jobs]
            for future in as_completed(futures):
                record_fk(future.result())
    # Output row order is independent of worker completion order.
    fk_rows.sort(key=lambda row: (row["level"], row["fk_seed"], row["method"], row["model_seed"] or -1, row["generation"]))
    fk_runs.sort(key=lambda row: (row["level"], row["fk_seed"], row["method"], row["model_seed"] or -1))
    csv_write(output / "fk_trace.csv", fk_rows)
    write_json(output / "fk_runs.json", fk_runs)
    report = {"status": "complete", "purpose": config["purpose"], "protocol": protocol.metadata(),
              "dataset_signature": manifest["signature"], "metrics_rows": len(rows), "fk_trace_rows": len(fk_rows),
              "uncertainty": "metric SD is across model seeds on fixed physical data; FK seeds are reported separately",
              "timing": "diagnostic per-call wall times include model inference and transfers for a stored prior; not a repeated hardware benchmark",
              "reference": "finite-time, finite-particle FK estimate; not exact minimal front speed",
              "evaluation_options": {key: getattr(args, key) for key in ("bins", "metric_particles", "metric_seed", "metric_workers", "fk", "fk_generations", "fk_repeats", "fk_seed", "fk_levels", "fk_tolerance", "fk_workers", "fk_backend")},
              "fk_backend": fk_backend if fk_jobs else None,
              "fk_timing_note": "Not applicable" if not fk_jobs else "Ensemble/concurrent timings must not be used to claim a warm-start speedup" if fk_backend == "cuda" or args.fk_workers > 1 else "Sequential per-call diagnostic timings",
              "summary": summary}
    write_json(output / "verification.json", report)
    package = Path(__file__).parent
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
        for source in package.glob("*.py"):
            snapshot.write(source, source.relative_to(package.parent).as_posix())
    write_json(output / "source_manifest.json", {source.name: file_hash(source) for source in package.glob("*.py")})
    from .plotting import plot_results
    plot_results(output, protocol, rows, fk_rows, config["purpose"])
    return output / "verification.json"
