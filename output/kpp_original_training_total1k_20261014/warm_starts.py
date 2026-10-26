"""Paired cold/learned FK evolution using the uploaded equations and resampling."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import experiment as exp
from particle_pipeline.data import read_npz, write_json, write_npz
from particle_pipeline.engine import csv_write
from particle_pipeline.protocol import file_hash

PHYSICAL_SEEDS = (1234, 1235, 1236)
METHODS = ("mf", "mf_init", "dp", "mf_dp")
LEVELS = (3.75, 4.5)


def source_clouds(model, level, signature):
    index = exp.EVAL_LEVELS.index(level)
    path = exp.OUT / model / "warm_inputs" / f"particles_{index:02d}.npz"
    arrays, meta = read_npz(path)
    if meta["dataset_signature"] != signature or meta["protocol"] != exp.protocol(model).metadata():
        raise ValueError("Warm-start particles have a different reference protocol")
    return arrays, meta, path


def prepare_inputs(model, p, manifest, splits, config):
    """Generate representative particles while independent OT evaluation runs."""
    folder = exp.OUT / model / "warm_inputs"
    folder.mkdir(parents=True, exist_ok=True)
    models = {seed: exp.engine.read_models(exp.OUT / model / "run", seed, config, p, "cuda")
              for seed in exp.MODEL_SEEDS}
    for level in LEVELS:
        case = next(c for c in splits["eval"] if c["level"] == level)
        clouds, timings = {}, {}
        for seed, selected in models.items():
            for method in METHODS:
                key = f"seed{seed}_{method}"
                clouds[key], timings[key] = exp.engine.infer(method, case["prior"], case, p, selected, "cuda")
        path = folder / f"particles_{exp.EVAL_LEVELS.index(level):02d}.npz"
        write_npz(path, meta={"protocol": p.metadata(), "dataset_signature": manifest["signature"],
                  "level": level, "inference_seconds": timings}, reference=case["target"], **clouds)
    del models
    torch.cuda.empty_cache()


def make_simulation(gpu, dimension, jobs):
    gpu.SEEDS = PHYSICAL_SEEDS
    gpu.jobs_for = lambda d, n: jobs
    simulation = gpu.OriginalGPU(dimension, 20000, graph=False)
    initial = [np.random.default_rng(job["seed"]).uniform(0, gpu.TWO_PI, (20000, dimension))
               if job["cloud"] is None else np.asarray(job["cloud"], dtype=np.float64).copy() for job in jobs]
    simulation.initial = initial
    simulation.q.copy_(torch.tensor(np.stack(initial), dtype=torch.float64, device="cuda"))
    simulation.pre_selection.copy_(simulation.q)
    simulation.capture()
    return simulation


def run(model):
    torch.set_num_threads(1)
    p, manifest, splits = exp.load_dataset(exp.OUT / "datasets" / model)
    config = json.loads((exp.OUT / model / "run/config.json").read_text(encoding="utf-8"))
    if config["status"] != "complete" or config["dataset_signature"] != manifest["signature"]:
        raise ValueError("Models or reference dataset changed")
    output = exp.OUT / model / "warm_starts"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "verification.json").exists():
        raise ValueError("Warm-start output already completed; use its verified records")
    gpu = exp.load_accelerator()
    prepare_inputs(model, p, manifest, splits, config)
    trace_rows, run_rows = [], []
    for level in LEVELS:
        arrays, meta, source_path = source_clouds(model, level, manifest["signature"])
        reference = next(c for c in splits["eval"] if c["level"] == level)
        jobs = []
        for seed in PHYSICAL_SEEDS:
            entries = [("cold", None, None)] + [(method, training_seed, arrays[f"seed{training_seed}_{method}"])
                       for training_seed in exp.MODEL_SEEDS for method in METHODS]
            for method, model_seed, cloud in entries:
                jobs.append({"dimension": p.dimension, "level": level, "kappa": p.parameter(level),
                             "seed": seed, "particles": 20000, "method": method,
                             "model_seed": model_seed, "cloud": cloud})
        simulation = make_simulation(gpu, p.dimension, jobs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for generation in range(1, 129):
            simulation.graph.replay()
            if generation & (generation - 1) == 0:
                torch.cuda.synchronize()
                values = (simulation.kappa + simulation.growth / generation).cpu().numpy() / simulation.lam
                for job, value in zip(jobs, values):
                    trace_rows.append({"model": model, "level": level, "kappa": p.parameter(level),
                                       "method": job["method"], "model_seed": job["model_seed"],
                                       "fk_seed": job["seed"], "generation": generation,
                                       "c_hat": float(value), "reference_c": reference["meta"]["c_hat"],
                                       "absolute_error": abs(float(value) - reference["meta"]["c_hat"])})
            if generation == 1 or generation % 32 == 0:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                write_json(output / "progress.json", {"model": model, "level": level, "generation": generation,
                           "generations": 128, "cases": len(jobs), "ensemble_seconds": elapsed})
                print(f"{model} s={level:g}: paired original FK {generation}/128, {len(jobs)} starts", flush=True)
        torch.cuda.synchronize()
        final = simulation.q.cpu().numpy().astype(np.float32)
        if not np.isfinite(final).all():
            raise FloatingPointError("Nonfinite FK warm-start particles")
        identities = []
        for job, initial in zip(jobs, simulation.initial):
            record = {k: v for k, v in job.items() if k != "cloud"}
            record["initial_sha256"] = hashlib.sha256(initial.tobytes()).hexdigest()
            record["reference_c"] = reference["meta"]["c_hat"]
            identities.append(record)
            selected = [row for row in trace_rows if row["level"] == level and row["method"] == job["method"]
                        and row["model_seed"] == job["model_seed"] and row["fk_seed"] == job["seed"]]
            first = next((row["generation"] for row in selected if row["absolute_error"] <= .01), None)
            run_rows.append({"model": model, "level": level, "method": job["method"], "model_seed": job["model_seed"],
                             "fk_seed": job["seed"], "first_recorded_error_below_0p01": first,
                             "error_T1": selected[0]["absolute_error"], "error_T128": selected[-1]["absolute_error"]})
        write_npz(output / f"terminal_s{str(level).replace('.', 'p')}.npz",
                  meta={"protocol": p.metadata(), "dataset_signature": manifest["signature"], "jobs": identities,
                        "generations": 128, "paired_noise": True, "reference_particles_sha256": file_hash(source_path)},
                  target=final)
        csv_write(output / "trace.csv", trace_rows)
        csv_write(output / "runs.csv", run_rows)
        del simulation
        torch.cuda.empty_cache()
    write_json(output / "verification.json", {"status": "complete", "protocol": p.metadata(),
               "dataset_signature": manifest["signature"], "runs": len(run_rows), "trace_rows": len(trace_rows),
               "model_seeds": exp.MODEL_SEEDS, "physical_seeds": PHYSICAL_SEEDS, "levels": LEVELS,
               "particles": 20000, "generations": 128,
               "equations_source_sha256": file_hash(exp.ORIGINAL / "original_gpu_accelerator.py"),
               "wrapper_sha256": file_hash(__file__),
               "meaning": "Actual paired FK evolutions from generated initial particles; fixed lambda=2 and uploaded kappa-offset convention",
               "limitations": ["First recorded tolerance hit does not prove sustained convergence.",
                               "GPU ensemble elapsed time is a run diagnostic, not a measured speedup.",
                               "Cold has three physical repeats; each learned method has three model seeds by three physical repeats."]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("kpp2d", "kpp3d"), required=True)
    run(parser.parse_args().model)
