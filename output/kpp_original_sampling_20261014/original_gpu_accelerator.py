"""Isolated float64 GPU port of the EXECUTED uploaded KPP generators.

The original NumPy source files remain unchanged. Initial uniforms are identical
to the originals; evolution uses per-seed CUDA Philox, not NumPy PCG64 samples.
The 3D negative potential sign is deliberate fidelity to the executed source.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper_reproduction.portability import verified_source_identity
ARCHIVE = ROOT / "deepparticle/data/particle_sources/kpp_archive_20261012/source_snapshot"
TWO_PI = 2.0 * np.pi
LEVELS = (3.75, 4.5)
SEEDS = (666, 999)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def archive_module(dimension):
    name = "kpp_common" if dimension == 2 else "kpp3d_kolmogorov"
    spec = importlib.util.spec_from_file_location("original_sampling_" + name, ARCHIVE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def jobs_for(dimension, particles):
    return [{"dimension": dimension, "level": level, "kappa": 2.0 ** (-level), "seed": seed,
             "particles": particles} for seed in SEEDS for level in LEVELS]


def protocol_record(dimension, particles, generations):
    source = ARCHIVE / ("kpp_common.py" if dimension == 2 else "kpp3d_kolmogorov.py")
    return {"dimension": dimension, "particles": particles, "generations": generations,
            "dt": 2.0 ** -8, "steps_per_period": 256, "lambda": 2.0, "theta": 1.0 if dimension == 2 else None,
            "resample_every": 256, "period": 1.0, "phase": 0,
            "drift": "2*kappa*lambda*e1 + velocity",
            "potential": "kappa*lambda^2 + lambda*v1" if dimension == 2 else "kappa*lambda^2 - lambda*v1",
            "reaction_in_weights": 0.0, "eigenvalue_offset": "+kappa after growth/time",
            "weight_position": "post-propagation", "weight_time": "j*dt (forward), same substep as propagation",
            "resampler": "multinomial inverse CDF once per period",
            **verified_source_identity(source, root=ROOT),
            "accelerator_sha256": sha256(__file__), "dtype": "float64",
            "initial_rng": "numpy.default_rng(seed), uniform(0,2*pi,(N,d))",
            "evolution_rng": "per-seed CUDA Philox, manual_seed(seed+7919), shared across kappa values",
            "rng_caveat": "Evolution random samples differ from original NumPy PCG64; no bitwise-original claim.",
            "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "numpy": np.__version__}


class OriginalGPU:
    def __init__(self, dimension, particles, *, graph=False):
        self.dimension, self.n = dimension, particles
        self.jobs = jobs_for(dimension, particles)
        self.dt, self.rn, self.lam = 2.0 ** -8, 256, 2.0
        self.initial = [np.random.default_rng(j["seed"]).uniform(0.0, TWO_PI, (particles, dimension)) for j in self.jobs]
        self.q = torch.tensor(np.stack(self.initial), dtype=torch.float64, device="cuda")
        self.seed_index = torch.tensor([SEEDS.index(j["seed"]) for j in self.jobs], device="cuda")
        self.generators = [torch.Generator(device="cuda").manual_seed(seed + 7919) for seed in SEEDS]
        self.noise = torch.empty((len(SEEDS), particles, dimension), dtype=torch.float64, device="cuda")
        self.uniforms = torch.empty((len(SEEDS), particles), dtype=torch.float64, device="cuda")
        kappa = torch.tensor([j["kappa"] for j in self.jobs], dtype=torch.float64, device="cuda")
        self.kappa = kappa
        self.sd = (2.0 * kappa * self.dt).sqrt()[:, None, None]
        self.tilt = 2.0 * kappa[:, None] * self.lam
        self.v_const = kappa[:, None] * self.lam * self.lam
        self.logw = torch.zeros((len(self.jobs), particles), dtype=torch.float64, device="cuda")
        self.growth = torch.zeros(len(self.jobs), dtype=torch.float64, device="cuda")
        self.ess_sum = torch.zeros_like(self.growth)
        self.pre_selection = self.q.clone()
        self.probability = torch.full_like(self.logw, 1.0 / particles)
        self.selection_indices = torch.arange(particles, device="cuda").expand(len(self.jobs), -1).clone()
        self.ancestry = self.selection_indices.clone()
        self.generation = 0
        self.selection_count = 0
        self.traces = [[] for _ in self.jobs]
        self.elapsed = 0.0
        self.graph = None
        if graph:
            self.capture()

    @torch.no_grad()
    def step(self, j, *, randomize=True):
        if randomize:
            for i, generator in enumerate(self.generators):
                self.noise[i].normal_(generator=generator)
        q = self.q
        if self.dimension == 2:
            tc = math.cos(TWO_PI * j * self.dt)
            c1, c2 = q[:, :, 0].cos(), q[:, :, 1].cos()
            s1, s2 = q[:, :, 0].sin(), q[:, :, 1].sin()
            q[:, :, 0].add_((self.tilt - c2 - s1 * tc) * self.dt)
            q[:, :, 1].add_((c1 + s2 * tc) * self.dt)
        else:
            ph = math.sin(TWO_PI * j * self.dt)
            sine = (q + ph).sin()
            q[:, :, 0].add_((self.tilt + sine[:, :, 2]) * self.dt)
            q[:, :, 1].add_(sine[:, :, 0] * self.dt)
            q[:, :, 2].add_(sine[:, :, 1] * self.dt)
        q.add_(self.sd * self.noise[self.seed_index])
        q.remainder_(TWO_PI)
        if self.dimension == 2:
            self.logw.add_((self.v_const - self.lam * (q[:, :, 1].cos() + tc * q[:, :, 0].sin())) * self.dt)
        else:
            self.logw.add_((self.v_const - self.lam * (q[:, :, 2] + ph).sin()) * self.dt)

    @torch.no_grad()
    def select(self, *, randomize=True):
        if randomize:
            for i, generator in enumerate(self.generators):
                self.uniforms[i].uniform_(generator=generator)
        maximum = self.logw.amax(1)
        weights = (self.logw - maximum[:, None]).exp()
        total = weights.sum(1)
        self.growth.add_(maximum + (total / self.n).log())
        probability = weights / total[:, None]
        self.ess_sum.add_(1.0 / probability.square().sum(1))
        cdf = probability.cumsum(1)
        cdf[:, -1] = 1.0
        indices = torch.searchsorted(cdf, self.uniforms[self.seed_index].contiguous(), right=True)
        indices.clamp_(0, self.n - 1)
        self.pre_selection.copy_(self.q)
        self.probability.copy_(probability)
        self.selection_indices.copy_(indices)
        self.ancestry.copy_(self.ancestry.gather(1, indices))
        self.q.copy_(self.q.gather(1, indices[:, :, None].expand(-1, -1, self.dimension)))
        self.logw.zero_()

    @torch.no_grad()
    def period(self):
        for j in range(self.rn):
            self.step(j)
        self.select()

    def _state(self):
        return {"q": self.q.clone(), "growth": self.growth.clone(), "ess_sum": self.ess_sum.clone(),
                "pre_selection": self.pre_selection.clone(), "probability": self.probability.clone(),
                "selection_indices": self.selection_indices.clone(), "ancestry": self.ancestry.clone(),
                "rng": [g.get_state() for g in self.generators]}

    def _restore(self, state):
        self.q.copy_(state["q"])
        self.growth.copy_(state["growth"])
        self.ess_sum.copy_(state["ess_sum"])
        self.pre_selection.copy_(state["pre_selection"])
        self.probability.copy_(state["probability"])
        self.selection_indices.copy_(state["selection_indices"])
        self.ancestry.copy_(state["ancestry"])
        self.logw.zero_()
        for generator, rng in zip(self.generators, state["rng"]):
            generator.set_state(rng)

    def capture(self):
        saved = self._state()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.period()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self._restore(saved)
        self.graph = torch.cuda.CUDAGraph()
        for generator in self.generators:
            self.graph.register_generator_state(generator)
        with torch.cuda.graph(self.graph):
            self.period()
        self._restore(saved)
        torch.cuda.synchronize()


def validate_dimension(dimension, particles=256, periods=3):
    """Feed identical original-PCG64 innovations to CPU equations and GPU kernels."""
    gpu = OriginalGPU(dimension, particles)
    q = np.stack(gpu.initial).copy()
    initial_exact = all(np.array_equal(a, b) for a, b in zip(gpu.initial, q))
    rngs = [np.random.default_rng(seed) for seed in SEEDS]
    for rng in rngs:
        rng.uniform(0.0, TWO_PI, (particles, dimension))
    kappa = np.array([j["kappa"] for j in gpu.jobs])
    sd = np.sqrt(2.0 * kappa * gpu.dt)[:, None, None]
    tilt = 2.0 * kappa[:, None] * gpu.lam
    v_const = kappa[:, None] * gpu.lam * gpu.lam
    cpu_logw = np.zeros((len(gpu.jobs), particles))
    growth = np.zeros(len(gpu.jobs))
    ess = np.zeros_like(growth)
    max_step_error = 0.0
    for generation in range(periods):
        for j in range(gpu.rn):
            noise = np.stack([rng.standard_normal((particles, dimension)) for rng in rngs])
            gpu.noise.copy_(torch.as_tensor(noise, device="cuda"))
            if dimension == 2:
                tc = np.cos(TWO_PI * j * gpu.dt)
                c1, c2 = np.cos(q[:, :, 0]), np.cos(q[:, :, 1])
                s1, s2 = np.sin(q[:, :, 0]), np.sin(q[:, :, 1])
                q[:, :, 0] += (tilt - c2 - s1 * tc) * gpu.dt
                q[:, :, 1] += (c1 + s2 * tc) * gpu.dt
            else:
                ph = np.sin(TWO_PI * j * gpu.dt)
                sine = np.sin(q + ph)
                q[:, :, 0] += (tilt + sine[:, :, 2]) * gpu.dt
                q[:, :, 1] += sine[:, :, 0] * gpu.dt
                q[:, :, 2] += sine[:, :, 1] * gpu.dt
            q += sd * noise[np.array([SEEDS.index(job["seed"]) for job in gpu.jobs])]
            q %= TWO_PI
            if dimension == 2:
                cpu_logw += (v_const - gpu.lam * (np.cos(q[:, :, 1]) + tc * np.sin(q[:, :, 0]))) * gpu.dt
            else:
                cpu_logw += (v_const - gpu.lam * np.sin(q[:, :, 2] + ph)) * gpu.dt
            gpu.step(j, randomize=False)
            if j in (0, 127, 255):
                max_step_error = max(max_step_error, float(np.max(np.abs(q - gpu.q.cpu().numpy()))))
        uniforms = np.stack([rng.random(particles) for rng in rngs])
        gpu.uniforms.copy_(torch.as_tensor(uniforms, device="cuda"))
        for i, job in enumerate(gpu.jobs):
            maximum = cpu_logw[i].max()
            weights = np.exp(cpu_logw[i] - maximum)
            total = weights.sum()
            growth[i] += maximum + math.log(total / particles)
            probability = weights / total
            ess[i] += 1.0 / np.square(probability).sum()
            cdf = np.cumsum(probability)
            cdf[-1] = 1.0
            indices = np.searchsorted(cdf, uniforms[SEEDS.index(job["seed"])], side="right")
            q[i] = q[i, np.clip(indices, 0, particles - 1)]
        cpu_logw.fill(0.0)
        gpu.select(randomize=False)
    gpu_q, gpu_growth, gpu_ess = gpu.q.cpu().numpy(), gpu.growth.cpu().numpy(), gpu.ess_sum.cpu().numpy()
    np.testing.assert_allclose(gpu_q, q, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(gpu_growth, growth, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(gpu_ess, ess, rtol=0.0, atol=2e-10)
    module = archive_module(dimension)
    source_errors = []
    for i, job in enumerate(gpu.jobs):
        config_cls = module.IPMConfig if dimension == 2 else module.FK3DConfig
        cfg = config_cls(M=particles, T=periods, seed=job["seed"])
        function = module.kpp2d_fk if dimension == 2 else module.kpp3d_fk
        result = function(-job["level"], cfg)
        target = result["qT" if dimension == 2 else "xT"]
        np.testing.assert_array_equal(q[i].astype(np.float32), target)
        np.testing.assert_allclose(kappa[i] + growth[i] / periods, result["mu_hat"], rtol=0.0, atol=2e-12)
        source_errors.append(float(np.max(np.abs(q[i].astype(np.float32) - target))))
    del gpu
    torch.cuda.empty_cache()
    eager, captured = OriginalGPU(dimension, 512), OriginalGPU(dimension, 512, graph=True)
    for _ in range(3):
        eager.period()
        captured.graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(eager.q, captured.q, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(eager.growth, captured.growth, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(eager.ess_sum, captured.ess_sum, rtol=0.0, atol=1e-9)
    torch.testing.assert_close(eager.pre_selection, captured.pre_selection, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(eager.probability, captured.probability, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(eager.selection_indices, captured.selection_indices, rtol=0.0, atol=0.0)
    torch.testing.assert_close(eager.ancestry, captured.ancestry, rtol=0.0, atol=0.0)
    return {"dimension": dimension, "status": "passed", "identical_supplied_innovations": True,
            "particles_per_case": particles, "periods": periods, "cases": len(source_errors),
            "initial_numpy_uniforms_exact": initial_exact, "source_terminal_float32_max_errors": source_errors,
            "gpu_vs_cpu_float64_max_step_error": max_step_error,
            "gpu_vs_cpu_growth_max_error": float(np.max(np.abs(gpu_growth - growth))),
            "cuda_graph_vs_eager": "passed, 512 particles, 3 periods, same Philox seeds",
            "resampling_calls_per_case": periods, "steps_per_period": 256,
            "three_dimensional_potential_sign": "negative" if dimension == 3 else None}


def run_dimension(dimension, particles, generations, validation):
    protocol = protocol_record(dimension, particles, generations)
    simulation = OriginalGPU(dimension, particles, graph=True)
    checkpoint_path = OUT / f"original_gpu_checkpoint_{dimension}d.pt"
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Checkpoint belongs to a different original protocol or source.")
        simulation._restore(saved["state"])
        simulation.generation = saved["generation"]
        simulation.selection_count = saved["generation"]
        simulation.traces = saved["traces"]
        simulation.elapsed = saved["seconds"]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for generation in range(simulation.generation + 1, generations + 1):
        simulation.graph.replay()
        simulation.generation = generation
        simulation.selection_count += 1
        if generation & (generation - 1) == 0 or generation == generations:
            torch.cuda.synchronize()
            mu = (simulation.kappa + simulation.growth / generation).cpu().numpy()
            seconds = simulation.elapsed + time.perf_counter() - start
            ess = (simulation.ess_sum / generation).cpu().numpy()
            for trace, value, effective in zip(simulation.traces, mu, ess):
                trace.append({"generation": generation, "mu": float(value), "c": float(value / simulation.lam),
                              "ess_mean": float(effective), "seconds": seconds})
        if generation == 1 or generation % 64 == 0 or generation == generations:
            torch.cuda.synchronize()
            seconds = simulation.elapsed + time.perf_counter() - start
            write_json(OUT / f"gpu_progress_{dimension}d.json",
                       {"status": "running", "dimension": dimension, "generation": generation,
                        "total_generations": generations, "cases": len(simulation.jobs),
                        "particles_per_case": particles, "seconds": seconds,
                        "seconds_per_period": seconds / generation,
                        "estimated_remaining_seconds": seconds / generation * (generations - generation)})
            print(json.dumps({"dimension": dimension, "generation": generation, "seconds": round(seconds, 2),
                              "remaining_seconds": round(seconds / generation * (generations - generation), 2)}), flush=True)
        if generation % 256 == 0 or generation == generations:
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save({"identity": identity, "generation": generation, "state": simulation._state(),
                        "traces": simulation.traces, "seconds": simulation.elapsed + time.perf_counter() - start}, temporary)
            temporary.replace(checkpoint_path)
    torch.cuda.synchronize()
    seconds = simulation.elapsed + time.perf_counter() - start
    final = simulation.q.cpu().numpy().astype(np.float32)
    pre_selection = simulation.pre_selection.cpu().numpy().astype(np.float32)
    probability = simulation.probability.cpu().numpy()
    selection_indices = simulation.selection_indices.cpu().numpy()
    ancestry = simulation.ancestry.cpu().numpy()
    ess = (simulation.ess_sum / generations).cpu().numpy()
    outputs = []
    for index, (job, initial, target, trace, effective) in enumerate(zip(simulation.jobs, simulation.initial, final, simulation.traces, ess)):
        if not np.isfinite(target).all() or not math.isfinite(trace[-1]["mu"]):
            raise FloatingPointError("Nonfinite full-run output.")
        suffix = str(job["level"]).replace(".", "p")
        path = OUT / f"original_gpu_kpp{dimension}d_s{suffix}_seed{job['seed']}.npz"
        meta = {"model": f"kpp{dimension}d", "level": job["level"], "seed": job["seed"],
                "generations": generations, "particles": particles, "source_hash": protocol["source_sha256"],
                "protocol": protocol, "job": job, "trace": trace, "mu_hat": trace[-1]["mu"],
                "c_hat": trace[-1]["c"], "ess_mean": float(effective),
                "n_unique": int(len(np.unique(target, axis=0))), "seconds": seconds,
                "n_unique_ancestors": int(len(np.unique(ancestry[index]))),
                "selection_count": simulation.selection_count, "validation": validation,
                "backend": "float64_cuda_graph_per_seed_philox_original_discretization"}
        keys = {"q0": initial.astype(np.float32), "qT": target} if dimension == 2 else {"x0": initial.astype(np.float32), "xT": target}
        np.testing.assert_array_equal(target, pre_selection[index][selection_indices[index]])
        np.savez_compressed(path, initial=initial.astype(np.float32), target=target,
                            pre_selection=pre_selection[index], probability=probability[index],
                            selection_indices=selection_indices[index], ancestry=ancestry[index],
                            meta=json.dumps(meta, ensure_ascii=False, allow_nan=False), **keys)
        write_json(path.with_suffix(".json"), meta)
        outputs.append({"file": str(path), "sha256": sha256(path), "dimension": dimension,
                        "level": job["level"], "kappa": job["kappa"], "seed": job["seed"],
                        "c_hat": meta["c_hat"], "ess_mean": float(effective), "n_unique": meta["n_unique"]})
    write_json(OUT / f"gpu_progress_{dimension}d.json", {"status": "complete", "dimension": dimension,
               "generation": generations, "total_generations": generations, "seconds": seconds, "outputs": outputs})
    print(json.dumps({"dimension": dimension, "status": "complete", "seconds": round(seconds, 2), "outputs": outputs}), flush=True)
    del simulation
    torch.cuda.empty_cache()
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--particles", type=int, default=20000)
    parser.add_argument("--generations", type=int, default=4096)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    validation = [validate_dimension(d) for d in (2, 3)]
    write_json(OUT / "gpu_validation.json", {"status": "passed", "results": validation,
               "source_hashes": {p.name: verified_source_identity(p, root=ROOT)["source_sha256"]
                                 for p in (ARCHIVE / "kpp_common.py", ARCHIVE / "kpp3d_kolmogorov.py")}})
    print(json.dumps({"validation": validation}), flush=True)
    if args.validate_only:
        return
    outputs = []
    for dimension, result in zip((2, 3), validation):
        outputs.extend(run_dimension(dimension, args.particles, args.generations, result))
        write_json(OUT / "gpu_manifest.json", {"status": "complete" if len(outputs) == 8 else "running",
                   "particles_per_case": args.particles, "generations": args.generations,
                   "levels": LEVELS, "seeds": SEEDS, "validation": validation, "outputs": outputs})


if __name__ == "__main__":
    main()
