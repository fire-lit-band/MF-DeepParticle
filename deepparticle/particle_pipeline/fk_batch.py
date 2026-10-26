"""Float64 FK ensembles with the same paired NumPy RNG streams as scalar FK."""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from .physics import TWO_PI, check_cloud


def kpp_fk_batch(jobs, *, device="cuda", progress=None):
    if not jobs:
        return []
    p, n, generations = jobs[0]["protocol"], jobs[0]["particles"], jobs[0]["generations"]
    if p.family != "kpp" or any(j["protocol"] != p or j["particles"] != n or j["generations"] != generations for j in jobs):
        raise ValueError("FK ensemble requires one physical protocol, particle count and horizon")
    seeds = sorted({j["fk_seed"] for j in jobs})
    rngs = [np.random.default_rng(seed + 7919) for seed in seeds]
    seed_index = torch.tensor([seeds.index(j["fk_seed"]) for j in jobs], device=device)
    initial = [np.random.default_rng(j["fk_seed"]).uniform(0, TWO_PI, (n, p.dimension)) if j["cloud"] is None
               else check_cloud(j["cloud"], p.dimension, periodic=True) for j in jobs]
    if any(len(q) != n for q in initial):
        raise ValueError("Warm-start count differs from ensemble count")
    points = torch.tensor(np.stack(initial), dtype=torch.float64, device=device)
    kappa = torch.tensor([p.parameter(j["level"]) for j in jobs], dtype=torch.float64, device=device)
    sd = (2 * kappa * p.dt).sqrt()[:, None, None]
    tilt = 2 * kappa[:, None] * p.lam
    constant = kappa[:, None] * p.lam ** 2 + p.reaction_rate
    growth = torch.zeros(len(jobs), dtype=torch.float64, device=device)
    logw = torch.zeros((len(jobs), n), dtype=torch.float64, device=device)
    traces = [[] for _ in jobs]
    rn = round(1 / p.dt)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for generation in range(1, generations + 1):
        for step in range(rn):
            t = step * p.dt
            noise = torch.as_tensor(np.stack([rng.standard_normal((n, p.dimension)) for rng in rngs]), device=device)
            if p.dimension == 2:
                tc = p.theta * math.cos(TWO_PI * t)
                velocity = torch.stack((-points[:, :, 1].cos() - tc * points[:, :, 0].sin(),
                                        points[:, :, 0].cos() + tc * points[:, :, 1].sin()), dim=-1)
            else:
                phase = math.sin(TWO_PI * t)
                velocity = (points[:, :, [2, 0, 1]] + phase).sin()
            velocity *= p.flow_amplitude
            velocity[:, :, 0] += tilt
            points = points + velocity * p.dt + sd * noise[seed_index]
            points.remainder_(TWO_PI)
            v1 = (-points[:, :, 1].cos() - tc * points[:, :, 0].sin()) if p.dimension == 2 else (points[:, :, 2] + phase).sin()
            logw += (constant + p.lam * p.flow_amplitude * v1) * p.dt
            if (step + 1) % p.resample_every == 0:
                maximum = logw.amax(dim=1)
                weights = (logw - maximum[:, None]).exp()
                total = weights.sum(dim=1)
                growth += maximum + (total / n).log()
                cdf = (weights / total[:, None]).cumsum(dim=1)
                cdf[:, -1] = 1.
                uniforms = torch.as_tensor(np.stack([rng.random(n) for rng in rngs]), device=device)
                indices = torch.searchsorted(cdf, uniforms[seed_index].contiguous(), right=True)
                points = points.gather(1, indices[:, :, None].expand(-1, -1, p.dimension))
                logw.zero_()
        if generation & (generation - 1) == 0 or generation == generations:
            mus = (growth / (generation * rn * p.dt)).cpu().numpy()
            seconds = time.perf_counter() - start
            for trace, mu in zip(traces, mus):
                trace.append({"generation": generation, "mu": float(mu), "c": float(mu / p.lam), "seconds": seconds})
        if progress is not None and (generation == 1 or generation % 16 == 0 or generation == generations):
            progress(generation, generations)
    if not torch.isfinite(points).all() or not torch.isfinite(growth).all():
        raise FloatingPointError("Nonfinite FK ensemble")
    final = points.cpu().numpy().astype(np.float32)
    seconds = time.perf_counter() - start
    return [(job, {"initial": q0.astype(np.float32), "target": target, "trace": trace,
                   "mu_hat": trace[-1]["mu"], "c_hat": trace[-1]["c"], "seconds": seconds,
                   "backend": f"torch_float64_{device}_shared_numpy_rng_ensemble"})
            for job, q0, target, trace in zip(jobs, initial, final, traces)]
