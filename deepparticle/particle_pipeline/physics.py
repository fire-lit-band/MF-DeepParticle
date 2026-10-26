"""One source of truth for signed KPP operators and dimension-aware KS particles."""
from __future__ import annotations

import math
import time
import numpy as np

TWO_PI = 2 * math.pi


def check_cloud(points, dimension, periodic=False):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != dimension or not len(points) or not np.isfinite(points).all():
        raise ValueError(f"Expected a finite nonempty (N,{dimension}) particle cloud")
    return np.remainder(points, TWO_PI) if periodic else points.copy()


def unit_ball(n, dimension, rng):
    directions = rng.standard_normal((n, dimension))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise FloatingPointError("Zero random direction")
    return directions / norms * rng.random((n, 1)) ** (1 / dimension)


def kpp_velocity(points, t, protocol):
    if protocol.dimension == 2:
        x, y = points.T
        tc = protocol.theta * math.cos(TWO_PI * t)
        velocity = np.column_stack((-np.cos(y) - tc * np.sin(x), np.cos(x) + tc * np.sin(y)))
    else:
        phase = math.sin(TWO_PI * t)
        velocity = np.sin(points[:, [2, 0, 1]] + phase)
    return protocol.flow_amplitude * velocity


def kpp_drift_potential(points, t, level, protocol):
    kappa = protocol.parameter(level)
    velocity = kpp_velocity(points, t, protocol)
    drift = velocity.copy()
    drift[:, 0] += 2 * kappa * protocol.lam
    potential = kappa * protocol.lam ** 2 + protocol.lam * velocity[:, 0] + protocol.reaction_rate
    return drift, potential


def multinomial_indices(probability, rng):
    cdf = np.cumsum(probability, dtype=np.float64)
    cdf[-1] = 1.
    return np.searchsorted(cdf, rng.random(len(probability)), side="right")


def kpp_fk(level, protocol, *, seed=666, particles=None, initial=None,
           generations=None, progress=None):
    """Return terminal particles and genuine FK normalizer traces, including warm starts.

    Initial and evolution RNG streams are separate. Equal evolution seeds give
    paired Brownian/resampling streams across cold and learned initializations.
    The reaction constant is inside the weights and is never added a second time.
    """
    if protocol.family != "kpp":
        raise ValueError("FK evaluation is defined for KPP only")
    n = protocol.n_eval if particles is None else int(particles)
    generations = protocol.generations if generations is None else int(generations)
    if n <= 0 or generations <= 0:
        raise ValueError("Particle count and generations must be positive")
    if initial is None:
        points = np.random.default_rng(seed).uniform(0, TWO_PI, (n, protocol.dimension))
    else:
        points = check_cloud(initial, protocol.dimension, periodic=True)
        if len(points) != n:
            raise ValueError("Warm-start particle count differs from FK count")
    q0 = points.astype(np.float32).copy()
    rng = np.random.default_rng(int(seed) + 7919)
    rn = round(1 / protocol.dt)
    sd = math.sqrt(2 * protocol.parameter(level) * protocol.dt)
    logw = np.zeros(n)
    growth, ess_sum, selections = 0., 0., 0
    trace = []
    start = time.perf_counter()
    for generation in range(1, generations + 1):
        for j in range(rn):
            t = j * protocol.dt
            drift, _ = kpp_drift_potential(points, t, level, protocol)
            points = np.remainder(points + drift * protocol.dt + sd * rng.standard_normal(points.shape), TWO_PI)
            # Propagation, then weighting at the propagated position and the
            # documented substep phase, exactly the chosen splitting convention.
            _, potential = kpp_drift_potential(points, t, level, protocol)
            logw += potential * protocol.dt
            if (j + 1) % protocol.resample_every == 0:
                maximum = float(logw.max())
                weights = np.exp(logw - maximum)
                total = float(weights.sum())
                growth += maximum + math.log(total / n)
                probability = weights / total
                ess_sum += 1 / float(np.square(probability).sum())
                selections += 1
                points = points[multinomial_indices(probability, rng)]
                logw.fill(0.)
        if generation & (generation - 1) == 0 or generation == generations:
            mu = growth / (generation * rn * protocol.dt)
            trace.append({"generation": generation, "mu": mu, "c": mu / protocol.lam,
                          "seconds": time.perf_counter() - start})
        if progress is not None and (generation == 1 or generation % 64 == 0 or generation == generations):
            progress(generation, generations)
    if not np.isfinite(points).all() or not math.isfinite(growth):
        raise FloatingPointError("Nonfinite KPP simulation")
    return {"initial": q0, "target": points.astype(np.float32), "trace": trace,
            "mu_hat": trace[-1]["mu"], "c_hat": trace[-1]["c"],
            "ess_mean": ess_sum / max(selections, 1), "seconds": time.perf_counter() - start,
            "seed": seed, "evolution_seed": seed + 7919, "phase": 0}


def ks_simulate(initial, level, protocol, *, seed=666, chunk=256, device="cpu"):
    """Shared shear/interaction/diffusion split; 4D retains the manuscript kernel.

    KS2D uses the regularized logarithmic-kernel force from the archived
    KS_data.py, not the three-dimensional fundamental solution.
    """
    import torch
    if protocol.family != "ks":
        raise ValueError("KS simulation requires a KS protocol")
    initial = check_cloud(initial, protocol.dimension)
    if chunk < 1:
        raise ValueError("Interaction chunk must be positive")
    points = torch.tensor(initial, dtype=torch.float64, device=device)
    rng = np.random.default_rng(seed)
    n = len(points)
    start = time.perf_counter()
    for _ in range(round(protocol.final_time / protocol.dt)):
        points[:, 0] += protocol.dt * protocol.parameter(level) * torch.exp(-points[:, 1:].square().sum(-1))
        force = torch.empty_like(points)
        for lo in range(0, n, chunk):
            displacement = points[lo:lo + chunk, None] - points[None]
            squared = displacement.square().sum(-1)
            if protocol.dimension == 2:
                factor = 1 / (2 * math.pi * (squared + protocol.delta ** 2))
            else:
                radius = torch.where(squared > 0, squared.sqrt(), torch.ones_like(squared))
                factor = (protocol.delta ** 2 - squared) / (4 * math.pi * radius * (squared + protocol.delta ** 2).square())
            factor = torch.where(squared > 0, factor, torch.zeros_like(factor))
            force[lo:lo + chunk] = -(protocol.chi / n) * (displacement * factor[..., None]).sum(1)
        noise = torch.as_tensor(rng.standard_normal(tuple(points.shape)), device=device)
        points = points + protocol.dt * force + math.sqrt(2 * protocol.mu * protocol.dt) * noise
        if not torch.isfinite(points).all():
            raise FloatingPointError("Nonfinite KS particles")
    return {"initial": initial.astype(np.float32), "target": points.cpu().numpy().astype(np.float32),
            "trace": [], "seconds": time.perf_counter() - start, "seed": seed}
