"""The audited original-three-layer experiment protocol (no notebook imports)."""
from __future__ import annotations

import math
import numpy as np
import torch

from . import core as c

def sigma_grid():
    """The original float32 sigma labels, preserving their recorded rounding."""
    return c.TRAIN_SIGMA.copy()


def physics(mu=1., chi=16 * math.pi, delta=.1):
    return {
        "physics_id": c.PHYSICS_ID,
        "kernel": "+r/(4*pi*(r^2+delta^2))",
        "gradient_variable": "current particle X_j; displacement X_j-X_i",
        "interaction": "-chi/J * sum_i!=j grad_Xj K_delta",
        "integrator": "split advection -> interaction -> diffusion; float64",
        "noise_sampling": "NumPy normal; sqrt(2*mu*dt) per coordinate",
        "mu": float(mu), "chi": float(chi), "delta": float(delta),
        "final_time": .02, "dt": .005,
        "numerical_parameters_confirmed_from_manuscript": False,
    }


def sample_times_original(batch, mean=-.4, std=1.):
    samples = np.random.randn(batch, 2).astype(np.float32) * std + mean
    samples = 1 / (1 + np.exp(-samples))
    t = np.maximum(samples[:, 0], samples[:, 1])
    r = np.minimum(samples[:, 0], samples[:, 1])
    selected = np.random.permutation(batch)[:batch // 2]
    r[selected] = t[selected]
    return torch.from_numpy(r[:, None]), torch.from_numpy(t[:, None])


@torch.no_grad()
def simulate_original(initial, sigma, *, mu=1., chi=16 * math.pi, delta=.1,
                      t=.02, dt=.005, chunk=512, device="cpu"):
    if (initial.ndim != 2 or initial.shape[1] != 3 or len(initial) == 0 or
            not initial.is_floating_point() or not torch.isfinite(initial).all()):
        raise ValueError("initial must be a nonempty finite floating (N, 3) tensor")
    if (not all(math.isfinite(float(v)) for v in (sigma, mu, chi, delta, t, dt)) or
            sigma <= 0 or min(mu, chi, t) < 0 or min(delta, dt, chunk) <= 0):
        raise ValueError("Invalid physical parameters")
    steps = round(t / dt)
    if not math.isclose(steps * dt, t, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("t must be an integer multiple of dt")
    points = initial.detach().to(device=device, dtype=torch.float64).clone()
    for _ in range(steps):
        points[:, 0] += dt * float(sigma) * torch.exp(-points[:, 1].square() - points[:, 2].square())
        force = torch.empty_like(points)
        for start in range(0, len(points), chunk):
            displacement = points[start:start + chunk, None, :] - points[None, :, :]
            squared = displacement.square().sum(-1)
            radius = torch.where(squared > 0, squared.sqrt(), torch.ones_like(squared))
            factor = (delta**2 - squared) / (4 * math.pi * radius * (squared + delta**2).square())
            factor = torch.where(squared > 0, factor, torch.zeros_like(factor))
            force[start:start + chunk] = -(chi / len(points)) * (displacement * factor[..., None]).sum(1)
        noise = torch.as_tensor(np.random.normal(size=tuple(points.shape)), dtype=points.dtype, device=device)
        points = points + dt * force + math.sqrt(2 * mu * dt) * noise
        if not torch.isfinite(points).all():
            raise FloatingPointError("Non-finite KS particles")
    return points.cpu().to(initial.dtype)


def simulate(initial, sigma, args):
    return simulate_original(initial, sigma, mu=args.mu, chi=args.chi, delta=args.delta,
                    chunk=args.chunk, device=args.device)


def rng_state():
    state = np.random.get_state()
    return {"torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng": {"name": state[0], "keys": state[1].tolist(), "pos": state[2],
                          "has_gauss": state[3], "cached_gaussian": state[4]}}


def initialize_mf(seed, args, factory=c.Meanflow):
    c.set_seed(seed)
    if getattr(args, "rng_mode", "fresh") == "replay_20261001":
        # Reproduce the explicitly recorded initialization offset of the old run.
        # This mode is for its fixed cached data; it is not a new architecture.
        c.unit_ball(15000)
        for _ in range(32):
            np.random.normal(size=(15000, 3))
        torch.rand(18819)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            return factory().to(args.device)
    return factory().to(args.device)
