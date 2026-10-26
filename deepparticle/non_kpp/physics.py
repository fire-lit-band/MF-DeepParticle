"""PyTorch port of the supplied SGIP; physical conventions are kept explicit.

The supplied CuPy FDM programs remain unchanged in sources/. This module does
not claim FDM validation. It reproduces the particle algorithm with seeded RNG.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import time

import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


@dataclass(frozen=True)
class Physics:
    reaction: str
    particles: int = 5_000_000
    bins: int = 100
    half_width: float = 60.
    initial_radius: float = 5.
    diffusion: float = 1.
    final_time: float = 20.
    dt: float = .5
    seed: int = 666

    def __post_init__(self):
        if self.reaction not in ('cubic', 'arrhenius'):
            raise ValueError('Unknown reaction')
        if min(self.particles, self.bins, self.half_width, self.initial_radius,
               self.diffusion, self.final_time, self.dt) <= 0:
            raise ValueError('Positive simulation parameters required')
        if not math.isclose(round(self.final_time / self.dt) * self.dt, self.final_time):
            raise ValueError('dt must divide final_time')

    @property
    def dx(self):
        return 2 * self.half_width / self.bins


def ball(count, radius, seed):
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((count, 3), dtype=np.float32)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True),
                             np.finfo(np.float32).tiny)
    return directions * (radius * rng.random(count, dtype=np.float32) ** (1 / 3))[:, None]


def velocity(x):
    result = torch.empty_like(x)
    a, b, c = 1., math.sqrt(2 / 3), math.sqrt(1 / 3)
    result[:, 0] = a * x[:, 2].sin() + c * x[:, 1].cos()
    result[:, 1] = b * x[:, 0].sin() + a * x[:, 2].cos()
    result[:, 2] = c * x[:, 1].sin() + b * x[:, 0].cos()
    return result


def reaction_value(x, kind):
    if kind == 'cubic':
        return x.square() * (1 - x)
    safe = x.clamp_min(1e-12)
    return torch.where(x > 0, torch.exp(-.5 / safe) * (1 - x), 0.)


def reaction_derivative(x, kind):
    if kind == 'cubic':
        return 2 * x - 3 * x.square()
    safe = x.clamp_min(1e-12)
    return torch.where(x > 0, torch.exp(-.5 / safe) * (.5 * (1 - x) / safe.square() - 1), 0.)


def apply_reaction(density, config):
    result = density.clone()
    active = result > 0
    old = result[active]
    updated = (old + config.dt * reaction_value(old, config.reaction)).clamp(0, 1)
    for _ in range(10):
        residual = updated - old - config.dt * reaction_value(updated, config.reaction)
        jacobian = 1 - config.dt * reaction_derivative(updated, config.reaction)
        denominator = torch.where(jacobian.abs() > 1e-12, jacobian, 1.)
        updated = (updated - residual / denominator).clamp(0, 1)
    result[active] = updated
    return result


def reconstruct(particles, total_mass, config):
    indices = ((particles + config.half_width) / config.dx).floor().long()
    valid = ((indices >= 0) & (indices < config.bins)).all(1)
    idx = indices[valid]
    ids = (idx[:, 0] * config.bins + idx[:, 1]) * config.bins + idx[:, 2]
    counts = torch.bincount(ids, minlength=config.bins ** 3)
    density = counts.double() * total_mass / config.particles / config.dx ** 3
    return density, ids, valid


def resample(particles, density, ids, valid, config, generator):
    device = particles.device
    active = torch.nonzero(density > 0, as_tuple=True)[0]
    cdf = torch.cumsum(density[active] / density.sum(), 0)
    cdf[-1] = 1.
    uniforms = torch.rand(config.particles, dtype=torch.float64, device=device, generator=generator)
    sampled = active[torch.searchsorted(cdf, uniforms).clamp_max(len(active) - 1)]
    sampled = sampled.sort().values
    current_counts = torch.bincount(ids, minlength=config.bins ** 3)
    target_counts = torch.bincount(sampled, minlength=config.bins ** 3)
    # The reaction vanishes at zero: every target bin is occupied, so the
    # original code's uniform-in-empty-bin branch is unreachable here.
    counts = current_counts[sampled]
    if bool((counts == 0).any()):
        raise RuntimeError('Reaction unexpectedly populated an empty bin')
    valid_indices = torch.nonzero(valid, as_tuple=True)[0]
    random_order = torch.randperm(len(ids), device=device, generator=generator)
    grouped = valid_indices[random_order[torch.argsort(ids[random_order], stable=True)]]
    current_starts = current_counts.cumsum(0) - current_counts
    target_starts = target_counts.cumsum(0) - target_counts
    ranks = torch.arange(config.particles, device=device) - target_starts[sampled]
    replacement = (torch.rand(config.particles, device=device, generator=generator,
                              dtype=torch.float64) * counts).long()
    offsets = torch.where(target_counts[sampled] <= counts, ranks, replacement)
    return particles[grouped[current_starts[sampled] + offsets]]


@torch.no_grad()
def simulate(config, out, device='cuda'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    signature = asdict(config)
    if (out / 'field.npz').exists():
        metadata = json.loads((out / 'simulation.json').read_text())
        if metadata['physics'] != signature or metadata['status'] != 'complete':
            raise ValueError('Existing simulation settings differ')
        return out / 'field.npz'
    generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    initial_mass = 4 / 3 * math.pi * config.initial_radius ** 3
    total_mass, history, records, first_step = initial_mass, [initial_mass], [], 1
    saved = out / 'resume.pt'
    if saved.exists():
        state = torch.load(saved, map_location='cpu', weights_only=True)
        if state['physics'] != signature:
            raise ValueError('Resume physics differs')
        particles = state['particles'].to(device)
        generator.set_state(state['rng'])
        total_mass, history, records = state['mass'], state['history'], state['records']
        first_step = state['step'] + 1
    else:
        particles = torch.from_numpy(ball(config.particles, config.initial_radius, config.seed)).to(device)
    started = time.perf_counter()
    steps = round(config.final_time / config.dt)
    for step in range(first_step, steps + 1):
        particles -= config.dt * velocity(particles)
        particles += math.sqrt(2 * config.diffusion * config.dt) * torch.randn(
            particles.shape, device=device, generator=generator)
        density, ids, valid = reconstruct(particles, total_mass, config)
        discarded = config.particles - int(valid.sum())
        over_one = int((density > 1).sum())
        density = apply_reaction(density, config)
        total_mass = float(density.sum()) * config.dx ** 3
        if not math.isfinite(total_mass) or total_mass <= 0:
            raise FloatingPointError('Invalid total mass')
        history.append(total_mass)
        records.append({'step': step, 'discarded': discarded, 'bins_over_one_before_reaction': over_one,
                        'mass': total_mass, 'maximum_density': float(density.max())})
        # Final resampling cannot change the returned bin density; omit this
        # otherwise unused draw. Earlier steps retain the source resampler.
        if step < steps:
            particles = resample(particles, density, ids, valid, config, generator)
        elapsed = time.perf_counter() - started
        print(f'{config.reaction} SGIP {step}/{steps} mass={total_mass:.7g} elapsed={elapsed:.1f}s', flush=True)
        write_json(out / 'progress.json', {'step': step, 'steps': steps, 'mass': total_mass,
                                          'elapsed_seconds_this_invocation': elapsed})
        if step % 5 == 0 and step < steps:
            temporary = out / 'resume.pt.tmp'
            torch.save({'physics': signature, 'particles': particles.cpu(), 'rng': generator.get_state(),
                        'mass': total_mass, 'history': history, 'records': records, 'step': step}, temporary)
            temporary.replace(saved)
    centers = np.arange(config.bins) * config.dx - config.half_width + config.dx / 2
    np.savez_compressed(out / 'field.npz', centers=centers,
                        density=density.reshape((config.bins,) * 3).cpu().numpy().astype(np.float32),
                        mass_history=history)
    write_json(out / 'simulation.json', {'physics': signature, 'status': 'complete',
               'seconds_this_invocation': time.perf_counter() - started,
               'total_mass': total_mass, 'records': records,
               'pde': 'u_t = D Laplacian(u) + w dot grad(u) + r(u)',
               'drift': '-w', 'boundary': 'discard outside [-60,60]^3 as in supplied SGIP',
               'backend': 'PyTorch; seeded NumPy initial ball and torch CUDA noise/resampling',
               'changes_from_source': ['seeded RNG', 'skip unused final resampling',
                 'uniform random permutation replaces random-key sort for within-bin shuffle'],
               'fdm_run': False})
    return out / 'field.npz'


def sample_field(density, config, count, seed):
    rng = np.random.default_rng(seed)
    flat = np.asarray(density, dtype=np.float64).ravel()
    if not np.isfinite(flat).all() or flat.min() < 0 or flat.sum() <= 0:
        raise ValueError('Invalid density')
    cdf = np.cumsum(flat / flat.sum())
    cdf[-1] = 1.
    bins = np.searchsorted(cdf, rng.random(count), side='right')
    indices = np.column_stack(np.unravel_index(bins, (config.bins,) * 3))
    return (-config.half_width + (indices + rng.random((count, 3))) * config.dx).astype(np.float32)
