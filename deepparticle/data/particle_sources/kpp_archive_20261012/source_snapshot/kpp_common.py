# -*- coding: utf-8 -*-
"""Archived 2D periodic KPP Feynman-Kac particle simulator."""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass
class IPMConfig:
    """Parameters for the periodic particle simulation. Each generation is one forcing period; resample_every=None selects once per generation."""
    lam: float = 2.0
    theta: float = 1.0
    log2dt: float = -8.0
    M: int = 20000
    T: int = 4096
    seed: int = 666
    resample_every: Optional[int] = None

    @property
    def dt(self) -> float:
        return 2.0 ** self.log2dt

    @property
    def rn(self) -> int:
        """Number of integration substeps per unit forcing period."""
        return int(round(2.0 ** (-self.log2dt)))

    @property
    def re(self) -> int:
        """Effective resampling interval in integration substeps."""
        return self.rn if self.resample_every is None else int(self.resample_every)


def training_log2sigma() -> List[float]:
    """Return the eight training levels of log2(diffusivity)."""
    return [-2.0 - 0.25 * k for k in range(8)]


def eval_log2sigma() -> List[float]:
    """Return eleven evaluation levels, including three extrapolation levels."""
    return [-2.0 - 0.25 * k for k in range(11)]


def dyadic_generations(t_max: int) -> List[int]:
    """Return powers of two up to t_max, also including the final generation."""
    out, g = [], 1
    while g <= t_max:
        out.append(g)
        g *= 2
    if out and out[-1] != t_max:
        out.append(t_max)
    return out


def kpp2d_fk(log2sigma: float, cfg: IPMConfig,
             record_generations: Optional[Sequence[int]] = None,
             progress_every: int = 0) -> Dict[str, object]:
    """Simulate Euler-Maruyama particles on the two-dimensional torus.

    Accumulate the potential at updated positions and use multinomial
    inverse-CDF resampling. The archived estimator adds kappa to the
    time-averaged log mean weights. Return initial/final clouds,
    eigenvalue and speed estimates, trace, and sampling diagnostics.
    Initial and final array rows are not paired particle trajectories."""
    t_start = time.perf_counter()
    rng = np.random.default_rng(int(cfg.seed))

    lam, th = float(cfg.lam), float(cfg.theta)
    dt, rn, nn, M = cfg.dt, cfg.rn, int(cfg.T), int(cfg.M)
    resample_every = cfg.re
    if resample_every <= 0:
        raise ValueError('resample_every 必须是正整数')
    if abs(rn * dt - 1.0) > 1e-12:
        warnings.warn(f'1/dt = {1.0/dt} 不是整数，每代实际时长 {rn*dt} != 1，周期会漂移。')

    kappa = 2.0 ** log2sigma
    sd = math.sqrt(2.0 * kappa * dt)
    drift_q1 = 2.0 * kappa * lam
    v_const = kappa * lam * lam


    thetacos = th * np.cos(TWO_PI * np.arange(rn) * dt)

    q = rng.uniform(0.0, TWO_PI, size=(M, 2))
    q0 = q.astype(np.float32).copy()

    if record_generations is None:
        record_generations = dyadic_generations(nn)
    record_set = {int(g) for g in record_generations if 1 <= int(g) <= nn}

    log_w_sum = 0.0
    logw = np.zeros(M)
    ess_sum, ess_cnt = 0.0, 0
    trace: List[Tuple[int, float, float]] = []

    for g in range(nn):
        for j in range(rn):
            tc = thetacos[j]


            # Evaluate both drift components at the old position.
            c1, c2 = np.cos(q[:, 0]), np.cos(q[:, 1])
            s1, s2 = np.sin(q[:, 0]), np.sin(q[:, 1])
            q[:, 0] += (drift_q1 - c2 - s1 * tc) * dt
            q[:, 1] += (c1 + s2 * tc) * dt
            q += sd * rng.standard_normal((M, 2))
            q %= TWO_PI


            logw += (v_const - lam * (np.cos(q[:, 1]) + tc * np.sin(q[:, 0]))) * dt


            if (j + 1) % resample_every == 0:
                # Shift log weights for stable normalization.
                mx = logw.max()
                w = np.exp(logw - mx)
                sw = w.sum()

                log_w_sum += mx + math.log(sw / M)
                p = w / sw
                ess_sum += 1.0 / float(np.sum(p * p)); ess_cnt += 1
                cdf = np.cumsum(p)
                # Avoid roundoff moving inverse-CDF indices out of range.
                cdf[-1] = 1.0
                idx = np.searchsorted(cdf, rng.random(M), side='right')
                np.clip(idx, 0, M - 1, out=idx)
                q = q[idx]
                logw.fill(0.0)

        gen = g + 1
        if gen in record_set:
            mu = kappa + log_w_sum / gen
            trace.append((gen, mu, mu / lam))
        if progress_every and gen % progress_every == 0:
            el = time.perf_counter() - t_start
            print(f'    gen {gen}/{nn}  {el/gen*1000:.0f} ms/gen  '
                  f'elapsed {el/60:.1f} min  ETA {(nn-gen)*el/gen/60:.1f} min', flush=True)

    if rn % resample_every != 0:
        warnings.warn('resample_every 不整除 rn，末段累积的权重被丢弃，估计有偏。')

    log_w_mean = log_w_sum / nn
    mu_hat = kappa + log_w_mean
    qT = q.astype(np.float32)
    return dict(
        q0=q0, qT=qT, trace=trace,
        mu_hat=float(mu_hat), c_hat=float(mu_hat / lam),
        log_w_mean=float(log_w_mean),
        ess_mean=float(ess_sum / max(ess_cnt, 1)),
        n_unique=int(np.unique(qT, axis=0).shape[0]),
        log2sigma=float(log2sigma), kappa=float(kappa),
        seconds=float(time.perf_counter() - t_start),
    )


def _tag(x) -> str:
    return str(x).replace('.', 'p').replace('-', 'm')


def _cache_path(cache_dir: str, name: str, log2sigma: float, cfg: IPMConfig) -> Path:
    """Construct a cache filename containing the numerical parameters."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    fn = (f"{name}_l2s{_tag(log2sigma)}_M{cfg.M}_T{cfg.T}_seed{cfg.seed}"
          f"_lam{_tag(cfg.lam)}_th{_tag(cfg.theta)}_dt{_tag(cfg.log2dt)}"
          f"_re{cfg.re}")
    return d / (fn + '.npz')


def load_or_make_2d(log2sigma: float, cfg: IPMConfig,
                    cache_dir: Optional[str] = 'kpp2d_cache',
                    name: str = 'ref',
                    progress_every: int = 0) -> Dict[str, object]:
    """Load a compatible cached simulation or run kpp2d_fk and cache its output."""
    p = _cache_path(cache_dir, name, log2sigma, cfg) if cache_dir else None
    if p is not None and p.exists():
        data = np.load(p)
        meta = json.loads(str(data['meta']))
        if meta.get('cfg') == asdict(cfg) and \
                abs(meta.get('log2sigma', 1e9) - log2sigma) < 1e-12:
            out = dict(meta)
            out.pop('cfg', None)
            out['q0'], out['qT'] = data['q0'], data['qT']
            out['trace'] = [tuple(t) for t in meta.get('trace', [])]
            return out
        warnings.warn(f'[cache mismatch] recomputing {p.name}')

    res = kpp2d_fk(log2sigma, cfg, progress_every=progress_every)
    if p is not None:
        meta = {k: v for k, v in res.items() if k not in ('q0', 'qT')}
        meta['cfg'] = asdict(cfg)
        np.savez_compressed(p, q0=res['q0'], qT=res['qT'], meta=json.dumps(meta))
    return res


def _fmt_secs(s: float) -> str:
    return f'{s/60:.1f} min' if s < 3600 else f'{s/3600:.2f} h'


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Generate 2D periodic KPP reference particle clouds.')
    ap.add_argument('--particles', type=int, default=20000, help='Particle count.')
    ap.add_argument('--generations', type=int, default=4096, help='Number of unit-period generations.')
    ap.add_argument('--seed', type=int, default=666)
    ap.add_argument('--cache', default='kpp2d_cache', help='Cache directory; an empty string disables caching.')
    ap.add_argument('--progress', type=int, default=64, help='Print progress every N generations; zero disables progress output.')
    ap.add_argument('--only-log2sigma', nargs='+', default=None, metavar='L2S',
                    help='Run only selected log2(diffusivity) levels. Each level has an independent cache and can run in a separate process.')
    ap.add_argument('--resample-every', dest='resample_every', default='gen',
                    metavar='N|gen',
                    help="Resample every N integration substeps; 'gen' selects once per generation (the default).")
    args = ap.parse_args()

    cfg = IPMConfig(M=args.particles, T=args.generations, seed=args.seed)
    if args.resample_every != 'gen':
        cfg = IPMConfig(**{**asdict(cfg), 'resample_every': int(args.resample_every)})

    levels = eval_log2sigma()
    if args.only_log2sigma:
        want = [float(x) for x in args.only_log2sigma]
        missing = [w for w in want if not any(abs(l - w) < 1e-9 for l in levels)]
        if missing:
            raise SystemExit(f'{missing} are outside the evaluation levels: {levels}')
        levels = [l for l in levels if any(abs(l - w) < 1e-9 for w in want)]

    print(f'=== 2D KPP reference particles: {len(levels)} 个 sigma, M={cfg.M}, T={cfg.T}, '
          f'rn={cfg.rn}, resample_every={cfg.re}, seed={cfg.seed} ===', flush=True)
    t0 = time.perf_counter()
    train = training_log2sigma()
    for i, l2s in enumerate(levels):
        kind = 'train' if any(abs(l2s - t) < 1e-9 for t in train) else 'extrap'
        print(f'[{i+1}/{len(levels)}] log2sigma = {l2s:.2f}  '
              f'(-log2 sigma = {-l2s:.2f}, {kind})', flush=True)
        res = load_or_make_2d(l2s, cfg, cache_dir=args.cache or None,
                              progress_every=args.progress)
        print(f'    c_hat = {res["c_hat"]:.6f}   ESS = {res["ess_mean"]:.0f}/{cfg.M}   '
              f'unique = {res["n_unique"]}/{cfg.M}   {_fmt_secs(res["seconds"])}', flush=True)
        print('    trace: ' + ', '.join(f'T={t}:{c:.4f}' for t, _, c in res['trace'][:8]),
              flush=True)
    print(f'total {_fmt_secs(time.perf_counter() - t0)}')


if __name__ == '__main__':
    main()
