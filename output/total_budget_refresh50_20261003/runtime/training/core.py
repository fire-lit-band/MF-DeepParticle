"""Shared original3 KS numerics and training kernels.

No archived experiment is imported at runtime. Entry points handle data,
configuration and evaluation in training.workflow.
"""
from __future__ import annotations
import csv
import hashlib
import inspect
import json
import logging
import math
import numbers
import os
import random
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.func import jvp
import ot
from scipy.spatial.distance import cdist

TRAIN_SIGMA = (1.5 * 10 ** torch.linspace(1, 2, 8)).numpy()
TEST_SIGMA = np.arange(10.0, 201.0, 10.0)
PHYSICS_ID = "mainv2_positive_regularized_newtonian_current_gradient_v1"
DP_OBJECTIVE = "stage2_ot_plus_residual_v2"
OT_SOLVER = "pot.ot.emd_network_simplex"
DEFAULT_RESIDUAL_WEIGHT = 1.0
DEFAULT_BATCH_REFRESH = 50
DEFAULT_OT_REFRESH = 50
DEFAULT_OT_WARMSTART = True
DEFAULT_OT_MAX_ITER = 1_000_000
DEFAULT_DP_BLOCKS = 2


def validate_ot_warmstart(enabled):
    if enabled and "potentials_init" not in inspect.signature(ot.emd).parameters:
        raise RuntimeError("OT warm start requires POT >= 0.9.7; install requirements-training.txt or use --no-ot-warmstart")


def validate_ot_max_iter(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError("ot-max-iter must be a positive integer")
    return int(value)


def ot_schedule_metadata(args):
    return {"ot_warmstart": getattr(args, "ot_warmstart", DEFAULT_OT_WARMSTART),
            "ot_warmstart_state": "per-sigma dual potentials; reset on every batch change",
            "batch_refresh": args.batch_refresh, "ot_refresh": args.ot_refresh,
            "ot_max_iter": validate_ot_max_iter(getattr(args, "ot_max_iter", DEFAULT_OT_MAX_ITER)),
            "ot_allow_approximate": bool(getattr(args, "ot_allow_approximate", False)),
            "pot_version": ot.__version__}


def write_csv(path, rows):
    if not rows:
        return
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))


def sigma_feature(sigma: torch.Tensor) -> torch.Tensor:
    return (torch.log10(sigma) - math.log10(15.0)) / math.log10(10.0)


class Meanflow(nn.Module):
    """Original notebook MLP: 6 inputs, three 64-wide SELU hidden layers.

    The public order stays (z, r, t, sigma). Internally the notebook's order
    (z, t, r, log10(sigma)) and default Linear initialization are preserved.
    """
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, width), nn.SELU(),
            nn.Linear(width, width), nn.SELU(),
            nn.Linear(width, width), nn.SELU(),
            nn.Linear(width, 3),
        )

    def forward(self, z, r, t, sigma):
        return self.net(torch.cat((z, t, r, torch.log10(sigma)), dim=-1))

    def forward_logsigma(self, z, r, t, logsigma):
        """Precomputed scalar feature, including the notebook's CPU rounding."""
        return self.net(torch.cat((z, t, r, logsigma), dim=-1))


class DPCorrector(nn.Module):
    def __init__(self, width: int = 64, alpha: float = 0.2, blocks: int = DEFAULT_DP_BLOCKS):
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, numbers.Integral) or width <= 0:
            raise ValueError("DP width must be a positive integer")
        if isinstance(blocks, bool) or not isinstance(blocks, numbers.Integral) or blocks <= 0:
            raise ValueError("DP block count must be a positive integer")
        width = int(width)
        self.alpha = alpha
        self.input = nn.Linear(4, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
            for _ in range(int(blocks))
        )
        self.output = nn.Linear(width, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def structure_from_state_dict(state):
        """Read the saved structure without changing tensors or random state."""
        weight = state.get("input.weight")
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[1] != 4:
            raise ValueError("DP checkpoint has an invalid input projection")
        indices = sorted({int(key.split(".")[1]) for key in state if key.startswith("blocks.")})
        if not indices or indices != list(range(len(indices))):
            raise ValueError("DP checkpoint has invalid residual block indices")
        return int(weight.shape[0]), len(indices)

    @classmethod
    def from_state_dict(cls, state, *, alpha=.1, width=None, blocks=None):
        """Load small and historical correctors using their saved structure."""
        saved_width, saved_blocks = cls.structure_from_state_dict(state)
        if width is not None and width != saved_width:
            raise ValueError("DP width metadata differs from checkpoint weights")
        if blocks is not None and blocks != saved_blocks:
            raise ValueError("DP block count metadata differs from checkpoint weights")
        model = cls(width=saved_width, alpha=alpha, blocks=saved_blocks)
        model.load_state_dict(state, strict=True)
        return model

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        h = self.input(torch.cat((x, sigma_feature(sigma)), dim=-1))
        for block in self.blocks:
            h = h + 0.2 * block(h)
        return x + self.alpha * self.output(h)


def validate_residual_weight(residual_weight=DEFAULT_RESIDUAL_WEIGHT):
    """Return a finite, nonnegative scalar lambda; reject ambiguous values."""
    if (isinstance(residual_weight, (bool, np.bool_))
            or not isinstance(residual_weight, numbers.Real)):
        raise ValueError("residual_weight must be a finite nonnegative real number")
    weight = float(residual_weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("residual_weight must be a finite nonnegative real number")
    return weight


def stage2_loss(refined, matched, source,
                residual_weight=DEFAULT_RESIDUAL_WEIGHT, reduction="mean"):
    """Return ``(total, transport, displacement_penalty)`` for frozen matching.

    Inputs have identical ``(..., B, D)`` shapes. ``matched`` is the target
    cloud reordered by :func:`assignment`, not a barycentric projection of a
    fractional plan. Matching and MF source tensors are fixed Stage II data;
    gradients flow only through ``refined``.
    """
    weight = validate_residual_weight(residual_weight)
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    if (refined.ndim < 2 or refined.shape != matched.shape
            or refined.shape != source.shape or refined.numel() == 0):
        raise ValueError("refined, matched and source need the same nonempty (..., B, D) shape")
    transport_per_particle = (refined - matched.detach()).square().sum(dim=-1)
    residual_per_particle = (refined - source.detach()).square().sum(dim=-1)
    if reduction == "mean":
        transport = transport_per_particle.mean()
        residual = residual_per_particle.mean()
    else:
        transport = transport_per_particle.sum()
        residual = residual_per_particle.sum()
    return transport + weight * residual, transport, residual


def _cost_matrix(pair):
    current, target = (np.asarray(points, dtype=np.float64) for points in pair)
    if (current.ndim != 2 or target.ndim != 2 or current.shape != target.shape
            or not current.size or not np.isfinite(current).all()
            or not np.isfinite(target).all()):
        raise ValueError("exact mini-batch OT requires equal-size finite nonempty (B, D) clouds")
    cost = cdist(current, target, metric="sqeuclidean")
    if not np.isfinite(cost).all():
        raise ValueError("squared OT costs overflowed; rescale the particle coordinates")
    return cost


def _gamma_from_cost(cost, *, potentials_init=None, return_info=False,
                     num_iter_max=DEFAULT_OT_MAX_ITER, allow_approximate=False):
    num_iter_max = validate_ot_max_iter(num_iter_max)
    batch_size = len(cost)
    mass = np.full(batch_size, 1.0 / batch_size, dtype=np.float64)
    # POT's network simplex solves the unregularized problem specified in the
    # manuscript. Iteration-limited solutions require explicit opt-in and the
    # same feasibility/permutation checks as converged solutions.
    options = {}
    if potentials_init is not None:
        validate_ot_warmstart(True)
        potentials_init = tuple(np.asarray(value, dtype=np.float64) for value in potentials_init)
        if (len(potentials_init) != 2 or
                any(value.shape != (batch_size,) or not np.isfinite(value).all() for value in potentials_init)):
            raise ValueError("Warm-start potentials must be two finite vectors matching the particle batch")
        options["potentials_init"] = potentials_init
    started = time.perf_counter()
    plan, info = ot.emd(mass, mass, cost, numItermax=num_iter_max, log=True, **options)
    solve_seconds = time.perf_counter() - started
    result_code = int(info["result_code"])
    if result_code != 1 and not (allow_approximate and result_code == 3):
        raise RuntimeError(f"Exact POT OT solve failed: {info['warning']}")
    gamma = batch_size * np.asarray(plan, dtype=np.float64)
    if (gamma.shape != cost.shape or not np.isfinite(gamma).all()
            or np.any(gamma < -1e-10)
            or not np.allclose(gamma.sum(axis=0), 1.0, rtol=0, atol=1e-8)
            or not np.allclose(gamma.sum(axis=1), 1.0, rtol=0, atol=1e-8)):
        raise RuntimeError("Exact POT OT solve returned an invalid unit-marginal coupling")
    if return_info:
        if any(np.asarray(info[name]).shape != (batch_size,) or not np.isfinite(info[name]).all()
               for name in ("u", "v")):
            raise RuntimeError("Exact POT OT solve returned invalid dual potentials")
        return gamma, {**info, "solve_seconds": solve_seconds,
                       "optimal": result_code == 1, "iteration_limited": result_code == 3,
                       "num_iter_max": num_iter_max}
    return gamma


def assignment(pair, *, potentials_init=None, return_state=False,
               num_iter_max=DEFAULT_OT_MAX_ITER, allow_approximate=False):
    """Return a verified permutation and mean squared transport cost from POT EMD.

    Uniform equal-size discrete OT admits a permutation optimum. Before using
    target indexing for the loss, verify that the returned numerical coupling
    really is a permutation. A fractional coupling cannot be silently rounded
    using argmax because that changes the training objective.
    With explicit ``allow_approximate``, an iteration-limited feasible
    permutation can be returned; its state records that it is not optimal.
    """
    started = time.perf_counter()
    cost = _cost_matrix(pair)
    cost_seconds = time.perf_counter() - started
    gamma, info = _gamma_from_cost(cost, potentials_init=potentials_init, return_info=True,
                                 num_iter_max=num_iter_max, allow_approximate=allow_approximate)
    columns = gamma.argmax(axis=1)
    permutation = np.zeros_like(gamma)
    permutation[np.arange(len(gamma)), columns] = 1.0
    if (not np.array_equal(np.sort(columns), np.arange(len(gamma)))
            or not np.allclose(gamma, permutation, rtol=0, atol=1e-8)):
        raise RuntimeError("POT coupling is not a permutation; cannot approximate it with target indices")
    result = columns, float(np.sum(cost * gamma) / len(gamma))
    if return_state:
        return (*result, {"potentials": (np.asarray(info["u"]).copy(), np.asarray(info["v"]).copy()),
                          "warmstart_used": potentials_init is not None,
                          "solve_seconds": info["solve_seconds"], "cost_seconds": cost_seconds,
                          "optimal": info["optimal"], "iteration_limited": info["iteration_limited"],
                          "result_code": int(info["result_code"]), "num_iter_max": num_iter_max})
    return result


def stage2_metadata(residual_weight=DEFAULT_RESIDUAL_WEIGHT):
    """Fields required in each Stage II config and resumable checkpoint."""
    return {"dp_objective": DP_OBJECTIVE,
            "residual_weight": validate_residual_weight(residual_weight),
            "ot_solver": OT_SOLVER}


def histogram(points, edges):
    counts, _ = np.histogramdd(points, bins=edges, density=False)
    retained = int(counts.sum())
    if retained != len(points):
        raise ValueError(f"Histogram lost {len(points) - retained} particles")
    index = np.argwhere(counts > 0)
    centers = [0.5 * (edge[1:] + edge[:-1]) for edge in edges]
    positions = np.column_stack([centers[d][index[:, d]] for d in range(3)])
    weights = counts[tuple(index.T)].astype(np.float64) / retained
    return positions, weights, retained


def histogram_ot(first, second):
    cost = cdist(first[0], second[0], metric="sqeuclidean")
    value, info = ot.emd2(first[1], second[1], cost, log=True, numItermax=5_000_000)
    if info.get("warning"):
        raise RuntimeError(f"OT solver warning: {info['warning']}")
    if not np.isfinite(value) or value < -1e-10:
        raise ValueError(f"Invalid OT cost: {value}")
    return max(float(value), 0.0)


def configure_device(device):
    """Keep full float32 precision and deterministic kernels on either device."""
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but no CUDA device is available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def mf_sample(model, source, sigmas):
    """Device-aware inference with the same CPU source pools as historical runs."""
    device = next(model.parameters()).device
    source = source.detach().cpu()
    outputs = []
    for i, sigma in enumerate(sigmas):
        value = torch.full((len(source[i]), 1), float(sigma), dtype=source.dtype).to(device)
        result = model.forward_logsigma(source[i].to(device), torch.zeros_like(value),
            torch.ones_like(value), torch.full_like(value, math.log10(float(sigma))))
        outputs.append(result.cpu())
    return source - torch.stack(outputs)


@torch.no_grad()
def dp_sample(model, source, sigmas):
    device = next(model.parameters()).device
    outputs = []
    for i, sigma in enumerate(sigmas):
        value = torch.full((len(source[i]), 1), float(sigma), dtype=source.dtype).to(device)
        outputs.append(model(source[i].to(device), value).cpu())
    return torch.stack(outputs)


def unit_ball(n):
    radius = torch.rand(n, 1).pow(1 / 3)
    direction = torch.randn(n, 3)
    return radius * direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)


def meanflow_loss(error):
    """Mean over particles of the squared vector error (sum over coordinates)."""
    if error.ndim != 2 or min(error.shape) == 0:
        raise ValueError("MF error must have shape (nonempty batch, vector dimension)")
    squared = error.square().sum(-1)
    return squared.mean(), squared


def train_mf(model, initial, target, args, run):
    device = torch.device(getattr(args, "device", next(model.parameters()).device))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mf_lr)
    sigmas = torch.as_tensor(getattr(args, "train_sigmas", TRAIN_SIGMA), dtype=torch.float32)
    milestones = set(getattr(args, "checkpoint_steps", []))
    rows, sigma_rows = [], []
    for step in range(1, args.mf_steps + 1):
        env = torch.randint(len(sigmas), (1,)).expand(args.batch_size)
        idx = torch.randint(args.n_train, (args.batch_size,))
        x0, x1 = target[env, idx].to(device), initial[idx].to(device)
        from .protocol import sample_times_original
        r, t = sample_times_original(args.batch_size, args.time_mean, args.time_std)
        r, t = r.to(device), t.to(device)
        velocity = x1 - x0
        z = (1 - t) * x0 + t * x1
        condition = torch.log10(sigmas[env, None]).to(device)
        u, derivative = jvp(lambda zz, rr, tt: model.forward_logsigma(zz, rr, tt, condition),
                            (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)))
        desired = (velocity - (t - r) * derivative).detach()
        loss, squared = meanflow_loss(u - desired)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"MF loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        rows.append({"step": step, "loss": loss.item(), "mse": squared.detach().mean().item()})
        if step == 1 or step % 250 == 0 or step == args.mf_steps:
            logging.info("%s MF %d/%d loss=%.6g", run.name, step, args.mf_steps, loss.item())
            sigma_rows.append({"step": step, "sigma": float(sigmas[env[0]]),
                               "count": args.batch_size, "mse": loss.item(), "loss": loss.item()})
        if step % 2500 == 0 or step == args.mf_steps or step in milestones:
            from .protocol import rng_state
            payload = {"model": cpu_state(model), "step": step,
                       "optimizer": optimizer.state_dict(), **rng_state()}
            torch.save(payload, run / "meanflow.pt")
            if step in milestones:
                torch.save(payload, run / f"meanflow_step{step}.pt")
            write_csv(run / "meanflow_loss.csv", rows)
            write_csv(run / "meanflow_loss_by_sigma.csv", sigma_rows)
    return model.eval()


def train_dp(model, source, target, args, run):
    """Run every requested update and save fixed-budget checkpoints."""
    validate_residual_weight(args.residual_weight)
    warmstart = getattr(args, "ot_warmstart", DEFAULT_OT_WARMSTART)
    validate_ot_warmstart(warmstart)
    ot_max_iter = validate_ot_max_iter(getattr(args, "ot_max_iter", DEFAULT_OT_MAX_ITER))
    ot_allow_approximate = bool(getattr(args, "ot_allow_approximate", False))
    device = torch.device(getattr(args, "device", next(model.parameters()).device))
    model.to(device)
    sigmas = getattr(args, "train_sigmas", TRAIN_SIGMA)
    milestones = set(getattr(args, "checkpoint_steps", []))
    sigma_count = getattr(args, "dp_sigmas_per_update", 1)
    if sigma_count not in (1, len(TRAIN_SIGMA)):
        raise ValueError("DP supports one condition or all training conditions per update")
    particle_count = sigma_count * args.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=args.dp_lr)
    rows, refreshes = [], []
    matched, last_refresh, batch_count = None, 0, 0
    ot_states = {}
    checkpoint_path = run / "dp_corrector.pt"

    def save_checkpoint(step):
        from .protocol import rng_state
        payload = {"model": cpu_state(model), "step": step,
                    "optimizer": optimizer.state_dict(), **rng_state(),
                    "selection_mode": "fixed_budget_final",
                    "training_device": str(device), "checkpoint_tensor_device": "cpu",
                    "sigmas_per_update": sigma_count, "particles_per_update": particle_count,
                    "alpha": getattr(args, "alpha", getattr(model, "alpha", None)), "loss_reduction": "sum",
                    "dp_width": model.input.out_features,
                    "dp_blocks": len(model.blocks),
                    **stage2_metadata(args.residual_weight), **ot_schedule_metadata(args)}
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)
        if step in milestones:
            milestone = run / f"dp_step{step}.pt"
            temporary = milestone.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(milestone)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    for step in range(1, args.dp_steps + 1):
        model.train()
        batch_refreshed = (step - 1) % args.batch_refresh == 0
        if batch_refreshed:
            # Particle identities and their order change: indexed potentials
            # from the previous batch must never initialize this solve.
            ot_states.clear()
            environments = list(range(len(TRAIN_SIGMA))) if sigma_count > 1 else [int(torch.randint(len(TRAIN_SIGMA), ()).item())]
            source_batch, target_batch, batch_signature = [], [], hashlib.sha256()
            for env in environments:
                source_idx = torch.randperm(args.n_train)[:args.batch_size]
                target_idx = torch.randperm(args.n_train)[:args.batch_size]
                batch_signature.update(np.concatenate(([env], source_idx.numpy(), target_idx.numpy())).tobytes())
                source_batch.append(source[env, source_idx])
                target_batch.append(target[env, target_idx])
            signature = batch_signature.hexdigest()
            if sigma_count == 1:
                x, y = source_batch[0].to(device), target_batch[0].to(device)
                sigma = torch.full((args.batch_size, 1), float(sigmas[environments[0]])).to(device)
                sigma_label = float(sigmas[environments[0]])
            else:
                x, y = torch.stack(source_batch).to(device), torch.stack(target_batch).to(device)
                sigma = torch.as_tensor(sigmas, dtype=x.dtype).view(-1, 1, 1).expand(-1, args.batch_size, 1).to(device)
                sigma_label = "all_training_sigmas"
            batch_count += 1
        current = model(x, sigma)
        refreshed = matched is None or batch_refreshed or (step - 1) % args.ot_refresh == 0
        if refreshed:
            current_cpu, target_cpu = current.detach().cpu().numpy(), y.cpu().numpy()
            pairs = [(current_cpu, target_cpu)] if sigma_count == 1 else list(zip(current_cpu, target_cpu))
            solutions = []
            for env, pair in zip(environments, pairs):
                previous = ot_states.get(env) if warmstart else None
                if previous is not None and previous["batch_sha256"] != signature:
                    raise RuntimeError("Cannot warm-start OT with potentials from another particle batch")
                solution = assignment(pair, potentials_init=previous["potentials"] if previous else None,
                                      return_state=True, num_iter_max=ot_max_iter,
                                      allow_approximate=ot_allow_approximate)
                solutions.append(solution)
                if warmstart:
                    ot_states[env] = {"batch_sha256": signature, "potentials": solution[2]["potentials"]}
            permutation = solutions[0][0] if sigma_count == 1 else np.stack([item[0] for item in solutions])
            matched = torch.as_tensor(permutation, dtype=torch.long, device=device)
            refreshes.append({"step": step, "sigma": sigma_label,
                              "batch_refreshed": int(batch_refreshed),
                              "batch_sha256": signature,
                              "fresh_ot_mean_cost": statistics.mean(item[1] for item in solutions),
                              "warmstart_solves": sum(item[2]["warmstart_used"] for item in solutions),
                              "coldstart_solves": sum(not item[2]["warmstart_used"] for item in solutions),
                              "optimal_solves": sum(item[2]["optimal"] for item in solutions),
                              "iteration_limited_solves": sum(item[2]["iteration_limited"] for item in solutions),
                              "ot_solve_seconds": sum(item[2]["solve_seconds"] for item in solutions),
                              "ot_cost_seconds": sum(item[2]["cost_seconds"] for item in solutions),
                              "per_sigma_ot_mean_costs": json.dumps({str(float(sigmas[env])): solution[1]
                                                                     for env, solution in zip(environments, solutions)})})
            last_refresh = step
        matched_target = y[matched] if sigma_count == 1 else y.gather(1, matched[..., None].expand_as(y))
        loss, transport, residual = stage2_loss(current, matched_target, x,
                                               args.residual_weight, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"DP loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        rows.append({"step": step, "sigma": sigma_label, "loss_sum": loss.item(),
                     "sigmas_per_update": sigma_count, "particles_per_update": particle_count,
                     "transport_mean": transport.item() / particle_count,
                     "residual_mean": residual.item() / particle_count,
                     "batch_refreshed": int(batch_refreshed), "ot_refreshed": int(refreshed),
                     "coupling_age": step - last_refresh, "batch_sha256": signature})
        if step in milestones:
            write_csv(run / "dp_loss.csv", rows)
            write_csv(run / "ot_refreshes.csv", refreshes)
            save_checkpoint(step)
        if step == 1 or step % 250 == 0 or step == args.dp_steps:
            logging.info("%s DP %d/%d OT/particle=%.6g residual/particle=%.6g sigmas=%d", run.name, step,
                         args.dp_steps, transport.item() / particle_count, residual.item() / particle_count, sigma_count)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    save_checkpoint(step)
    write_csv(run / "dp_loss.csv", rows)
    write_csv(run / "ot_refreshes.csv", refreshes)
    audit = {"updates": step, "max_updates": args.dp_steps, "batch_count": batch_count,
             "sigmas_per_update": sigma_count, "batch_per_sigma": args.batch_size,
             "particles_per_update": particle_count, "ot_solves": len(refreshes) * sigma_count,
             "loss_normalization": "joint sum across conditions and particles; no division by condition count",
             "batch_refresh_steps": [row["step"] for row in rows if row["batch_refreshed"]],
             "ot_refresh_steps": [row["step"] for row in refreshes],
             "fresh_ot_updates": len(refreshes), "cross_batch_coupling_reuse": 0,
              "cross_batch_warmstart_reuse": 0,
              "warmstart_solves": sum(row["warmstart_solves"] for row in refreshes),
              "coldstart_solves": sum(row["coldstart_solves"] for row in refreshes),
              "optimal_solves": sum(row["optimal_solves"] for row in refreshes),
              "iteration_limited_solves": sum(row["iteration_limited_solves"] for row in refreshes),
              "ot_solve_seconds": sum(row["ot_solve_seconds"] for row in refreshes),
              "ot_cost_seconds": sum(row["ot_cost_seconds"] for row in refreshes),
              "training_seconds": training_seconds,
             "initial_update_uses_ot": refreshes[0]["step"] == 1,
             "selected_step": step,
             "stopped_early": False, "early_stopping_enabled": False, "selection_before_test": True,
             "selection_mode": "fixed_budget_final",
              **stage2_metadata(args.residual_weight), **ot_schedule_metadata(args)}
    (run / "schedule_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    model.training_summary = audit
    return model.eval()
