"""Periodic and Euclidean exact OT, with W2 and W2 squared kept distinct."""
from __future__ import annotations
import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import ot
from scipy.spatial.distance import cdist
from .physics import TWO_PI


def difference(first, second, periodic):
    delta = first - second
    return (delta + math.pi) % TWO_PI - math.pi if periodic else delta


def costs(first, second, periodic=False):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1] or min(len(first), len(second)) < 1:
        raise ValueError("OT requires nonempty clouds of the same dimension")
    if periodic:
        result = np.zeros((len(first), len(second)), dtype=np.float64)
        for axis in range(first.shape[1]):
            delta = first[:, None, axis] - second[None, :, axis]
            delta += math.pi
            np.remainder(delta, TWO_PI, out=delta)
            delta -= math.pi
            np.square(delta, out=delta)
            result += delta
    else:
        result = cdist(first, second, metric="sqeuclidean")
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite OT costs")
    return result


def exact_assignment(first, second, periodic=False):
    # Reuse the audited POT solver and permutation validation, with the correct
    # problem-specific ground cost rather than its legacy Euclidean wrapper.
    from training.core import _gamma_from_cost
    cost = costs(first, second, periodic)
    if cost.shape[0] != cost.shape[1]:
        raise ValueError("Training OT batches must have equal size")
    gamma = _gamma_from_cost(cost)
    columns = gamma.argmax(1)
    expected = np.zeros_like(gamma)
    expected[np.arange(len(columns)), columns] = 1
    if not np.array_equal(np.sort(columns), np.arange(len(columns))) or not np.allclose(gamma, expected, atol=1e-8, rtol=0):
        raise RuntimeError("Exact OT did not yield a verified permutation")
    return columns


def emd_squared(first, second, periodic=False, weights=None):
    cost = costs(first, second, periodic)
    a, b = (np.ones(len(first)) / len(first), np.ones(len(second)) / len(second)) if weights is None else weights
    value, info = ot.emd2(a, b, cost, numItermax=1_000_000, log=True)
    if info["result_code"] != 1:
        raise RuntimeError(f"Evaluation OT did not converge: {info['warning']}")
    return max(0., float(value))


def cloud_metric(reference, generated, periodic, sample_size, seed):
    rng = np.random.default_rng(seed)
    size = min(sample_size, len(reference), len(generated))
    first = reference[rng.choice(len(reference), size, replace=False)]
    second = generated[rng.choice(len(generated), size, replace=False)]
    squared = emd_squared(first, second, periodic)
    return {"w2": math.sqrt(squared), "w2_squared": squared, "particles_per_cloud": size}


def histogram_metrics(reference, generated_by_method, bins, periodic, projection=None, workers=1):
    all_clouds = {"reference": np.asarray(reference), **generated_by_method}
    if projection is not None:
        all_clouds = {key: cloud[:, projection] for key, cloud in all_clouds.items()}
    dimension = all_clouds["reference"].shape[1]
    if periodic:
        all_clouds = {key: np.remainder(cloud, TWO_PI) for key, cloud in all_clouds.items()}
        edges = [np.linspace(0, TWO_PI, bins + 1)] * dimension
    else:
        union = np.concatenate(list(all_clouds.values()))
        lo, hi = union.min(0), union.max(0)
        padding = np.maximum(.05 * (hi - lo), 1e-5)
        edges = [np.linspace(lo[i] - padding[i], hi[i] + padding[i], bins + 1) for i in range(dimension)]
    centers = [(e[:-1] + e[1:]) / 2 for e in edges]
    support = np.stack(np.meshgrid(*centers, indexing="ij"), -1).reshape(-1, dimension)
    measures = {}
    for key, points in all_clouds.items():
        mass = np.histogramdd(points, bins=edges)[0].ravel()
        if int(mass.sum()) != len(points):
            raise RuntimeError("Histogram discarded particles")
        mask = mass > 0
        measures[key] = support[mask], mass[mask] / mass.sum()
    ref, a = measures["reference"]
    def measure(method):
        gen, b = measures[method]
        squared = emd_squared(ref, gen, periodic, weights=(a, b))
        return method, {"w2": math.sqrt(squared), "w2_squared": squared,
                          "reference_retained": len(reference), "generated_retained": len(all_clouds[method]),
                          "bins_per_axis": bins}
    if workers > 1:
        import psutil
        largest = max(len(measures[method][0]) for method in generated_by_method)
        # Dense exact EMD needs more memory than the ground-cost array alone.
        peak_per_job = max(1, len(ref) * largest * 8 * 8)
        capacity = max(1, int(psutil.virtual_memory().available * .6 / peak_per_job))
        workers = min(workers, capacity)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return dict(executor.map(measure, generated_by_method))
    return dict(measure(method) for method in generated_by_method)
