"""KS-style histogram W2 for non-KPP: full 3D, physical Euclidean cost."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import ot
from scipy.spatial.distance import cdist

PARTICLES = 100000
BINS = 24
ANCHOR_PARTICLES = 10000
MAX_ITERATIONS = 5000000


@dataclass(frozen=True)
class Grid:
    origin: np.ndarray
    width: np.ndarray
    bins: int = BINS

    @classmethod
    def from_reference(cls, reference, bins=BINS):
        points = checked_points(reference)
        if bins < 2:
            raise ValueError('At least two bins per coordinate are required')
        anchor = points[:ANCHOR_PARTICLES]
        low, high = anchor.min(0), anchor.max(0)
        padding = np.maximum(high-low, 1.)*1e-6
        edges = np.stack([np.linspace(low[a]-padding[a], high[a]+padding[a], bins+1)
                          for a in range(3)])
        return cls(edges[:, 0], edges[:, 1]-edges[:, 0], bins)

    def metadata(self):
        return {'origin': self.origin.tolist(), 'bin_widths': self.width.tolist(),
                'initial_bins_per_axis': self.bins, 'dimension': 3, 'periodic': False,
                'anchor_particles': ANCHOR_PARTICLES, 'all_coordinates': True,
                'boundary_clipping': False}


def checked_points(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Finite nonempty 3D point matrix required')
    return points


def histogram(points, grid):
    points = checked_points(points)
    index, counts = np.unique(np.floor((points-grid.origin)/grid.width).astype(np.int64),
                              axis=0, return_counts=True)
    # Extend the same lattice for outliers; never clip or discard particles.
    assert int(counts.sum()) == len(points)
    centers = grid.origin+(index+.5)*grid.width
    weights = counts.astype(np.float64)/len(points)
    return {'centers': centers, 'weights': weights, 'indices': index,
            'counts': counts, 'particles': len(points)}


def score(first, second):
    started = time.perf_counter()
    cost = cdist(first['centers'], second['centers'], 'sqeuclidean')
    value, info = ot.emd2(first['weights'], second['weights'], cost,
                          numItermax=MAX_ITERATIONS, numThreads=1, log=True)
    if info['result_code'] != 1 or info.get('warning') or not np.isfinite(value) or value < -1e-10:
        raise RuntimeError(f'Non-KPP histogram OT failed: {info}')
    squared = max(0., float(value))
    return {'w2': math.sqrt(squared), 'w2_squared': squared,
            'reference_retained': first['particles'], 'generated_retained': second['particles'],
            'reference_occupied_bins': len(first['weights']),
            'generated_occupied_bins': len(second['weights']),
            'solver_result_code': int(info['result_code']), 'seconds': time.perf_counter()-started}
