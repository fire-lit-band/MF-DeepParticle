"""Render the seven KS figures used by mainV2.tex from retained results.

The W2 plotting specifications are adapted from
output/ks_histogram_adoption_20261023/prepare_figures.py. Projection histogram
and layout specifications are adapted from
output/mf_matched_manuscript_20261016/figures.py:projection_rows, with the
accepted ``DP refinement`` label. This module reads saved results only;
it neither loads training code nor edits manuscripts or historical results.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
    "axes.linewidth": .8, "legend.fontsize": 9, "axes.labelsize": 12,
}
COLORS = {"mf": "#2864B4", "dp": "#D37B2F", "mf_dp": "#168173", "mf_two_step": "#D37B2F"}
LABELS = {"mf": "MF", "dp": "Direct DP", "mf_dp": "DP refinement", "mf_two_step": "MF (2 steps)"}
MARKERS = {"mf": "o", "dp": "^", "mf_dp": "s", "mf_two_step": "^"}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _save(fig, destination: Path, relative: str) -> Path:
    path = destination / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=250, facecolor="white")
    finally:
        plt.close(fig)
    return path


def _curves(destination: Path) -> list[Path]:
    summary = _read(ROOT / "output/ks_full_histogram_100k_bins16_20261022/summary.csv")
    multi = _read(ROOT / "output/ks_histogram_adoption_20261023/multistep_summary.csv")
    assert len(summary) == 120 and len(multi) == 20
    global_max = max(float(row["max_w2"]) for row in summary)
    outputs = []
    specs = [
        ("ks3d", ("mf", "dp", "mf_dp"), "ks3d_w2"),
        ("ks4d", ("mf", "dp", "mf_dp"), "ks4d_w2"),
        ("ks3d", ("mf", "mf_two_step", "mf_dp"), "ks3d_multistep_w2"),
    ]
    for name, methods, stem in specs:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        selected = []
        for method in methods:
            values = sorted(
                [row for row in summary + multi if row["model"] == name and row["method"] == method],
                key=lambda row: int(row["sigma"]),
            )
            assert len(values) == 20
            x = np.array([int(row["sigma"]) for row in values])
            assert x.tolist() == list(range(10, 201, 10))
            y = np.array([float(row["mean_w2"]) for row in values])
            low = np.array([float(row["min_w2"]) for row in values])
            high = np.array([float(row["max_w2"]) for row in values])
            assert np.isfinite([y, low, high]).all() and (low <= y).all() and (y <= high).all()
            line, = ax.plot(x, y, color=COLORS[method], marker=MARKERS[method], ms=3.5,
                            lw=1.6, label=LABELS[method], zorder=3)
            np.testing.assert_array_equal(line.get_ydata(), y)
            ax.fill_between(x, low, high, color=COLORS[method], alpha=.13, lw=0, zorder=1)
            selected += values
        top = global_max if "multistep" not in stem else max(float(row["max_w2"]) for row in selected)
        ax.set(xlabel=r"$\sigma$", ylabel=r"$W_2$", xlim=(5, 205), ylim=(0, top * 1.05))
        ax.set_xticks([20, 50, 100, 150, 200])
        ax.axvline(150, color="#777777", ls="--" if "multistep" in stem else ":", lw=1.3)
        ax.grid(alpha=.23, ls="--")
        ax.legend(frameon=False, loc="upper left")
        outputs.append(_save(fig, destination, f"ks_histogram_100k_bins16_20261023/{stem}.pdf"))
    return outputs


def _projections(clouds, destination: Path, stem: str, *, bins=60, density=False, combined=False) -> list[Path]:
    pairs = [(0, 1), (0, 2), (1, 2)]
    assert all(points.shape == (100000, 3) and np.isfinite(points).all() for points in clouds.values())
    lows = np.min([points.min(0) for points in clouds.values()], axis=0)
    highs = np.max([points.max(0) for points in clouds.values()], axis=0)
    if bins is None:
        edges = [np.arange(np.floor(lo / .05) * .05 - .05, np.ceil(hi / .05) * .05 + .075, .05)
                 for lo, hi in zip(lows, highs)]
    else:
        edges = [np.linspace(np.floor((lo - .05) / .25) * .25,
                             np.ceil((hi + .05) / .25) * .25, bins + 1)
                 for lo, hi in zip(lows, highs)]
    histograms = []
    for key, cloud in clouds.items():
        row = []
        for a, b in pairs:
            counts = np.histogram2d(cloud[:, a], cloud[:, b], bins=(edges[a], edges[b]))[0]
            assert counts.sum() == len(cloud), (key, a, b, counts.sum(), len(cloud))
            if density:
                counts = counts / len(cloud) / np.diff(edges[a])[:, None] / np.diff(edges[b])[None, :]
            row.append(counts)
        histograms.append(row)
    maxima = [max(row[col].max() for row in histograms) for col in range(3)]
    outputs = []
    if combined:
        fig, axes = plt.subplots(len(clouds), 3, figsize=(9.4, 6.0), layout="constrained")
    for ri, (key, row) in enumerate(zip(clouds, histograms)):
        if combined:
            axs = axes[ri]
        else:
            fig, axs = plt.subplots(1, 3, figsize=(9.5, 2.7), layout="constrained")
        for col, (ax, (a, b), counts) in enumerate(zip(axs, pairs, row)):
            mesh = ax.imshow(counts.T, origin="lower",
                             extent=[edges[a][0], edges[a][-1], edges[b][0], edges[b][-1]],
                             norm=Normalize(0, maxima[col]), cmap="viridis", aspect="auto", interpolation="nearest")
            ax.set(xlabel="$" + "xyz"[a] + "$", ylabel="$" + "xyz"[b] + "$")
            ax.tick_params(labelsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            title = LABELS.get(key, "Reference") if combined or col == 1 else ""
            ax.set_title(title, fontsize=12)
            fig.colorbar(mesh, ax=ax, pad=.025, shrink=.88).ax.tick_params(labelsize=8)
        if not combined:
            outputs.append(_save(fig, destination, f"selected_dp_results_20261017/dplus1_{stem}_{key}_20261016.pdf"))
    if combined:
        outputs.append(_save(fig, destination, f"selected_dp_results_20261017/dplus1_{stem}_20261016.pdf"))
    return outputs


def render(destination: Path) -> list[Path]:
    """Write exactly seven PDF figures beneath destination; read saved results only."""
    destination = Path(destination)
    source = ROOT / "output/dp_alpha01_refinement_alpha10_20261017"
    with plt.rc_context(STYLE):
        outputs = _curves(destination)
        with np.load(source / "benchmarks/ks3d/evaluation/particles_15.npz") as data:
            reference = data["reference"].copy()
            refined = data["seed88_mf_dp"].copy()
            clouds = {"ref": reference, "mf": data["seed88_mf"].copy(), "mf_dp": refined}
        outputs += _projections(clouds, destination, "ks3d_projection", bins=None, density=True, combined=True)
        with np.load(source / "multistep_particles/sigma160.npz") as data:
            np.testing.assert_array_equal(data["reference"], reference)
            clouds = {"mf_two_step": data["seed88_mf_two_step"].copy(), "mf_dp": refined, "ref": reference}
        outputs += _projections(clouds, destination, "multistep_projection", bins=60)
    assert len(outputs) == 7
    return outputs
