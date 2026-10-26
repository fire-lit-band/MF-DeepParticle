"""Render only the 13 KPP figures included in the current mainV2 manuscript.

Plot definitions are adapted from the verified retained scripts
``output/mf_matched_manuscript_20261016/figures.py``,
``output/kpp_cubic_histogram_adoption_20261023/prepare_figures.py``, and
``output/figure8_layout_20261023/prepare_figures.py``.  This module reads their
selected numerical inputs directly; it does not import the historical adoption
scripts, retrain models, edit manuscripts, or export unused panels.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "output/dp_alpha01_refinement_alpha10_20261017/benchmarks"
SUMMARY = ROOT / "output/other_histogram_100k_bins16_20261023/summary.csv"
DENSITY_DIR = Path("selected_dp_results_20261017")
W2_DIR = Path("kpp_cubic_histogram_100k_bins16_20261023")
COMPACT_DIR = Path("figure8_compact_20261023")
COLORS = {"mf": "#2864B4", "dp": "#D37B2F", "mf_dp": "#168173", "cold": "#727272"}
LABELS = {"mf": "MF", "dp": "DP", "mf_dp": "DP refinement", "cold": "Cold start", "ref": "Reference"}
MARKERS = {"mf": "o", "dp": "^", "mf_dp": "s", "cold": "v"}
STYLE = {"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
         "axes.linewidth": .8, "legend.fontsize": 9, "axes.labelsize": 12}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _save(fig, destination: Path, relative: Path) -> Path:
    path = destination / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=250, facecolor="white")
    finally:
        plt.close(fig)
    return path


def _clouds(name: str, index: int, methods: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(SELECTED / name / "evaluation" / f"particles_{index:02d}.npz", allow_pickle=False) as data:
        clouds = {method: data["reference" if method == "ref" else f"seed88_{method}"].copy()
                  for method in methods}
    dimension = 2 if name == "kpp2d" else 3
    for method, points in clouds.items():
        if points.shape != (20000, dimension) or not np.isfinite(points).all():
            raise ValueError(f"Unexpected retained {name} density cloud: {method}")
    return clouds


def _histograms(clouds: dict[str, np.ndarray], pairs: tuple[tuple[int, int], ...]):
    edges = np.linspace(0, 2 * np.pi, 73)
    result = {}
    for method, points in clouds.items():
        panels = []
        for first, second in pairs:
            counts = np.histogram2d(points[:, first], points[:, second], bins=(edges, edges))[0]
            if int(counts.sum()) != len(points):
                raise ValueError(f"Density histogram loses particles: {method}, {first}, {second}")
            panels.append(counts / len(points) / np.diff(edges)[:, None] / np.diff(edges)[None, :])
        result[method] = panels
    return result


def _periodic_ticks(ax):
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([0, np.pi, 2 * np.pi], labels=["0", r"$\pi$", r"$2\pi$"])


def _densities(destination: Path) -> list[Path]:
    outputs = []
    for index, level, tag in ((7, 3.75, "boundary"), (8, 4., "extrapolation")):
        clouds = _clouds("kpp2d", index, ("mf", "mf_dp", "ref"))
        hist = _histograms(clouds, ((0, 1),))
        vmax = max(row[0].max() for row in hist.values())
        for method, row in hist.items():
            fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
            mesh = ax.imshow(row[0].T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
                             norm=Normalize(0, vmax), cmap="viridis", aspect="equal", interpolation="nearest")
            ax.set(xlabel="$x$", ylabel="$y$", title=LABELS[method] + f", $s={level:g}$")
            _periodic_ticks(ax)
            fig.colorbar(mesh, ax=ax, shrink=.85, pad=.025).ax.tick_params(labelsize=8)
            outputs.append(_save(fig, destination, DENSITY_DIR / f"dplus1_kpp2d_{tag}_{method}_20261016.pdf"))

    clouds = _clouds("kpp3d", 10, ("ref", "mf", "mf_dp"))
    pairs = ((0, 1), (0, 2), (1, 2))
    hist = _histograms(clouds, pairs)
    maxima = [max(row[column].max() for row in hist.values()) for column in range(3)]
    for method, row in hist.items():
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.7), layout="constrained")
        for column, (ax, (first, second), counts) in enumerate(zip(axes, pairs, row)):
            mesh = ax.imshow(counts.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
                             norm=Normalize(0, maxima[column]), cmap="viridis", aspect="auto", interpolation="nearest")
            ax.set(xlabel="$" + "xyz"[first] + "$", ylabel="$" + "xyz"[second] + "$")
            ax.tick_params(labelsize=10)
            _periodic_ticks(ax)
            ax.set_title(LABELS[method] if column == 1 else "", fontsize=12)
            fig.colorbar(mesh, ax=ax, pad=.025, shrink=.88).ax.tick_params(labelsize=8)
        outputs.append(_save(fig, destination, DENSITY_DIR / f"dplus1_kpp3d_projection_{method}_20261016.pdf"))
    return outputs


def _w2(destination: Path) -> list[Path]:
    outputs = []
    rows = _read(SUMMARY)
    for model in ("kpp2d", "kpp3d"):
        fig, ax = plt.subplots(figsize=(5.2, 3.6) if model == "kpp2d" else (5.2, 2.8))
        fig.subplots_adjust(left=.145, right=.985, bottom=.18 if model == "kpp2d" else .235, top=.975)
        for method in ("mf", "dp", "mf_dp"):
            selected = sorted((row for row in rows if row["model"] == model and row["method"] == method
                               and row["kind"] == "target_error"), key=lambda row: float(row["level"]))
            if len(selected) != 11 or any(int(row["particles"]) != 100000 or int(row["bins"]) != 16
                                          or int(row["seeds"]) != 3 for row in selected):
                raise ValueError(f"Unexpected current histogram protocol: {model}, {method}")
            x, mean, low, high = (np.array([float(row[key]) for row in selected])
                                  for key in ("level", "w2_mean", "w2_min", "w2_max"))
            if not (np.array_equal(x, np.arange(2., 4.51, .25)) and np.all(low <= mean)
                    and np.all(mean <= high) and low.min() > .25 and high.max() < .68):
                raise ValueError(f"Unexpected histogram summary: {model}, {method}")
            ax.plot(x, mean, color=COLORS[method], marker=MARKERS[method], ms=4, lw=1.6, label=LABELS[method])
            ax.fill_between(x, low, high, color=COLORS[method], alpha=.13, lw=0)
        ax.axvline(3.75, color="gray", ls=":", lw=1.3)
        ax.set(xlabel=r"$s=-\log_2\kappa$", ylabel=r"$W_2$", xlim=(1.9, 4.6), ylim=(.25, .68))
        ax.set_xticks(np.arange(2, 4.51, .5))
        ax.set_yticks(np.arange(.3, .651, .1))
        ax.grid(alpha=.23, ls="--")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
        ax.legend(frameon=False, loc="upper left", fontsize=8.5, borderaxespad=.65,
                  labelspacing=.35, handlelength=1.8)
        outputs.append(_save(fig, destination, W2_DIR / f"{model}_w2.pdf"))
    return outputs


def _fk(destination: Path) -> list[Path]:
    outputs = []
    for model in ("kpp2d", "kpp3d"):
        rows = [row for row in _read(SELECTED / model / "warm_starts/trace.csv")
                if row["model"] == model and float(row["level"]) == 4.5]
        compact = model == "kpp3d"
        if compact:
            fig, ax = plt.subplots(figsize=(5.2, 2.8))
            fig.subplots_adjust(left=.145, right=.985, bottom=.235, top=.975)
            ax.grid(alpha=.23, ls="--")
            ax.tick_params(labelsize=10)
            ax.set_axisbelow(True)
        else:
            fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        generations = sorted({int(row["generation"]) for row in rows})
        references = {float(row["reference_c"]) for row in rows}
        if generations != [1, 2, 4, 8, 16, 32, 64, 128] or len(references) != 1:
            raise ValueError(f"Unexpected retained FK traces: {model}")
        reference = references.pop()
        if compact and reference != 0.5761390835712005:
            raise ValueError("The compact 3D FK reference changed")
        for method in ("mf", "dp", "mf_dp", "cold"):
            selected = [row for row in rows if row["method"] == method]
            samples = [[float(row["c_hat"]) for row in selected if int(row["generation"]) == generation]
                       for generation in generations]
            if not all(len(values) == (3 if method == "cold" else 9) for values in samples):
                raise ValueError(f"Unexpected FK replicate count: {model}, {method}")
            values = np.array(samples)
            mean, low, high = values.mean(axis=1), values.min(axis=1), values.max(axis=1)
            if not np.isfinite(values).all() or (compact and (low.min() <= .34 or high.max() >= .655)):
                raise ValueError(f"Unexpected retained FK values: {model}, {method}")
            ax.plot(generations, mean, color=COLORS[method], marker=MARKERS[method], ms=3, lw=1.4,
                    label=LABELS[method])
            ax.fill_between(generations, low, high, color=COLORS[method], alpha=.12, lw=0)
        ax.axhline(reference, ls="--", color="#222222", lw=1, label="FK reference")
        ax.set_xscale("log", base=2)
        ax.set_xticks(generations, [str(value) for value in generations])
        ax.set(xlabel=r"FK generations $T$", ylabel=r"$\widehat c_T(2)$")
        if compact:
            ax.set_ylim(.34, .655)
            ax.set_yticks(np.arange(.35, .651, .05))
            ax.legend(frameon=False, ncol=2, fontsize=8.5, loc="lower right", bbox_to_anchor=(.99, .08),
                      borderaxespad=.3, labelspacing=.4, handlelength=1.8, columnspacing=1.2)
            relative = COMPACT_DIR / "kpp3d_fk.pdf"
        else:
            ax.set_title(r"$s=4.5$")
            ax.grid(alpha=.23, ls="--")
            ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
            relative = DENSITY_DIR / "dplus1_kpp2d_fk_20261016.pdf"
        outputs.append(_save(fig, destination, relative))
    return outputs


def render(destination: Path) -> list[Path]:
    """Write the 13 current KPP PDFs beneath a figure-root destination."""
    destination = Path(destination)
    with plt.rc_context(STYLE):
        outputs = _densities(destination) + _w2(destination) + _fk(destination)
    if len(outputs) != 13 or len(set(outputs)) != 13:
        raise RuntimeError("KPP reproduction did not produce the expected 13 PDFs")
    return outputs
