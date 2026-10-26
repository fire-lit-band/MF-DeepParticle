"""Render the current cubic appendix figure from its saved per-seed results."""
from pathlib import Path
import csv
import json
import statistics


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output/non_kpp_cubic_ood_20261023"
RELATIVE_OUTPUT = Path("non_kpp_cubic_ood_upto3_20261023/cubic_total2k_normalized_w2.pdf")
METHODS = ("mf", "dp", "mf_dp")
LEVELS = (.25, .375, .5, .75, 1., 1.5, 2., 2.5, 3.)
TRAIN = (.25, .5, 1., 2.)


def _values():
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    with (DATA / "metrics_per_seed.csv").open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    rows = [r for r in summary if r["method"] in METHODS and r["D"] in LEVELS]
    expected = {(d, method) for d in LEVELS for method in METHODS}
    if len(rows) != len(expected) or {(r["D"], r["method"]) for r in rows} != expected:
        raise ValueError("The cubic figure requires all nine manuscript levels and three methods")
    for row in rows:
        group = [r for r in raw if float(r["level"]) == row["D"] and r["method"] == row["method"]]
        if len(group) != 3 or {int(r["seed"]) for r in group} != {88, 89, 90}:
            raise ValueError("Each cubic point requires the three original training seeds")
        values = [float(r["w2_normalized"]) for r in group]
        for field, value in (("mean", statistics.mean(values)), ("minimum", min(values)), ("maximum", max(values))):
            if abs(float(row[field]) - value) > 1e-12:
                raise ValueError(f"Cubic summary does not match per-seed results: D={row['D']}, {row['method']}, {field}")
    return rows


def render(destination: Path) -> list[Path]:
    """Write the manuscript PDF beneath destination, without changing source data."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = _values()
    colors = {"mf": "#2864B4", "dp": "#D37B2F", "mf_dp": "#168173"}
    markers = {"mf": "o", "dp": "^", "mf_dp": "s"}
    output = Path(destination) / RELATIVE_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
             "axes.linewidth": .8, "axes.labelsize": 12}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.4, 4.1))
        try:
            fig.subplots_adjust(left=.10, right=.98, bottom=.15, top=.81)
            for method in METHODS:
                group = sorted((r for r in rows if r["method"] == method), key=lambda r: r["D"])
                x = np.array([r["D"] for r in group])
                mean = np.array([r["mean"] for r in group])
                low = np.array([r["minimum"] for r in group])
                high = np.array([r["maximum"] for r in group])
                ax.fill_between(x, low, high, color=colors[method], alpha=.13, lw=0)
                ax.plot(x, mean, color=colors[method], lw=2)
                for heldout in (False, True):
                    mask = np.array([(d not in TRAIN) == heldout for d in x])
                    ax.plot(x[mask], mean[mask], ls="none", marker=markers[method], ms=5.5,
                            markerfacecolor="white" if heldout else colors[method],
                            markeredgecolor=colors[method], markeredgewidth=1.3)
            ax.axvline(2., color="#777777", ls=":", lw=1.3)
            if max(r["maximum"] for r in rows) >= 1.2:
                raise ValueError("Cubic data exceed the manuscript figure axis")
            ax.set(xlabel=r"Diffusion coefficient $D$", ylabel=r"Normalized $W_2$",
                   xlim=(.15, 3.08), ylim=(0, 1.2))
            ax.set_xticks([.25, .5, .75, 1., 1.5, 2., 2.5, 3.],
                          ["0.25", "0.5", "0.75", "1", "1.5", "2", "2.5", "3"], fontsize=9)
            ax.set_yticks(np.arange(0, 1.21, .2))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E5E7EB", lw=.8)
            ax.set_axisbelow(True)
            labels = {"mf": "MF", "dp": "DP", "mf_dp": "DP refinement"}
            handles = [Line2D([], [], color=colors[m], marker=markers[m], lw=2, label=labels[m]) for m in METHODS]
            fig.suptitle("Non-KPP Cubic", y=.98, fontsize=13, fontweight="bold")
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .915), ncol=3,
                       frameon=False, fontsize=9, columnspacing=2., handlelength=2.6)
            fig.savefig(output, dpi=250, facecolor="white")
        finally:
            plt.close(fig)
    return [output]
