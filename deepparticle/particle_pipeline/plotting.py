"""Standalone figures computed from this run; never paste manuscript result tables."""
from __future__ import annotations
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"mf": "MF (full budget)", "mf_init": "MF initializer", "dp": "Direct DP",
          "mf_dp": "MF+DP", "cold": "Uniform cold start"}


def save(fig, path):
    # Decimal parameter levels are part of the name, not a file suffix.
    fig.savefig(path.parent / (path.name + ".png"), dpi=160)
    fig.savefig(path.parent / (path.name + ".pdf"))
    plt.close(fig)


def plot_results(output, protocol, rows, fk_rows, purpose="experiment"):
    for metric in sorted({r["metric"] for r in rows}):
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for method in ("mf", "mf_init", "dp", "mf_dp"):
            items = [r for r in rows if r["metric"] == metric and r["method"] == method]
            levels = sorted({r["level"] for r in items})
            mean, sd = [], []
            for level in levels:
                values = [r["w2"] for r in items if r["level"] == level]
                mean.append(statistics.mean(values))
                sd.append(statistics.stdev(values) if len(values) > 1 else 0.)
            ax.errorbar(levels, mean, yerr=sd, marker="o", capsize=2, label=LABELS[method])
        ax.set(xlabel="s = -log2(kappa)" if protocol.periodic else "Advection amplitude sigma",
               ylabel="W2", title=f"{protocol.name}: {metric}; {purpose}; N={protocol.n_eval}")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        save(fig, output / f"w2_{metric}")
    for level in sorted({r["level"] for r in fk_rows}):
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        items = [r for r in fk_rows if r["level"] == level]
        for method in ("cold", "mf", "mf_init", "dp", "mf_dp"):
            selected = [r for r in items if r["method"] == method]
            generations = sorted({r["generation"] for r in selected})
            means = [statistics.mean(r["c_hat"] for r in selected if r["generation"] == g) for g in generations]
            ax.plot(generations, means, "o-", label=LABELS[method])
        ax.axhline(items[0]["reference_c"], color="grey", ls="--", label="Independent finite-time FK reference")
        ax.set(xscale="log", xlabel="FK generations", ylabel="mu(lambda) / lambda",
               title=f"{protocol.name}: s={level:g}; {purpose}; FK reference T={protocol.generations}")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        save(fig, output / f"fk_level_{level:g}")
