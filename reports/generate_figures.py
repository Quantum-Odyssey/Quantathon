#!/usr/bin/env python3
"""Generate report figures from validated repository results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#102A43"
BLUE = "#167D9A"
GOLD = "#E6A23C"
GRID = "#D9E2EC"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.edgecolor": NAVY,
    "axes.titleweight": "bold",
    "figure.dpi": 180,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


labels = ["Exactitud", "Precisión", "Recall", "F1", "Ex. bal."]
baseline_mean = np.array([0.6482, 0.5477, 0.5652, 0.5558, 0.6332])
baseline_sd = np.array([0.0180, 0.0227, 0.0437, 0.0288, 0.0205])

fig, ax = plt.subplots(figsize=(6.6, 2.45))
x = np.arange(len(labels))
ax.bar(x, baseline_mean, yerr=baseline_sd, capsize=4, color=BLUE, width=.58)
ax.set_xticks(x, labels)
ax.set_ylim(0, 0.78)
ax.set_ylabel("Puntuación")
ax.set_title("SVM-RBF: media ± desviación estándar en 10 divisiones estratificadas")
ax.grid(axis="y", color=GRID, linewidth=.6)
ax.set_axisbelow(True)
for i, value in enumerate(baseline_mean):
    ax.text(i, value + baseline_sd[i] + .025, f"{value:.3f}", ha="center", color=NAVY, fontsize=7)
save(fig, "baseline_metrics.pdf")


classical = np.array([.50, 1.00, .25, .40, .625])
quantum = np.array([.667, .75, .75, .75, .625])
fig, ax = plt.subplots(figsize=(6.6, 2.45))
w = .34
ax.bar(x - w/2, classical, width=w, color=BLUE, label="SVM-RBF")
ax.bar(x + w/2, quantum, width=w, color=GOLD, label="QSVM")
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Puntuación")
ax.set_title("Comparación directa en la misma instancia: 24 train / 6 test")
ax.legend(frameon=False, ncol=2, loc="upper right")
ax.grid(axis="y", color=GRID, linewidth=.6)
ax.set_axisbelow(True)
save(fig, "same_instance_metrics.pdf")


fig, axes = plt.subplots(1, 2, figsize=(5.3, 2.15))
for ax, cm, title, cmap in [
    (axes[0], np.array([[2, 0], [3, 1]]), "SVM-RBF", "Blues"),
    (axes[1], np.array([[1, 1], [1, 3]]), "QSVM", "YlOrBr"),
]:
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, square=True, cmap=cmap,
                vmin=0, vmax=3, ax=ax, linewidths=.5, linecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
save(fig, "confusion_matrices.pdf")


kernel_path = ROOT / "data" / "kernel" / "n_24_dim_2_z_feature_map.csv"
kernel = pd.read_csv(kernel_path, index_col=0).to_numpy()
fig, ax = plt.subplots(figsize=(3.25, 2.75))
sns.heatmap(kernel, vmin=0, vmax=1, cmap="mako", square=True, ax=ax,
            xticklabels=4, yticklabels=4, cbar_kws={"label": r"$K_{ij}$", "shrink": .75})
ax.set_title("Kernel cuántico Z (24 × 24)")
ax.set_xlabel("Muestra j")
ax.set_ylabel("Muestra i")
save(fig, "quantum_kernel.pdf")

print(OUT)
