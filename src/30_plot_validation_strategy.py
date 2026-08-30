from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


methods = [
    "Random 5-fold CV",
    "Framework GroupKFold",
]

r2_values = [
    0.5949,
    0.4935,
]

mae_values = [
    0.3667,
    0.4185,
]


x = np.arange(len(methods))

fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)

bars = ax.bar(
    x,
    r2_values,
    width=0.55,
)

for bar, value in zip(
    bars,
    r2_values,
):
    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.015,
        f"R² = {value:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
    )


ax.set_xticks(x)
ax.set_xticklabels(methods)

ax.set_ylabel(
    "Cross-validated R²"
)

ax.set_ylim(
    0,
    0.70,
)

ax.set_title(
    "Random Split Overestimates "
    "Framework-level Generalization"
)

ax.text(
    0,
    0.08,
    f"MAE = {mae_values[0]:.3f} V",
    ha="center",
    fontsize=10,
)

ax.text(
    1,
    0.08,
    f"MAE = {mae_values[1]:.3f} V",
    ha="center",
    fontsize=10,
)

gap = (
    r2_values[0]
    - r2_values[1]
)

ax.annotate(
    f"Generalization gap\nΔR² = {gap:.3f}",
    xy=(1, r2_values[1]),
    xytext=(0.5, 0.66),
    ha="center",
    arrowprops={
        "arrowstyle": "->",
    },
)

fig.tight_layout()

output_path = (
    OUTPUT_DIR
    / "random_vs_groupcv.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Random CV R²: "
    f"{r2_values[0]:.4f}"
)

print(
    f"Framework GroupKFold R²: "
    f"{r2_values[1]:.4f}"
)

print(
    f"Generalization gap: "
    f"{gap:.4f}"
)

print()

print(
    f"Saved: {output_path}"
)