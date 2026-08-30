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
    "Composition",
    "Reaction",
    "Composition + Reaction",
]

r2_values = [
    0.4935,
    0.1026,
    0.5302,
]

mae_values = [
    0.4185,
    0.5990,
    0.3966,
]


x = np.arange(len(methods))

fig, ax = plt.subplots(
    figsize=(8.0, 5.3)
)

bars = ax.bar(
    x,
    r2_values,
    width=0.58,
)

for bar, r2, mae in zip(
    bars,
    r2_values,
    mae_values,
):
    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        r2 + 0.015,
        f"R² = {r2:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        0.035,
        f"MAE = {mae:.3f} V",
        ha="center",
        va="bottom",
        fontsize=9,
    )


ax.set_xticks(x)

ax.set_xticklabels(
    methods
)

ax.set_ylabel(
    "Framework GroupKFold R²"
)

ax.set_ylim(
    0,
    0.62,
)

ax.set_title(
    "Reaction-aware Features Improve Voltage Prediction"
)


delta_r2 = (
    r2_values[2]
    - r2_values[0]
)

delta_mae = (
    mae_values[0]
    - mae_values[2]
)


ax.annotate(
    (
        f"ΔR² = +{delta_r2:.3f}\n"
        f"MAE improves by {delta_mae:.3f} V"
    ),
    xy=(
        2,
        r2_values[2],
    ),
    xytext=(
        1.35,
        0.585,
    ),
    ha="center",
    arrowprops={
        "arrowstyle": "->",
    },
)


fig.tight_layout()


output_path = (
    OUTPUT_DIR
    / "voltage_feature_ablation.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print(
    f"Composition R²: "
    f"{r2_values[0]:.4f}"
)

print(
    f"Reaction R²: "
    f"{r2_values[1]:.4f}"
)

print(
    f"Composition + Reaction R²: "
    f"{r2_values[2]:.4f}"
)

print(
    f"R² improvement: "
    f"{delta_r2:+.4f}"
)

print(
    f"MAE improvement: "
    f"{delta_mae:.4f} V"
)

print()

print(
    f"Saved: {output_path}"
)