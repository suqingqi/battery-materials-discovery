from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "voltage_shap_importance.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "voltage_shap_top_features.png"
)


df = pd.read_csv(INPUT_PATH)

print("Input shape:", df.shape)
print("Columns:", list(df.columns))
print()

required_cols = [
    "feature",
    "mean_abs_shap",
]

for col in required_cols:
    if col not in df.columns:
        raise RuntimeError(
            f"Missing required column: {col}"
        )

top_n = 12

plot_df = (
    df.sort_values(
        by="mean_abs_shap",
        ascending=False,
    )
    .head(top_n)
    .copy()
)

plot_df = plot_df.iloc[::-1]

print("Top features:")
print(
    plot_df[
        ["feature", "mean_abs_shap"]
    ]
)
print()


fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.barh(
    plot_df["feature"],
    plot_df["mean_abs_shap"],
)

ax.set_xlabel(
    "Mean |SHAP value|"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "Top SHAP Features for Voltage Prediction"
)

for i, value in enumerate(
    plot_df["mean_abs_shap"]
):
    ax.text(
        value,
        i,
        f" {value:.3f}",
        va="center",
        fontsize=9,
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved: {OUTPUT_PATH}")