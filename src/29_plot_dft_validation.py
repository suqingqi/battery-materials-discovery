from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "dft_voltage_validation.csv"
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


if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_PATH}\n"
        "Run 28_dft_voltage_validation.py first."
    )


df = pd.read_csv(INPUT_PATH)


method_order = [
    "ML-OOF",
    "DFT-PBE",
    "Materials Project",
]

df = (
    df
    .set_index("method")
    .loc[method_order]
    .reset_index()
)


fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)


bars = ax.bar(
    df["method"],
    df["voltage_V"],
    width=0.62,
)


for bar, voltage in zip(
    bars,
    df["voltage_V"],
):
    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        voltage + 0.06,
        f"{voltage:.2f} V",
        ha="center",
        va="bottom",
        fontsize=11,
    )


mp_voltage = df.loc[
    df["method"]
    == "Materials Project",
    "voltage_V",
].iloc[0]


ax.axhline(
    y=mp_voltage,
    linestyle="--",
    linewidth=1.2,
    label=(
        f"MP reference = "
        f"{mp_voltage:.2f} V"
    ),
)


ax.set_ylabel(
    "Average Li Insertion Voltage (V)"
)

ax.set_title(
    "CoPO$_4$ to LiCoPO$_4$: "
    "Independent Voltage Validation"
)

ax.set_ylim(
    0,
    df["voltage_V"].max() + 0.8,
)

ax.legend(
    frameon=False
)


ax.text(
    0.5,
    -0.18,
    (
        "DFT-PBE is an independent "
        "first-principles check; "
        "it is not a reproduction of the "
        "Materials Project GGA/GGA+U workflow."
    ),
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=9,
)


fig.tight_layout()


output_path = (
    OUTPUT_DIR
    / "dft_voltage_validation.png"
)


fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print(df)

print()

print(
    f"Saved: "
    f"{output_path}"
)