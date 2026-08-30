from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_application.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_csv(DATA_FILE)


print("Dataset shape:", df.shape)


targets = [
    "average_voltage",
    "capacity_grav",
    "max_delta_volume",
    "stability_charge",
    "stability_discharge",
]


print()
print("Target statistics:")

print(
    df[targets]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .T
)


print()
print("Target correlations:")

correlations = (
    df[targets]
    .corr()
)

print(correlations)


for column in targets:

    plt.figure(figsize=(7, 5))

    plt.hist(
        df[column],
        bins=40,
        edgecolor="black",
    )

    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(
        f"Distribution of {column}"
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / f"{column}_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


plt.figure(figsize=(7, 5))

plt.scatter(
    df["capacity_grav"],
    df["average_voltage"],
    alpha=0.5,
)

plt.xlabel(
    "Gravimetric Capacity (mAh/g)"
)

plt.ylabel(
    "Average Voltage (V)"
)

plt.title(
    "Voltage vs Gravimetric Capacity"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "voltage_vs_capacity.png",
    dpi=300,
)

plt.close()


plt.figure(figsize=(7, 5))

plt.scatter(
    df["max_delta_volume"] * 100,
    df["capacity_grav"],
    alpha=0.5,
)

plt.xlabel(
    "Maximum Volume Change (%)"
)

plt.ylabel(
    "Gravimetric Capacity (mAh/g)"
)

plt.title(
    "Capacity vs Volume Change"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "capacity_vs_volume_change.png",
    dpi=300,
)

plt.close()


main_metals = [
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
]


def parse_elements(element_string):

    if pd.isna(element_string):
        return set()

    return {
        item.strip()
        for item in element_string.split(",")
        if item.strip()
    }


metal_rows = []


for metal in main_metals:

    mask = (
        df["elements"]
        .apply(
            lambda value:
            metal in parse_elements(value)
        )
    )

    subset = df.loc[mask]

    if subset.empty:
        continue

    metal_rows.append(
        {
            "metal": metal,
            "count": len(subset),
            "mean_voltage":
                subset[
                    "average_voltage"
                ].mean(),
            "mean_capacity":
                subset[
                    "capacity_grav"
                ].mean(),
            "mean_volume_change":
                subset[
                    "max_delta_volume"
                ].mean(),
            "median_stability_charge":
                subset[
                    "stability_charge"
                ].median(),
            "median_stability_discharge":
                subset[
                    "stability_discharge"
                ].median(),
        }
    )


metal_summary = pd.DataFrame(
    metal_rows
)


print()
print("Transition-metal summary:")

print(
    metal_summary
    .sort_values(
        "mean_voltage",
        ascending=False,
    )
)


metal_summary.to_csv(
    PROJECT_ROOT
    / "results"
    / "metal_summary.csv",
    index=False,
)


stability_limits = [
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
]


print()
print(
    "Stability threshold analysis:"
)


for limit in stability_limits:

    stable_mask = (
        (df["stability_charge"] <= limit)
        &
        (
            df["stability_discharge"]
            <= limit
        )
    )

    count = int(
        stable_mask.sum()
    )

    percentage = (
        count
        / len(df)
        * 100
    )

    print(
        f"<= {limit:.2f} eV/atom:"
        f" {count} "
        f"({percentage:.1f}%)"
    )


high_stability_outliers = df[
    (
        df["stability_charge"]
        > 1.0
    )
    |
    (
        df["stability_discharge"]
        > 1.0
    )
].copy()


print()
print(
    "Samples with stability > 1 eV/atom:"
)

print(
    len(high_stability_outliers)
)


print()
print("Top stability outliers:")

print(
    high_stability_outliers[
        [
            "battery_id",
            "battery_formula",
            "average_voltage",
            "capacity_grav",
            "stability_charge",
            "stability_discharge",
        ]
    ]
    .sort_values(
        [
            "stability_charge",
            "stability_discharge",
        ],
        ascending=False,
    )
    .head(20)
)


print()
print(
    "EDA figures saved to:"
)

print(FIGURE_DIR)

print()
print("STEP 04 completed.")