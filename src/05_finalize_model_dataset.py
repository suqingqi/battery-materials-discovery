from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_application.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_model.csv"
)


df = pd.read_csv(INPUT_FILE)


print("Input shape:", df.shape)


df["max_stability"] = df[
    [
        "stability_charge",
        "stability_discharge",
    ]
].max(axis=1)


extreme_stability_mask = (
    df["max_stability"] > 1.0
)


print()
print(
    "Extreme stability samples removed:",
    int(extreme_stability_mask.sum()),
)


model_df = (
    df.loc[~extreme_stability_mask]
    .copy()
    .reset_index(drop=True)
)


print(
    "Final model dataset shape:",
    model_df.shape,
)


print()
print("Target statistics:")

print(
    model_df[
        [
            "average_voltage",
            "capacity_grav",
            "max_delta_volume",
            "max_stability",
        ]
    ].describe().T
)


model_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("STEP 05 completed.")