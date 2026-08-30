from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "li_insertion_electrodes_raw.csv"
)

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_csv(RAW_FILE)


print("Raw dataset shape:", df.shape)

print()
print("Working ion counts:")
print(df["working_ion"].value_counts(dropna=False))


target_columns = [
    "average_voltage",
    "capacity_grav",
    "max_delta_volume",
    "stability_charge",
    "stability_discharge",
]


print()
print("Target statistics:")
print(df[target_columns].describe().T)


print()
print("Duplicate battery IDs:")
print(df["battery_id"].duplicated().sum())


print()
print("Duplicate framework formulas:")
print(df["framework_formula"].duplicated().sum())


print()
print("Number of elements distribution:")
print(
    df["nelements"]
    .value_counts()
    .sort_index()
)


df["element_list"] = (
    df["elements"]
    .fillna("")
    .str.split(",")
)


element_counter = {}


for element_list in df["element_list"]:

    for element in element_list:

        element = element.strip()

        if not element:
            continue

        element_counter[element] = (
            element_counter.get(element, 0) + 1
        )


element_counts = (
    pd.Series(element_counter)
    .sort_values(ascending=False)
)


print()
print("Top 30 elements:")
print(element_counts.head(30))


transition_metals = {
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
}


def contains_transition_metal(element_list):

    return any(
        element.strip() in transition_metals
        for element in element_list
    )


df["has_transition_metal"] = (
    df["element_list"]
    .apply(contains_transition_metal)
)


print()
print("Transition metal statistics:")
print(
    df["has_transition_metal"]
    .value_counts()
)


anion_groups = {
    "O": "oxide",
    "S": "sulfide",
    "F": "fluoride",
    "P": "phosphate_possible",
    "Si": "silicate_possible",
}


for element, column_name in anion_groups.items():

    df[column_name] = (
        df["element_list"]
        .apply(
            lambda elements:
            element in [
                e.strip()
                for e in elements
            ]
        )
    )


print()
print("Major chemistry families:")

for column in anion_groups.values():

    print(
        column,
        int(df[column].sum()),
    )


warning_count = (
    df["warnings"]
    .notna()
    .sum()
)


print()
print("Samples containing warnings:")
print(warning_count)


if warning_count > 0:

    print()
    print("Most common warnings:")

    print(
        df.loc[
            df["warnings"].notna(),
            "warnings",
        ]
        .value_counts()
        .head(15)
    )


basic_physical_mask = (
    df["average_voltage"]
    .between(
        0.5,
        6.0,
        inclusive="both",
    )
    &
    df["capacity_grav"]
    .between(
        20,
        500,
        inclusive="both",
    )
    &
    df["max_delta_volume"]
    .between(
        0,
        1.0,
        inclusive="both",
    )
)


print()
print("Basic physical-range filter:")
print("Pass:", int(basic_physical_mask.sum()))
print(
    "Removed:",
    int((~basic_physical_mask).sum()),
)


transition_mask = (
    df["has_transition_metal"]
)


battery_domain_mask = (
    basic_physical_mask
    &
    transition_mask
)


print()
print("Physical + transition-metal filter:")
print(
    "Pass:",
    int(battery_domain_mask.sum()),
)
print(
    "Removed:",
    int((~battery_domain_mask).sum()),
)


analysis_file = (
    INTERIM_DIR
    / "li_battery_domain_analysis.csv"
)


df.to_csv(
    analysis_file,
    index=False,
)


print()
print("Saved analysis dataset:")
print(analysis_file)

print()
print("STEP 02 analysis completed.")