from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "li_battery_domain_analysis.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_csv(INPUT_FILE)


print("Input shape:", df.shape)


target_columns = [
    "average_voltage",
    "capacity_grav",
    "max_delta_volume",
    "stability_charge",
    "stability_discharge",
]


df = df.dropna(
    subset=target_columns
).copy()


broad_mask = (
    df["average_voltage"].between(
        0.5,
        6.0,
        inclusive="both",
    )
    &
    df["capacity_grav"].between(
        20,
        500,
        inclusive="both",
    )
    &
    df["max_delta_volume"].between(
        0,
        1.0,
        inclusive="both",
    )
)


broad_df = (
    df.loc[broad_mask]
    .copy()
    .reset_index(drop=True)
)


print()
print("Broad dataset shape:", broad_df.shape)


main_transition_metals = {
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
}


def parse_elements(element_string):

    if pd.isna(element_string):
        return set()

    return {
        item.strip()
        for item in element_string.split(",")
        if item.strip()
    }


def contains_main_transition_metal(element_string):

    elements = parse_elements(element_string)

    return len(
        elements.intersection(main_transition_metals)
    ) > 0


broad_df["has_main_transition_metal"] = (
    broad_df["elements"]
    .apply(contains_main_transition_metal)
)


allowed_chemistry_mask = (
    broad_df["oxide"]
    |
    broad_df["phosphate_possible"]
    |
    broad_df["fluoride"]
    |
    broad_df["sulfide"]
)


no_transition_warning_mask = (
    broad_df["warnings"]
    .fillna("")
    .ne("Transition metal not found")
)


application_mask = (
    broad_df["has_main_transition_metal"]
    &
    allowed_chemistry_mask
    &
    no_transition_warning_mask
    &
    broad_df["average_voltage"].between(
        2.0,
        5.5,
        inclusive="both",
    )
    &
    broad_df["capacity_grav"].between(
        50,
        350,
        inclusive="both",
    )
    &
    broad_df["max_delta_volume"].between(
        0,
        0.30,
        inclusive="both",
    )
)


application_df = (
    broad_df.loc[application_mask]
    .copy()
    .reset_index(drop=True)
)


print()
print(
    "Application dataset shape:",
    application_df.shape,
)


print()
print("Application target statistics:")

print(
    application_df[
        [
            "average_voltage",
            "capacity_grav",
            "max_delta_volume",
            "stability_charge",
            "stability_discharge",
        ]
    ].describe().T
)


print()
print("Top elements in application dataset:")


element_counter = {}


for element_string in application_df["elements"]:

    for element in parse_elements(element_string):

        element_counter[element] = (
            element_counter.get(element, 0) + 1
        )


element_counts = (
    pd.Series(element_counter)
    .sort_values(ascending=False)
)


print(element_counts.head(25))


print()
print("Main transition-metal counts:")


for metal in sorted(main_transition_metals):

    count = application_df["elements"].apply(
        lambda x: metal in parse_elements(x)
    ).sum()

    print(
        metal,
        int(count),
    )


print()
print("Chemistry families:")

for column in [
    "oxide",
    "phosphate_possible",
    "fluoride",
    "sulfide",
]:

    print(
        column,
        int(application_df[column].sum()),
    )


print()
print("Warnings in application dataset:")

print(
    application_df["warnings"]
    .fillna("No warning")
    .value_counts()
)


broad_file = (
    PROCESSED_DIR
    / "li_insertion_broad.csv"
)

application_file = (
    PROCESSED_DIR
    / "li_cathode_application.csv"
)


broad_df.to_csv(
    broad_file,
    index=False,
)

application_df.to_csv(
    application_file,
    index=False,
)


print()
print("Saved broad dataset:")
print(broad_file)

print()
print("Saved application dataset:")
print(application_file)

print()
print("STEP 03 completed.")