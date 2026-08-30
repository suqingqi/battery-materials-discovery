from pathlib import Path

import numpy as np
import pandas as pd

from pymatgen.core import Composition


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_model.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_reaction_features.csv"
)


TOLERANCE = 1e-6


def safe_composition(
    formula,
):

    try:

        return Composition(
            formula
        )

    except Exception:

        return None


def get_element_amount(
    composition,
    element,
):

    if composition is None:

        return np.nan

    try:

        return float(
            composition[
                element
            ]
        )

    except Exception:

        return 0.0


def remove_li(
    composition,
):

    if composition is None:

        return None

    host_dict = {}

    for element, amount in (
        composition.get_el_amt_dict().items()
    ):

        if element == "Li":
            continue

        host_dict[
            element
        ] = float(
            amount
        )

    if len(host_dict) == 0:

        return None

    return Composition(
        host_dict
    )


def get_total_atoms(
    composition,
):

    if composition is None:

        return np.nan

    return float(
        composition.num_atoms
    )


def calculate_host_scale(
    charge_host,
    discharge_host,
):

    if (
        charge_host is None
        or discharge_host is None
    ):

        return np.nan, False

    charge_dict = (
        charge_host
        .get_el_amt_dict()
    )

    discharge_dict = (
        discharge_host
        .get_el_amt_dict()
    )

    charge_elements = set(
        charge_dict.keys()
    )

    discharge_elements = set(
        discharge_dict.keys()
    )

    if (
        charge_elements
        != discharge_elements
    ):

        return np.nan, False

    ratios = []

    for element in charge_elements:

        charge_amount = float(
            charge_dict[
                element
            ]
        )

        discharge_amount = float(
            discharge_dict[
                element
            ]
        )

        if (
            charge_amount <= 0
            or discharge_amount <= 0
        ):

            return np.nan, False

        ratio = (
            discharge_amount
            /
            charge_amount
        )

        ratios.append(
            ratio
        )

    ratios = np.array(
        ratios,
        dtype=float,
    )

    mean_ratio = float(
        np.mean(
            ratios
        )
    )

    if mean_ratio <= 0:

        return np.nan, False

    relative_deviation = np.max(
        np.abs(
            ratios
            -
            mean_ratio
        )
        /
        mean_ratio
    )

    framework_match = (
        relative_deviation
        < 1e-4
    )

    return (
        mean_ratio,
        framework_match,
    )


def extract_reaction_features(
    row,
):

    charge_comp = safe_composition(
        row[
            "formula_charge"
        ]
    )

    discharge_comp = safe_composition(
        row[
            "formula_discharge"
        ]
    )

    if (
        charge_comp is None
        or discharge_comp is None
    ):

        return pd.Series(
            {
                "li_charge_raw":
                    np.nan,
                "li_discharge_raw":
                    np.nan,
                "delta_li_raw":
                    np.nan,
                "host_scale":
                    np.nan,
                "framework_match":
                    False,
                "li_discharge_normalized":
                    np.nan,
                "normalized_delta_li":
                    np.nan,
                "abs_normalized_delta_li":
                    np.nan,
                "relative_delta_li":
                    np.nan,
            }
        )

    li_charge = get_element_amount(
        charge_comp,
        "Li",
    )

    li_discharge = get_element_amount(
        discharge_comp,
        "Li",
    )

    delta_li_raw = (
        li_discharge
        -
        li_charge
    )

    charge_host = remove_li(
        charge_comp
    )

    discharge_host = remove_li(
        discharge_comp
    )

    (
        host_scale,
        framework_match,
    ) = calculate_host_scale(
        charge_host,
        discharge_host,
    )

    if (
        framework_match
        and
        not np.isnan(
            host_scale
        )
    ):

        li_discharge_normalized = (
            li_discharge
            /
            host_scale
        )

        normalized_delta_li = (
            li_discharge_normalized
            -
            li_charge
        )

    else:

        li_discharge_normalized = (
            np.nan
        )

        normalized_delta_li = (
            np.nan
        )

    abs_normalized_delta_li = (
        abs(
            normalized_delta_li
        )
        if not np.isnan(
            normalized_delta_li
        )
        else np.nan
    )

    charge_host_atoms = (
        get_total_atoms(
            charge_host
        )
    )

    if (
        not np.isnan(
            normalized_delta_li
        )
        and
        charge_host_atoms > 0
    ):

        relative_delta_li = (
            abs_normalized_delta_li
            /
            charge_host_atoms
        )

    else:

        relative_delta_li = (
            np.nan
        )

    return pd.Series(
        {
            "li_charge_raw":
                li_charge,
            "li_discharge_raw":
                li_discharge,
            "delta_li_raw":
                delta_li_raw,
            "host_scale":
                host_scale,
            "framework_match":
                framework_match,
            "li_discharge_normalized":
                li_discharge_normalized,
            "normalized_delta_li":
                normalized_delta_li,
            "abs_normalized_delta_li":
                abs_normalized_delta_li,
            "relative_delta_li":
                relative_delta_li,
        }
    )


def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Dataset shape:",
        df.shape,
    )

    print()

    print(
        "Formula examples:"
    )

    print(
        df[
            [
                "formula_charge",
                "formula_discharge",
            ]
        ]
        .head(
            10
        )
    )

    reaction_features = (
        df.apply(
            extract_reaction_features,
            axis=1,
        )
    )

    df = pd.concat(
        [
            df,
            reaction_features,
        ],
        axis=1,
    )

    print()
    print(
        "Framework normalization summary:"
    )

    print(
        df[
            "framework_match"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()
    print(
        "Missing normalized reaction features:"
    )

    print(
        df[
            [
                "normalized_delta_li",
                "abs_normalized_delta_li",
                "relative_delta_li",
            ]
        ]
        .isna()
        .sum()
    )

    print()
    print(
        "Raw delta Li statistics:"
    )

    print(
        df[
            [
                "li_charge_raw",
                "li_discharge_raw",
                "delta_li_raw",
            ]
        ]
        .describe()
    )

    print()
    print(
        "Normalized Li reaction statistics:"
    )

    print(
        df[
            [
                "normalized_delta_li",
                "abs_normalized_delta_li",
                "relative_delta_li",
            ]
        ]
        .describe()
    )

    positive_count = (
        df[
            "normalized_delta_li"
        ]
        > 0
    ).sum()

    zero_count = (
        np.isclose(
            df[
                "normalized_delta_li"
            ],
            0.0,
            atol=TOLERANCE,
        )
    ).sum()

    negative_count = (
        df[
            "normalized_delta_li"
        ]
        < 0
    ).sum()

    print()
    print(
        "Normalized delta Li direction:"
    )

    print(
        "Positive:",
        positive_count,
    )

    print(
        "Zero:",
        zero_count,
    )

    print(
        "Negative:",
        negative_count,
    )

    print()
    print(
        "Largest raw |delta Li| samples:"
    )

    largest_raw = (
        df.assign(
            abs_delta_li_raw=(
                df[
                    "delta_li_raw"
                ]
                .abs()
            )
        )
        [
            [
                "battery_id",
                "framework_formula",
                "formula_charge",
                "formula_discharge",
                "li_charge_raw",
                "li_discharge_raw",
                "delta_li_raw",
                "host_scale",
                "framework_match",
                "normalized_delta_li",
                "capacity_grav",
                "average_voltage",
            ]
        ]
        .sort_values(
            "delta_li_raw",
            key=lambda s: s.abs(),
            ascending=False,
        )
        .head(
            15
        )
    )

    print(
        largest_raw
    )

    print()
    print(
        "Largest normalized |delta Li| samples:"
    )

    largest_normalized = (
        df[
            [
                "battery_id",
                "framework_formula",
                "formula_charge",
                "formula_discharge",
                "li_charge_raw",
                "li_discharge_raw",
                "host_scale",
                "normalized_delta_li",
                "abs_normalized_delta_li",
                "relative_delta_li",
                "capacity_grav",
                "average_voltage",
            ]
        ]
        .sort_values(
            "abs_normalized_delta_li",
            ascending=False,
        )
        .head(
            15
        )
    )

    print(
        largest_normalized
    )

    analysis_columns = [
        "average_voltage",
        "capacity_grav",
        "max_delta_volume",
        "max_stability",
        "num_steps",
        "normalized_delta_li",
        "abs_normalized_delta_li",
        "relative_delta_li",
    ]

    correlation_df = (
        df[
            analysis_columns
        ]
        .corr(
            numeric_only=True
        )
    )

    print()
    print(
        "Correlation with average_voltage:"
    )

    print(
        correlation_df[
            "average_voltage"
        ]
        .sort_values(
            ascending=False
        )
    )

    print()
    print(
        "Correlation with capacity_grav:"
    )

    print(
        correlation_df[
            "capacity_grav"
        ]
        .sort_values(
            ascending=False
        )
    )

    print()
    print(
        "Correlation with max_delta_volume:"
    )

    print(
        correlation_df[
            "max_delta_volume"
        ]
        .sort_values(
            ascending=False
        )
    )

    print()
    print(
        "Grouped by num_steps:"
    )

    step_summary = (
        df
        .groupby(
            "num_steps"
        )
        .agg(
            count=(
                "battery_id",
                "count",
            ),
            mean_voltage=(
                "average_voltage",
                "mean",
            ),
            mean_capacity=(
                "capacity_grav",
                "mean",
            ),
            mean_delta_li=(
                "abs_normalized_delta_li",
                "mean",
            ),
            mean_volume_change=(
                "max_delta_volume",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        step_summary
    )

    valid_mask = (
        df[
            [
                "normalized_delta_li",
                "abs_normalized_delta_li",
                "relative_delta_li",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    print()
    print(
        "Valid normalized reaction-feature samples:",
        valid_mask.sum(),
    )

    print(
        "Invalid normalized samples:",
        (
            ~valid_mask
        ).sum(),
    )

    invalid_samples = (
        df.loc[
            ~valid_mask,
            [
                "battery_id",
                "framework_formula",
                "formula_charge",
                "formula_discharge",
                "host_scale",
                "framework_match",
            ],
        ]
    )

    if len(
        invalid_samples
    ) > 0:

        print()
        print(
            "Invalid normalization examples:"
        )

        print(
            invalid_samples
            .head(
                20
            )
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "STEP 15 completed."
    )


if __name__ == "__main__":

    main()