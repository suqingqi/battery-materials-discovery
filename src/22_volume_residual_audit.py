from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_volume_physics.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "volume_residual_audit.csv"
)


THRESHOLDS = [
    1e-6,
    1e-4,
    1e-3,
    0.005,
    0.01,
    0.02,
    0.05,
]


def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    valid_df = (
        df.dropna(
            subset=[
                "max_delta_volume",
                "endpoint_delta_abs",
                "volume_residual",
            ]
        )
        .copy()
    )

    valid_df[
        "abs_volume_residual"
    ] = (
        valid_df[
            "volume_residual"
        ]
        .abs()
    )

    print(
        "Valid samples:",
        len(valid_df),
    )

    print()

    print(
        "Residual threshold coverage:"
    )

    threshold_rows = []

    for threshold in THRESHOLDS:

        count = (
            valid_df[
                "abs_volume_residual"
            ]
            <= threshold
        ).sum()

        fraction = (
            count
            /
            len(valid_df)
        )

        print(
            f"|residual| <= {threshold}: "
            f"{count} "
            f"({fraction:.2%})"
        )

        threshold_rows.append(
            {
                "threshold":
                    threshold,
                "count":
                    count,
                "fraction":
                    fraction,
            }
        )

    print()
    print(
        "Single-step threshold coverage:"
    )

    single_df = (
        valid_df[
            valid_df[
                "num_steps"
            ]
            == 1
        ]
        .copy()
    )

    for threshold in THRESHOLDS:

        count = (
            single_df[
                "abs_volume_residual"
            ]
            <= threshold
        ).sum()

        fraction = (
            count
            /
            len(single_df)
        )

        print(
            f"|residual| <= {threshold}: "
            f"{count} "
            f"({fraction:.2%})"
        )

    print()
    print(
        "Multi-step threshold coverage:"
    )

    multi_df = (
        valid_df[
            valid_df[
                "num_steps"
            ]
            > 1
        ]
        .copy()
    )

    for threshold in THRESHOLDS:

        count = (
            multi_df[
                "abs_volume_residual"
            ]
            <= threshold
        ).sum()

        fraction = (
            count
            /
            len(multi_df)
        )

        print(
            f"|residual| <= {threshold}: "
            f"{count} "
            f"({fraction:.2%})"
        )

    valid_df[
        "contains_F"
    ] = (
        valid_df[
            "framework_formula"
        ]
        .astype(str)
        .str.contains(
            "F",
            regex=False,
        )
    )

    valid_df[
        "contains_V"
    ] = (
        valid_df[
            "framework_formula"
        ]
        .astype(str)
        .str.contains(
            "V",
            regex=False,
        )
    )

    valid_df[
        "contains_Mn"
    ] = (
        valid_df[
            "framework_formula"
        ]
        .astype(str)
        .str.contains(
            "Mn",
            regex=False,
        )
    )

    valid_df[
        "contains_Fe"
    ] = (
        valid_df[
            "framework_formula"
        ]
        .astype(str)
        .str.contains(
            "Fe",
            regex=False,
        )
    )

    valid_df[
        "contains_Co"
    ] = (
        valid_df[
            "framework_formula"
        ]
        .astype(str)
        .str.contains(
            "Co",
            regex=False,
        )
    )

    valid_df[
        "contains_Ni"
    ] = (
        valid_df[
            "framework_formula"
        ]
        .astype(str)
        .str.contains(
            "Ni",
            regex=False,
        )
    )

    print()
    print(
        "Residual by chemistry flag:"
    )

    chemistry_flags = [
        "contains_F",
        "contains_V",
        "contains_Mn",
        "contains_Fe",
        "contains_Co",
        "contains_Ni",
    ]

    chemistry_rows = []

    for flag in chemistry_flags:

        subset = (
            valid_df[
                valid_df[
                    flag
                ]
            ]
        )

        mean_abs_residual = (
            subset[
                "abs_volume_residual"
            ]
            .mean()
        )

        median_abs_residual = (
            subset[
                "abs_volume_residual"
            ]
            .median()
        )

        print()
        print(
            flag
        )

        print(
            "Samples:",
            len(subset),
        )

        print(
            "Mean |residual|:",
            round(
                mean_abs_residual,
                6,
            ),
        )

        print(
            "Median |residual|:",
            round(
                median_abs_residual,
                6,
            ),
        )

        chemistry_rows.append(
            {
                "group":
                    flag,
                "samples":
                    len(subset),
                "mean_abs_residual":
                    mean_abs_residual,
                "median_abs_residual":
                    median_abs_residual,
            }
        )

    print()
    print(
        "Top 30 residual samples:"
    )

    print(
        valid_df[
            [
                "battery_id",
                "framework_formula",
                "num_steps",
                "max_delta_volume",
                "endpoint_delta_abs",
                "volume_residual",
                "abs_volume_residual",
                "contains_F",
                "contains_V",
            ]
        ]
        .sort_values(
            "abs_volume_residual",
            ascending=False,
        )
        .head(
            30
        )
    )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    chemistry_df = pd.DataFrame(
        chemistry_rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_output = (
        valid_df[
            [
                "battery_id",
                "framework_formula",
                "num_steps",
                "max_delta_volume",
                "endpoint_delta_abs",
                "volume_residual",
                "abs_volume_residual",
                "contains_F",
                "contains_V",
                "contains_Mn",
                "contains_Fe",
                "contains_Co",
                "contains_Ni",
            ]
        ]
        .copy()
    )

    combined_output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    threshold_df.to_csv(
        PROJECT_ROOT
        / "results"
        / "metrics"
        / "volume_residual_thresholds.csv",
        index=False,
    )

    chemistry_df.to_csv(
        PROJECT_ROOT
        / "results"
        / "metrics"
        / "volume_residual_chemistry.csv",
        index=False,
    )

    print()
    print(
        "STEP 22 completed."
    )


if __name__ == "__main__":

    main()