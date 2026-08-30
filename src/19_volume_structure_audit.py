from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_model.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "volume_structure_audit.csv"
)


def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Dataset shape:",
        df.shape,
    )

    audit_columns = [
        "battery_id",
        "framework_formula",
        "formula_charge",
        "formula_discharge",
        "id_charge",
        "id_discharge",
        "max_delta_volume",
        "num_steps",
    ]

    audit_df = (
        df[
            audit_columns
        ]
        .copy()
    )

    print()
    print(
        "Missing IDs:"
    )

    print(
        audit_df[
            [
                "id_charge",
                "id_discharge",
            ]
        ]
        .isna()
        .sum()
    )

    print()
    print(
        "Unique charge structures:",
        audit_df[
            "id_charge"
        ]
        .nunique(),
    )

    print(
        "Unique discharge structures:",
        audit_df[
            "id_discharge"
        ]
        .nunique(),
    )

    print()
    print(
        "Same charge/discharge ID:"
    )

    same_id = (
        audit_df[
            "id_charge"
        ]
        ==
        audit_df[
            "id_discharge"
        ]
    )

    print(
        same_id.sum(),
    )

    print()
    print(
        "max_delta_volume statistics:"
    )

    print(
        audit_df[
            "max_delta_volume"
        ]
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
    )

    print()
    print(
        "num_steps distribution:"
    )

    print(
        audit_df[
            "num_steps"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Volume change by num_steps:"
    )

    volume_by_step = (
        audit_df
        .groupby(
            "num_steps"
        )
        .agg(
            count=(
                "battery_id",
                "count",
            ),
            mean_volume_change=(
                "max_delta_volume",
                "mean",
            ),
            median_volume_change=(
                "max_delta_volume",
                "median",
            ),
            max_volume_change=(
                "max_delta_volume",
                "max",
            ),
        )
        .reset_index()
    )

    print(
        volume_by_step
    )

    print()
    print(
        "Largest volume-change samples:"
    )

    print(
        audit_df[
            [
                "battery_id",
                "framework_formula",
                "formula_charge",
                "formula_discharge",
                "id_charge",
                "id_discharge",
                "num_steps",
                "max_delta_volume",
            ]
        ]
        .sort_values(
            "max_delta_volume",
            ascending=False,
        )
        .head(
            20
        )
    )

    print()
    print(
        "Smallest volume-change samples:"
    )

    print(
        audit_df[
            [
                "battery_id",
                "framework_formula",
                "formula_charge",
                "formula_discharge",
                "id_charge",
                "id_discharge",
                "num_steps",
                "max_delta_volume",
            ]
        ]
        .sort_values(
            "max_delta_volume",
            ascending=True,
        )
        .head(
            20
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_df.to_csv(
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
        "STEP 19 completed."
    )


if __name__ == "__main__":

    main()