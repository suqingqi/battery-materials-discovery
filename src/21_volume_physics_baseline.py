from pathlib import Path

import numpy as np
import pandas as pd

from pymatgen.core import Composition
from pymatgen.core import Structure

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_model.csv"
)

ENDPOINT_INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "endpoint_structure_index.csv"
)

CHARGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "structures_charge"
)

DISCHARGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "structures_discharge"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_volume_physics.csv"
)

METRIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "volume_physics_baseline.csv"
)


def remove_li(
    composition,
):

    host_dict = {}

    for element, amount in (
        composition
        .get_el_amt_dict()
        .items()
    ):

        if element == "Li":
            continue

        host_dict[element] = float(
            amount
        )

    if len(host_dict) == 0:
        return None

    return Composition(
        host_dict
    )


def get_framework_units(
    structure,
):

    host_composition = remove_li(
        structure.composition
    )

    if host_composition is None:
        return np.nan

    (
        reduced_composition,
        reduction_factor,
    ) = (
        host_composition
        .get_reduced_composition_and_factor()
    )

    framework_units = float(
        reduction_factor
    )

    if framework_units <= 0:
        return np.nan

    return framework_units


def get_normalized_volume(
    structure,
):

    framework_units = (
        get_framework_units(
            structure
        )
    )

    if np.isnan(
        framework_units
    ):
        return np.nan

    return (
        float(
            structure.volume
        )
        /
        framework_units
    )


def calculate_endpoint_features(
    row,
):

    charge_file_name = row[
        "charge_file"
    ]

    discharge_file_name = row[
        "discharge_file"
    ]

    empty_result = {
        "charge_volume_raw":
            np.nan,
        "discharge_volume_raw":
            np.nan,
        "charge_framework_units":
            np.nan,
        "discharge_framework_units":
            np.nan,
        "charge_volume_normalized":
            np.nan,
        "discharge_volume_normalized":
            np.nan,
        "endpoint_delta_signed":
            np.nan,
        "endpoint_delta_abs":
            np.nan,
    }

    if (
        pd.isna(
            charge_file_name
        )
        or
        pd.isna(
            discharge_file_name
        )
    ):
        return pd.Series(
            empty_result
        )

    charge_file = (
        CHARGE_DIR
        / str(
            charge_file_name
        )
    )

    discharge_file = (
        DISCHARGE_DIR
        / str(
            discharge_file_name
        )
    )

    if (
        not charge_file.exists()
        or
        not discharge_file.exists()
    ):
        return pd.Series(
            empty_result
        )

    try:

        charge_structure = (
            Structure.from_file(
                charge_file
            )
        )

        discharge_structure = (
            Structure.from_file(
                discharge_file
            )
        )

        charge_units = (
            get_framework_units(
                charge_structure
            )
        )

        discharge_units = (
            get_framework_units(
                discharge_structure
            )
        )

        charge_volume_raw = float(
            charge_structure.volume
        )

        discharge_volume_raw = float(
            discharge_structure.volume
        )

        charge_volume_normalized = (
            get_normalized_volume(
                charge_structure
            )
        )

        discharge_volume_normalized = (
            get_normalized_volume(
                discharge_structure
            )
        )

        if (
            charge_volume_normalized <= 0
            or
            discharge_volume_normalized <= 0
        ):
            return pd.Series(
                empty_result
            )

        endpoint_delta_signed = (
            discharge_volume_normalized
            -
            charge_volume_normalized
        ) / charge_volume_normalized

        endpoint_delta_abs = abs(
            endpoint_delta_signed
        )

        return pd.Series(
            {
                "charge_volume_raw":
                    charge_volume_raw,
                "discharge_volume_raw":
                    discharge_volume_raw,
                "charge_framework_units":
                    charge_units,
                "discharge_framework_units":
                    discharge_units,
                "charge_volume_normalized":
                    charge_volume_normalized,
                "discharge_volume_normalized":
                    discharge_volume_normalized,
                "endpoint_delta_signed":
                    endpoint_delta_signed,
                "endpoint_delta_abs":
                    endpoint_delta_abs,
            }
        )

    except Exception:

        return pd.Series(
            empty_result
        )


def evaluate_subset(
    df,
    subset_name,
):

    valid_df = (
        df.dropna(
            subset=[
                "max_delta_volume",
                "endpoint_delta_abs",
            ]
        )
        .copy()
    )

    y_true = (
        valid_df[
            "max_delta_volume"
        ]
        .astype(float)
    )

    y_pred = (
        valid_df[
            "endpoint_delta_abs"
        ]
        .astype(float)
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    correlation = np.corrcoef(
        y_true,
        y_pred,
    )[0, 1]

    residual = (
        y_true
        -
        y_pred
    )

    print()
    print(
        "Subset:",
        subset_name,
    )

    print(
        "Samples:",
        len(
            valid_df
        ),
    )

    print(
        "R2:",
        round(
            r2,
            6,
        ),
    )

    print(
        "MAE:",
        round(
            mae,
            6,
        ),
    )

    print(
        "RMSE:",
        round(
            rmse,
            6,
        ),
    )

    print(
        "Correlation:",
        round(
            correlation,
            6,
        ),
    )

    print(
        "Mean residual:",
        round(
            residual.mean(),
            6,
        ),
    )

    return {
        "subset":
            subset_name,
        "samples":
            len(
                valid_df
            ),
        "r2":
            r2,
        "mae":
            mae,
        "rmse":
            rmse,
        "correlation":
            correlation,
        "mean_residual":
            residual.mean(),
    }


def main():

    model_df = pd.read_csv(
        MODEL_FILE
    )

    endpoint_df = pd.read_csv(
        ENDPOINT_INDEX_FILE
    )

    print(
        "Model dataset:",
        model_df.shape,
    )

    print(
        "Endpoint index:",
        endpoint_df.shape,
    )

    endpoint_columns = [
        "battery_id",
        "charge_file",
        "discharge_file",
        "charge_status",
        "discharge_status",
    ]

    df = model_df.merge(
        endpoint_df[
            endpoint_columns
        ],
        on="battery_id",
        how="left",
    )

    print()
    print(
        "Merged dataset:",
        df.shape,
    )

    print()
    print(
        "Calculating normalized endpoint volumes"
    )

    volume_features = df.apply(
        calculate_endpoint_features,
        axis=1,
    )

    df = pd.concat(
        [
            df,
            volume_features,
        ],
        axis=1,
    )

    valid_mask = (
        df[
            [
                "charge_volume_normalized",
                "discharge_volume_normalized",
                "endpoint_delta_abs",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    print()
    print(
        "Valid structure pairs:",
        valid_mask.sum(),
    )

    print(
        "Invalid structure pairs:",
        (
            ~valid_mask
        ).sum(),
    )

    print()
    print(
        "Endpoint delta statistics:"
    )

    print(
        df.loc[
            valid_mask,
            "endpoint_delta_abs",
        ]
        .describe()
    )

    metrics = []

    metrics.append(
        evaluate_subset(
            df.loc[
                valid_mask
            ],
            "All valid pairs",
        )
    )

    single_step_df = (
        df.loc[
            valid_mask
            &
            (
                df[
                    "num_steps"
                ]
                == 1
            )
        ]
        .copy()
    )

    metrics.append(
        evaluate_subset(
            single_step_df,
            "Single-step",
        )
    )

    multi_step_df = (
        df.loc[
            valid_mask
            &
            (
                df[
                    "num_steps"
                ]
                > 1
            )
        ]
        .copy()
    )

    metrics.append(
        evaluate_subset(
            multi_step_df,
            "Multi-step",
        )
    )

    df[
        "volume_residual"
    ] = (
        df[
            "max_delta_volume"
        ]
        -
        df[
            "endpoint_delta_abs"
        ]
    )

    df[
        "abs_volume_residual"
    ] = (
        df[
            "volume_residual"
        ]
        .abs()
    )

    print()
    print(
        "Largest absolute residuals:"
    )

    print(
        df.loc[
            valid_mask,
            [
                "battery_id",
                "framework_formula",
                "num_steps",
                "max_delta_volume",
                "endpoint_delta_abs",
                "volume_residual",
                "abs_volume_residual",
            ],
        ]
        .sort_values(
            "abs_volume_residual",
            ascending=False,
        )
        .head(
            20
        )
    )

    print()
    print(
        "Single-step residual statistics:"
    )

    print(
        df.loc[
            valid_mask
            &
            (
                df[
                    "num_steps"
                ]
                == 1
            ),
            "volume_residual",
        ]
        .describe()
    )

    print()
    print(
        "Multi-step residual statistics:"
    )

    print(
        df.loc[
            valid_mask
            &
            (
                df[
                    "num_steps"
                ]
                > 1
            ),
            "volume_residual",
        ]
        .describe()
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRIC_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    metrics_df = pd.DataFrame(
        metrics
    )

    metrics_df.to_csv(
        METRIC_FILE,
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        METRIC_FILE
    )

    print()
    print(
        "STEP 21 completed."
    )


if __name__ == "__main__":

    main()