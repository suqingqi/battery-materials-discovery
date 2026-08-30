from pathlib import Path

import numpy as np
import pandas as pd

from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_reaction_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_capacity_physics.csv"
)

METRIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "capacity_physics_baseline.csv"
)


FARADAY_CONSTANT = 96485.33212

LI_MOLAR_MASS = float(
    Element("Li").atomic_mass
)


def remove_li(
    formula,
):

    composition = Composition(
        formula
    )

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

    return Composition(
        host_dict
    )


def calculate_capacity_features(
    row,
):

    try:

        charge_host = remove_li(
            row[
                "formula_charge"
            ]
        )

        host_mass = float(
            charge_host.weight
        )

        delta_li = float(
            row[
                "normalized_delta_li"
            ]
        )

        li_discharge = float(
            row[
                "li_discharge_normalized"
            ]
        )

        if (
            host_mass <= 0
            or delta_li <= 0
            or li_discharge < 0
        ):

            return pd.Series(
                {
                    "host_mass":
                        np.nan,
                    "discharged_mass":
                        np.nan,
                    "capacity_physics":
                        np.nan,
                }
            )

        discharged_mass = (
            host_mass
            +
            li_discharge
            *
            LI_MOLAR_MASS
        )

        capacity_physics = (
            delta_li
            *
            FARADAY_CONSTANT
            /
            (
                3.6
                *
                discharged_mass
            )
        )

        return pd.Series(
            {
                "host_mass":
                    host_mass,
                "discharged_mass":
                    discharged_mass,
                "capacity_physics":
                    capacity_physics,
            }
        )

    except Exception:

        return pd.Series(
            {
                "host_mass":
                    np.nan,
                "discharged_mass":
                    np.nan,
                "capacity_physics":
                    np.nan,
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

    physics_features = df.apply(
        calculate_capacity_features,
        axis=1,
    )

    df = pd.concat(
        [
            df,
            physics_features,
        ],
        axis=1,
    )

    print()
    print(
        "Missing physics capacities:"
    )

    print(
        df[
            "capacity_physics"
        ]
        .isna()
        .sum()
    )

    valid_df = (
        df.dropna(
            subset=[
                "capacity_grav",
                "capacity_physics",
            ]
        )
        .copy()
    )

    y_true = (
        valid_df[
            "capacity_grav"
        ]
        .astype(float)
    )

    y_pred = (
        valid_df[
            "capacity_physics"
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

    correlation = (
        np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    )

    valid_df[
        "capacity_residual"
    ] = (
        valid_df[
            "capacity_grav"
        ]
        -
        valid_df[
            "capacity_physics"
        ]
    )

    valid_df[
        "abs_capacity_residual"
    ] = (
        valid_df[
            "capacity_residual"
        ]
        .abs()
    )

    print()
    print(
        "Physics capacity statistics:"
    )

    print(
        valid_df[
            "capacity_physics"
        ]
        .describe()
    )

    print()
    print(
        "Materials Project capacity statistics:"
    )

    print(
        valid_df[
            "capacity_grav"
        ]
        .describe()
    )

    print()
    print(
        "Physics baseline:"
    )

    print(
        "R2:",
        round(
            r2,
            6,
        )
    )

    print(
        "MAE:",
        round(
            mae,
            6,
        ),
        "mAh/g",
    )

    print(
        "RMSE:",
        round(
            rmse,
            6,
        ),
        "mAh/g",
    )

    print(
        "Correlation:",
        round(
            correlation,
            6,
        )
    )

    print()
    print(
        "Residual statistics:"
    )

    print(
        valid_df[
            "capacity_residual"
        ]
        .describe()
    )

    print()
    print(
        "Largest absolute residuals:"
    )

    print(
        valid_df[
            [
                "battery_id",
                "framework_formula",
                "formula_charge",
                "formula_discharge",
                "normalized_delta_li",
                "li_discharge_normalized",
                "host_mass",
                "discharged_mass",
                "capacity_grav",
                "capacity_physics",
                "capacity_residual",
                "abs_capacity_residual",
            ]
        ]
        .sort_values(
            "abs_capacity_residual",
            ascending=False,
        )
        .head(
            15
        )
    )

    residual_map = (
        valid_df[
            [
                "battery_id",
                "capacity_residual",
                "abs_capacity_residual",
            ]
        ]
        .copy()
    )

    df = df.merge(
        residual_map,
        on="battery_id",
        how="left",
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

    metric_df = pd.DataFrame(
        [
            {
                "model":
                    "Physics capacity baseline",
                "normalization":
                    "fully discharged mass",
                "r2":
                    r2,
                "mae":
                    mae,
                "rmse":
                    rmse,
                "correlation":
                    correlation,
            }
        ]
    )

    metric_df.to_csv(
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
        "STEP 18 completed."
    )


if __name__ == "__main__":

    main()