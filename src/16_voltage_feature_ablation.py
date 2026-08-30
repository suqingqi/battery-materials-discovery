from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold

from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_features.csv"
)

REACTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_reaction_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "voltage_feature_ablation.csv"
)


RANDOM_SEED = 42


REACTION_FEATURES = [
    "normalized_delta_li",
    "relative_delta_li",
    "num_steps",
]


BASE_COLUMNS = [
    "battery_id",
    "battery_formula",
    "framework_formula",
    "formula_charge",
    "formula_discharge",
    "working_ion",
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "max_delta_volume",
    "stability_charge",
    "stability_discharge",
    "max_stability",
    "num_steps",
    "nelements",
    "chemsys",
    "elements",
    "element_list",
    "id_charge",
    "id_discharge",
    "material_ids",
    "warnings",
    "has_transition_metal",
    "has_main_transition_metal",
    "oxide",
    "sulfide",
    "fluoride",
    "phosphate_possible",
    "silicate_possible",
]


def build_model():

    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=4,
    )

    return model


def evaluate_feature_set(
    X,
    y,
    groups,
    model_name,
):

    group_cv = GroupKFold(
        n_splits=5
    )

    fold_results = []

    print()
    print(
        "Model:",
        model_name
    )

    print(
        "Number of features:",
        X.shape[1]
    )

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        group_cv.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):

        X_train = (
            X.iloc[
                train_index
            ]
        )

        X_test = (
            X.iloc[
                test_index
            ]
        )

        y_train = (
            y.iloc[
                train_index
            ]
        )

        y_test = (
            y.iloc[
                test_index
            ]
        )

        model = build_model()

        model.fit(
            X_train,
            y_train,
        )

        prediction = model.predict(
            X_test
        )

        r2 = r2_score(
            y_test,
            prediction,
        )

        mae = mean_absolute_error(
            y_test,
            prediction,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                prediction,
            )
        )

        fold_results.append(
            {
                "fold":
                    fold_number,
                "r2":
                    r2,
                "mae":
                    mae,
                "rmse":
                    rmse,
            }
        )

        print(
            f"Fold {fold_number}: "
            f"R2={r2:.4f}, "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}"
        )

    fold_df = pd.DataFrame(
        fold_results
    )

    summary = {
        "model":
            model_name,
        "num_features":
            X.shape[1],
        "mean_r2":
            fold_df[
                "r2"
            ].mean(),
        "std_r2":
            fold_df[
                "r2"
            ].std(),
        "mean_mae":
            fold_df[
                "mae"
            ].mean(),
        "mean_rmse":
            fold_df[
                "rmse"
            ].mean(),
    }

    print()

    print(
        "Mean R2:",
        round(
            summary[
                "mean_r2"
            ],
            4,
        )
    )

    print(
        "Std R2:",
        round(
            summary[
                "std_r2"
            ],
            4,
        )
    )

    print(
        "Mean MAE:",
        round(
            summary[
                "mean_mae"
            ],
            4,
        )
    )

    print(
        "Mean RMSE:",
        round(
            summary[
                "mean_rmse"
            ],
            4,
        )
    )

    return summary


def main():

    feature_df = pd.read_csv(
        FEATURE_FILE
    )

    reaction_df = pd.read_csv(
        REACTION_FILE
    )

    print(
        "Composition dataset:",
        feature_df.shape,
    )

    print(
        "Reaction dataset:",
        reaction_df.shape,
    )

    reaction_columns = [
        "battery_id",
        "normalized_delta_li",
        "relative_delta_li",
        "num_steps",
    ]

    reaction_subset = (
        reaction_df[
            reaction_columns
        ]
        .copy()
    )

    merged_df = feature_df.merge(
        reaction_subset,
        on="battery_id",
        how="inner",
        suffixes=(
            "",
            "_reaction",
        ),
    )

    if "num_steps_reaction" in merged_df.columns:

        merged_df[
            "num_steps"
        ] = merged_df[
            "num_steps_reaction"
        ]

        merged_df = merged_df.drop(
            columns=[
                "num_steps_reaction"
            ]
        )

    print()
    print(
        "Merged dataset:",
        merged_df.shape,
    )

    print(
        "Unique frameworks:",
        merged_df[
            "framework_formula"
        ].nunique(),
    )

    matminer_features = []

    for column in feature_df.columns:

        if column in BASE_COLUMNS:

            continue

        if not pd.api.types.is_numeric_dtype(
            feature_df[
                column
            ]
        ):

            continue

        matminer_features.append(
            column
        )

    print()
    print(
        "Composition features:",
        len(
            matminer_features
        ),
    )

    print(
        "Reaction features:",
        REACTION_FEATURES,
    )

    y = (
        merged_df[
            "average_voltage"
        ]
        .astype(
            float
        )
    )

    groups = (
        merged_df[
            "framework_formula"
        ]
    )

    X_composition = (
        merged_df[
            matminer_features
        ]
        .astype(
            float
        )
    )

    X_reaction = (
        merged_df[
            REACTION_FEATURES
        ]
        .astype(
            float
        )
    )

    combined_features = (
        matminer_features
        +
        REACTION_FEATURES
    )

    X_combined = (
        merged_df[
            combined_features
        ]
        .astype(
            float
        )
    )

    summaries = []

    summaries.append(
        evaluate_feature_set(
            X_composition,
            y,
            groups,
            "Composition only",
        )
    )

    summaries.append(
        evaluate_feature_set(
            X_reaction,
            y,
            groups,
            "Reaction only",
        )
    )

    summaries.append(
        evaluate_feature_set(
            X_combined,
            y,
            groups,
            "Composition + Reaction",
        )
    )

    summary_df = pd.DataFrame(
        summaries
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Ablation summary:"
    )

    print(
        summary_df
    )

    composition_r2 = (
        summary_df.loc[
            summary_df[
                "model"
            ]
            ==
            "Composition only",
            "mean_r2",
        ]
        .iloc[
            0
        ]
    )

    combined_r2 = (
        summary_df.loc[
            summary_df[
                "model"
            ]
            ==
            "Composition + Reaction",
            "mean_r2",
        ]
        .iloc[
            0
        ]
    )

    delta_r2 = (
        combined_r2
        -
        composition_r2
    )

    print()
    print(
        "R2 change after adding reaction features:",
        round(
            delta_r2,
            4,
        )
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
        "STEP 16 completed."
    )


if __name__ == "__main__":

    main()