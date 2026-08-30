from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xgboost_voltage_final.pkl"
)

IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "voltage_shap_importance.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "voltage_shap"
)


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


RANDOM_SEED = 42


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


def load_data():

    feature_df = pd.read_csv(
        FEATURE_FILE
    )

    reaction_df = pd.read_csv(
        REACTION_FILE
    )

    reaction_subset = (
        reaction_df[
            [
                "battery_id",
                "normalized_delta_li",
                "relative_delta_li",
                "num_steps",
            ]
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

    matminer_features = []

    for column in feature_df.columns:

        if column in BASE_COLUMNS:
            continue

        if not pd.api.types.is_numeric_dtype(
            feature_df[column]
        ):
            continue

        matminer_features.append(
            column
        )

    final_features = (
        matminer_features
        +
        REACTION_FEATURES
    )

    X = (
        merged_df[
            final_features
        ]
        .astype(float)
        .copy()
    )

    y = (
        merged_df[
            "average_voltage"
        ]
        .astype(float)
        .copy()
    )

    return (
        merged_df,
        X,
        y,
        matminer_features,
        final_features,
    )


def save_shap_importance(
    shap_values,
    feature_names,
):

    mean_abs_shap = np.abs(
        shap_values.values
    ).mean(
        axis=0
    )

    importance_df = pd.DataFrame(
        {
            "feature":
                feature_names,
            "mean_abs_shap":
                mean_abs_shap,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    importance_df[
        "rank"
    ] = (
        np.arange(
            1,
            len(
                importance_df
            ) + 1,
        )
    )

    importance_df = (
        importance_df[
            [
                "rank",
                "feature",
                "mean_abs_shap",
            ]
        ]
    )

    importance_df.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    return importance_df


def save_beeswarm(
    shap_values,
):

    plt.figure()

    shap.plots.beeswarm(
        shap_values,
        max_display=20,
        show=False,
    )

    plt.tight_layout()

    output_file = (
        FIGURE_DIR
        / "voltage_shap_beeswarm.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_file


def save_bar_plot(
    shap_values,
):

    plt.figure()

    shap.plots.bar(
        shap_values,
        max_display=20,
        show=False,
    )

    plt.tight_layout()

    output_file = (
        FIGURE_DIR
        / "voltage_shap_bar.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_file


def save_reaction_scatter(
    shap_values,
    X,
    feature_name,
):

    if feature_name not in X.columns:

        return None

    plt.figure()

    shap.plots.scatter(
        shap_values[
            :,
            feature_name
        ],
        show=False,
    )

    plt.tight_layout()

    safe_name = (
        feature_name
        .replace(
            " ",
            "_",
        )
    )

    output_file = (
        FIGURE_DIR
        / f"voltage_shap_{safe_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_file


def main():

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    IMPORTANCE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        merged_df,
        X,
        y,
        matminer_features,
        final_features,
    ) = load_data()

    print(
        "Dataset shape:",
        merged_df.shape,
    )

    print(
        "Composition features:",
        len(
            matminer_features
        ),
    )

    print(
        "Reaction features:",
        len(
            REACTION_FEATURES
        ),
    )

    print(
        "Total final features:",
        len(
            final_features
        ),
    )

    print()

    print(
        "Final benchmark from STEP 16:"
    )

    print(
        "Framework Group 5-fold R2: 0.5302"
    )

    print(
        "Framework Group 5-fold MAE: 0.3966 V"
    )

    print()

    print(
        "Training final model on all samples"
    )

    model = build_model()

    model.fit(
        X,
        y,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print()

    print(
        "Model saved:"
    )

    print(
        MODEL_FILE
    )

    print()

    print(
        "Calculating SHAP values"
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        X
    )

    print(
        "SHAP shape:",
        shap_values.values.shape,
    )

    importance_df = (
        save_shap_importance(
            shap_values,
            final_features,
        )
    )

    print()

    print(
        "Top 20 SHAP features:"
    )

    print(
        importance_df.head(
            20
        )
    )

    print()

    print(
        "Reaction feature rankings:"
    )

    reaction_rankings = (
        importance_df[
            importance_df[
                "feature"
            ].isin(
                REACTION_FEATURES
            )
        ]
    )

    print(
        reaction_rankings
    )

    beeswarm_file = (
        save_beeswarm(
            shap_values
        )
    )

    bar_file = (
        save_bar_plot(
            shap_values
        )
    )

    relative_li_file = (
        save_reaction_scatter(
            shap_values,
            X,
            "relative_delta_li",
        )
    )

    num_steps_file = (
        save_reaction_scatter(
            shap_values,
            X,
            "num_steps",
        )
    )

    print()

    print(
        "Figures saved:"
    )

    print(
        beeswarm_file
    )

    print(
        bar_file
    )

    if relative_li_file is not None:

        print(
            relative_li_file
        )

    if num_steps_file is not None:

        print(
            num_steps_file
        )

    print()

    print(
        "Importance CSV:"
    )

    print(
        IMPORTANCE_FILE
    )

    print()

    print(
        "STEP 17 completed."
    )


if __name__ == "__main__":

    main()