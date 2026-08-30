from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_features.csv"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_csv(DATA_FILE)


print("Dataset shape:", df.shape)


base_columns = {
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
}


feature_columns = [
    column
    for column in df.columns
    if column not in base_columns
]


X = df[feature_columns].copy()

groups = df["framework_formula"].copy()


targets = {
    "average_voltage": "V",
    "capacity_grav": "mAh/g",
    "max_delta_volume": "fraction",
}


print("Number of features:", len(feature_columns))
print("Unique frameworks:", groups.nunique())


group_cv = GroupKFold(
    n_splits=5
)


results = []


for target_name, unit in targets.items():

    print()
    print("Target:", target_name)

    y = df[target_name].copy()

    fold_r2 = []
    fold_mae = []
    fold_rmse = []

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

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )

        model.fit(
            X_train,
            y_train,
        )

        y_pred = model.predict(
            X_test
        )

        r2 = r2_score(
            y_test,
            y_pred,
        )

        mae = mean_absolute_error(
            y_test,
            y_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred,
            )
        )

        fold_r2.append(r2)
        fold_mae.append(mae)
        fold_rmse.append(rmse)

        print(
            f"Fold {fold_number}: "
            f"R2={r2:.4f}, "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}"
        )

    mean_r2 = np.mean(
        fold_r2
    )

    std_r2 = np.std(
        fold_r2
    )

    mean_mae = np.mean(
        fold_mae
    )

    mean_rmse = np.mean(
        fold_rmse
    )

    results.append(
        {
            "target": target_name,
            "unit": unit,
            "mean_group_cv_r2": mean_r2,
            "std_group_cv_r2": std_r2,
            "mean_group_cv_mae": mean_mae,
            "mean_group_cv_rmse": mean_rmse,
        }
    )

    print()
    print(
        "Mean R2:",
        round(mean_r2, 4),
    )

    print(
        "Std R2:",
        round(std_r2, 4),
    )

    print(
        "Mean MAE:",
        round(mean_mae, 4),
    )

    print(
        "Mean RMSE:",
        round(mean_rmse, 4),
    )


results_df = pd.DataFrame(
    results
)


print()
print("Summary:")

print(results_df)


output_file = (
    RESULT_DIR
    / "xgboost_multitarget_group_cv.csv"
)


results_df.to_csv(
    output_file,
    index=False,
)


print()
print("Saved:")
print(output_file)

print()
print("STEP 09 completed.")