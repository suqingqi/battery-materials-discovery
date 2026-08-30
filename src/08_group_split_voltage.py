from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
    KFold,
    cross_val_score,
)

from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_features.csv"
)


df = pd.read_csv(DATA_FILE)


target = "average_voltage"


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


X = df[feature_columns]

y = df[target]

groups = df["framework_formula"]


print("Dataset shape:", df.shape)
print("Features:", len(feature_columns))
print(
    "Unique framework formulas:",
    groups.nunique(),
)


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


random_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


random_r2 = cross_val_score(
    model,
    X,
    y,
    cv=random_cv,
    scoring="r2",
    n_jobs=1,
)


random_mae = -cross_val_score(
    model,
    X,
    y,
    cv=random_cv,
    scoring="neg_mean_absolute_error",
    n_jobs=1,
)


print()
print("Random 5-fold CV")

print("R2:")
print(random_r2)

print(
    "Mean R2:",
    round(
        random_r2.mean(),
        4,
    )
)

print(
    "Std R2:",
    round(
        random_r2.std(),
        4,
    )
)

print(
    "Mean MAE:",
    round(
        random_mae.mean(),
        4,
    )
)


group_cv = GroupKFold(
    n_splits=5
)


group_r2 = cross_val_score(
    model,
    X,
    y,
    groups=groups,
    cv=group_cv,
    scoring="r2",
    n_jobs=1,
)


group_mae = -cross_val_score(
    model,
    X,
    y,
    groups=groups,
    cv=group_cv,
    scoring="neg_mean_absolute_error",
    n_jobs=1,
)


print()
print("Framework Group 5-fold CV")

print("R2:")
print(group_r2)

print(
    "Mean R2:",
    round(
        group_r2.mean(),
        4,
    )
)

print(
    "Std R2:",
    round(
        group_r2.std(),
        4,
    )
)

print(
    "Mean MAE:",
    round(
        group_mae.mean(),
        4,
    )
)


r2_drop = (
    random_r2.mean()
    - group_r2.mean()
)


mae_increase = (
    group_mae.mean()
    - random_mae.mean()
)


print()
print("Generalization gap")

print(
    "R2 drop:",
    round(
        r2_drop,
        4,
    )
)

print(
    "MAE increase:",
    round(
        mae_increase,
        4,
    )
)


print()
print("STEP 08 completed.")