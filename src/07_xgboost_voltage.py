from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
)

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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
)

FIGURE_DIR = (
    RESULT_DIR
    / "figures"
)

METRIC_DIR = (
    RESULT_DIR
    / "metrics"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_csv(DATA_FILE)


print("Dataset shape:", df.shape)


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


X = df[feature_columns].copy()

y = df[target].copy()


print("Number of features:", X.shape[1])


non_numeric_columns = (
    X.select_dtypes(
        exclude="number"
    )
    .columns
    .tolist()
)


print(
    "Non-numeric feature columns:",
    non_numeric_columns,
)


if len(non_numeric_columns) > 0:

    raise RuntimeError(
        "Non-numeric columns found in feature matrix."
    )


print(
    "Missing values in X:",
    int(X.isna().sum().sum()),
)


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )
)


print()
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


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


print()
print("Holdout performance:")
print("R2:", round(r2, 4))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))


cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


cv_r2 = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="r2",
    n_jobs=1,
)


cv_mae = -cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=1,
)


print()
print("5-fold CV R2:")
print(cv_r2)

print(
    "Mean CV R2:",
    round(cv_r2.mean(), 4),
)

print(
    "Std CV R2:",
    round(cv_r2.std(), 4),
)


print()
print("5-fold CV MAE:")
print(cv_mae)

print(
    "Mean CV MAE:",
    round(cv_mae.mean(), 4),
)


importance_df = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance":
            model.feature_importances_,
    }
)


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


importance_file = (
    METRIC_DIR
    / "voltage_feature_importance.csv"
)


importance_df.to_csv(
    importance_file,
    index=False,
)


print()
print("Top 20 features:")
print(
    importance_df.head(20)
)


plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6,
)

minimum = min(
    y_test.min(),
    y_pred.min(),
)

maximum = max(
    y_test.max(),
    y_pred.max(),
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
)

plt.xlabel(
    "Actual Voltage (V)"
)

plt.ylabel(
    "Predicted Voltage (V)"
)

plt.title(
    "XGBoost Voltage Prediction"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "xgboost_voltage_actual_vs_predicted.png",
    dpi=300,
)

plt.close()


top_features = (
    importance_df
    .head(20)
    .sort_values(
        "importance",
        ascending=True,
    )
)


plt.figure(
    figsize=(8, 7)
)

plt.barh(
    top_features["feature"],
    top_features["importance"],
)

plt.xlabel(
    "Feature Importance"
)

plt.title(
    "Top 20 XGBoost Features"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "xgboost_voltage_feature_importance.png",
    dpi=300,
)

plt.close()


prediction_df = pd.DataFrame(
    {
        "actual_voltage":
            y_test.values,
        "predicted_voltage":
            y_pred,
        "error":
            y_pred
            - y_test.values,
    }
)


prediction_df.to_csv(
    METRIC_DIR
    / "voltage_predictions.csv",
    index=False,
)


joblib.dump(
    model,
    MODEL_DIR
    / "xgboost_voltage.pkl",
)


print()
print(
    "Model saved:"
)

print(
    MODEL_DIR
    / "xgboost_voltage.pkl"
)

print()
print("STEP 07 completed.")