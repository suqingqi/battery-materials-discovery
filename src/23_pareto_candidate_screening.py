from pathlib import Path

import numpy as np
import pandas as pd

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

CAPACITY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_capacity_physics.csv"
)

VOLUME_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_volume_physics.csv"
)

OUTPUT_ALL = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "candidate_screening_all.csv"
)

OUTPUT_FILTERED = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "candidate_screening_filtered.csv"
)

OUTPUT_PARETO = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "candidate_pareto_front.csv"
)

OUTPUT_TOP = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "candidate_top20.csv"
)


RANDOM_SEED = 42


MIN_VOLTAGE = 3.0

MAX_VOLTAGE = 5.0

MIN_CAPACITY = 100.0

MAX_VOLUME_CHANGE = 0.10

MAX_STABILITY = 0.10

REQUIRE_SINGLE_STEP = True


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


def build_voltage_model():

    return XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=4,
    )


def minmax_scale(
    series,
    higher_is_better=True,
):

    minimum = series.min()

    maximum = series.max()

    if np.isclose(
        maximum,
        minimum,
    ):
        return pd.Series(
            np.ones(
                len(series)
            ),
            index=series.index,
        )

    scaled = (
        series
        -
        minimum
    ) / (
        maximum
        -
        minimum
    )

    if not higher_is_better:

        scaled = (
            1.0
            -
            scaled
        )

    return scaled


def identify_pareto_front(
    df,
):

    values = df[
        [
            "voltage_oof_pred",
            "capacity_physics",
            "endpoint_delta_abs",
            "max_stability",
        ]
    ].to_numpy(
        dtype=float
    )

    transformed = values.copy()

    transformed[
        :,
        2
    ] = (
        -transformed[
            :,
            2
        ]
    )

    transformed[
        :,
        3
    ] = (
        -transformed[
            :,
            3
        ]
    )

    n_samples = len(
        transformed
    )

    is_pareto = np.ones(
        n_samples,
        dtype=bool,
    )

    for i in range(
        n_samples
    ):

        if not is_pareto[
            i
        ]:
            continue

        point = transformed[
            i
        ]

        for j in range(
            n_samples
        ):

            if i == j:
                continue

            other = transformed[
                j
            ]

            at_least_as_good = np.all(
                other
                >=
                point
            )

            strictly_better = np.any(
                other
                >
                point
            )

            if (
                at_least_as_good
                and
                strictly_better
            ):

                is_pareto[
                    i
                ] = False

                break

    return is_pareto


def main():

    feature_df = pd.read_csv(
        FEATURE_FILE
    )

    reaction_df = pd.read_csv(
        REACTION_FILE
    )

    capacity_df = pd.read_csv(
        CAPACITY_FILE
    )

    volume_df = pd.read_csv(
        VOLUME_FILE
    )

    print(
        "Feature dataset:",
        feature_df.shape,
    )

    print(
        "Reaction dataset:",
        reaction_df.shape,
    )

    print(
        "Capacity dataset:",
        capacity_df.shape,
    )

    print(
        "Volume dataset:",
        volume_df.shape,
    )

    reaction_subset = (
        reaction_df[
            [
                "battery_id",
                "normalized_delta_li",
                "relative_delta_li",
            ]
        ]
        .copy()
    )

    capacity_subset = (
        capacity_df[
            [
                "battery_id",
                "capacity_physics",
            ]
        ]
        .copy()
    )

    volume_subset = (
        volume_df[
            [
                "battery_id",
                "endpoint_delta_abs",
                "volume_residual",
            ]
        ]
        .copy()
    )

    df = (
        feature_df
        .merge(
            reaction_subset,
            on="battery_id",
            how="inner",
        )
        .merge(
            capacity_subset,
            on="battery_id",
            how="inner",
        )
        .merge(
            volume_subset,
            on="battery_id",
            how="inner",
        )
    )

    print()
    print(
        "Merged dataset:",
        df.shape,
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

    final_voltage_features = (
        matminer_features
        +
        REACTION_FEATURES
    )

    print(
        "Voltage features:",
        len(
            final_voltage_features
        ),
    )

    X = (
        df[
            final_voltage_features
        ]
        .astype(float)
    )

    y = (
        df[
            "average_voltage"
        ]
        .astype(float)
    )

    groups = (
        df[
            "framework_formula"
        ]
    )

    group_cv = GroupKFold(
        n_splits=5
    )

    oof_predictions = np.full(
        len(
            df
        ),
        np.nan,
        dtype=float,
    )

    fold_number_array = np.zeros(
        len(
            df
        ),
        dtype=int,
    )

    print()
    print(
        "Generating framework-group OOF voltage predictions"
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

        model = (
            build_voltage_model()
        )

        model.fit(
            X.iloc[
                train_index
            ],
            y.iloc[
                train_index
            ],
        )

        predictions = (
            model.predict(
                X.iloc[
                    test_index
                ]
            )
        )

        oof_predictions[
            test_index
        ] = predictions

        fold_number_array[
            test_index
        ] = fold_number

        print(
            "Fold",
            fold_number,
            "completed:",
            len(
                test_index
            ),
            "predictions",
        )

    df[
        "voltage_oof_pred"
    ] = oof_predictions

    df[
        "oof_fold"
    ] = fold_number_array

    df[
        "voltage_oof_error"
    ] = (
        df[
            "average_voltage"
        ]
        -
        df[
            "voltage_oof_pred"
        ]
    )

    print()
    print(
        "Missing OOF predictions:",
        df[
            "voltage_oof_pred"
        ]
        .isna()
        .sum(),
    )

    valid_mask = (
        df[
            [
                "voltage_oof_pred",
                "capacity_physics",
                "endpoint_delta_abs",
                "max_stability",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    df = (
        df[
            valid_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "Candidates with complete metrics:",
        len(
            df
        ),
    )

    constraint_mask = (
        (
            df[
                "voltage_oof_pred"
            ]
            >=
            MIN_VOLTAGE
        )
        &
        (
            df[
                "voltage_oof_pred"
            ]
            <=
            MAX_VOLTAGE
        )
        &
        (
            df[
                "capacity_physics"
            ]
            >=
            MIN_CAPACITY
        )
        &
        (
            df[
                "endpoint_delta_abs"
            ]
            <=
            MAX_VOLUME_CHANGE
        )
        &
        (
            df[
                "max_stability"
            ]
            <=
            MAX_STABILITY
        )
    )

    if REQUIRE_SINGLE_STEP:

        constraint_mask = (
            constraint_mask
            &
            (
                df[
                    "num_steps"
                ]
                == 1
            )
        )

    filtered_df = (
        df[
            constraint_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "Engineering constraints:"
    )

    print(
        "Voltage:",
        MIN_VOLTAGE,
        "to",
        MAX_VOLTAGE,
        "V",
    )

    print(
        "Capacity >=",
        MIN_CAPACITY,
        "mAh/g",
    )

    print(
        "Endpoint volume change <=",
        MAX_VOLUME_CHANGE,
    )

    print(
        "Max stability <=",
        MAX_STABILITY,
        "eV/atom",
    )

    print(
        "Single-step required:",
        REQUIRE_SINGLE_STEP,
    )

    print()
    print(
        "Candidates after constraints:",
        len(
            filtered_df
        ),
    )

    if len(
        filtered_df
    ) == 0:

        raise RuntimeError(
            "No candidates passed engineering constraints."
        )

    filtered_df[
        "pareto"
    ] = (
        identify_pareto_front(
            filtered_df
        )
    )

    pareto_df = (
        filtered_df[
            filtered_df[
                "pareto"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "Pareto candidates:",
        len(
            pareto_df
        ),
    )

    filtered_df[
        "score_voltage"
    ] = minmax_scale(
        filtered_df[
            "voltage_oof_pred"
        ],
        higher_is_better=True,
    )

    filtered_df[
        "score_capacity"
    ] = minmax_scale(
        filtered_df[
            "capacity_physics"
        ],
        higher_is_better=True,
    )

    filtered_df[
        "score_volume"
    ] = minmax_scale(
        filtered_df[
            "endpoint_delta_abs"
        ],
        higher_is_better=False,
    )

    filtered_df[
        "score_stability"
    ] = minmax_scale(
        filtered_df[
            "max_stability"
        ],
        higher_is_better=False,
    )

    filtered_df[
        "composite_score"
    ] = (
        0.30
        *
        filtered_df[
            "score_voltage"
        ]
        +
        0.30
        *
        filtered_df[
            "score_capacity"
        ]
        +
        0.20
        *
        filtered_df[
            "score_volume"
        ]
        +
        0.20
        *
        filtered_df[
            "score_stability"
        ]
    )

    filtered_df = (
        filtered_df
        .sort_values(
            [
                "pareto",
                "composite_score",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    filtered_df[
        "screening_rank"
    ] = np.arange(
        1,
        len(
            filtered_df
        ) + 1,
    )

    pareto_ids = set(
        pareto_df[
            "battery_id"
        ]
    )

    pareto_df = (
        filtered_df[
            filtered_df[
                "battery_id"
            ]
            .isin(
                pareto_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    top20_df = (
        filtered_df
        .head(
            20
        )
        .copy()
    )

    display_columns = [
        "screening_rank",
        "battery_id",
        "framework_formula",
        "formula_charge",
        "formula_discharge",
        "voltage_oof_pred",
        "average_voltage",
        "capacity_physics",
        "endpoint_delta_abs",
        "max_stability",
        "num_steps",
        "pareto",
        "composite_score",
    ]

    print()
    print(
        "Top 20 candidates:"
    )

    print(
        top20_df[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Top Pareto candidates:"
    )

    print(
        pareto_df[
            display_columns
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    OUTPUT_ALL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_ALL,
        index=False,
    )

    filtered_df.to_csv(
        OUTPUT_FILTERED,
        index=False,
    )

    pareto_df.to_csv(
        OUTPUT_PARETO,
        index=False,
    )

    top20_df.to_csv(
        OUTPUT_TOP,
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_ALL
    )

    print(
        OUTPUT_FILTERED
    )

    print(
        OUTPUT_PARETO
    )

    print(
        OUTPUT_TOP
    )

    print()
    print(
        "STEP 23 completed."
    )


if __name__ == "__main__":

    main()