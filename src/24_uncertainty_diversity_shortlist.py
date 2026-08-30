from pathlib import Path

import numpy as np
import pandas as pd

from pymatgen.core import Composition

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCREENING_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "candidate_screening_all.csv"
)

PARETO_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "candidate_pareto_front.csv"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_features.csv"
)

OUTPUT_UNCERTAINTY = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "candidate_uncertainty.csv"
)

OUTPUT_PARETO = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "candidate_pareto_uncertainty.csv"
)

OUTPUT_SHORTLIST = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "dft_shortlist.csv"
)


K_NEIGHBORS = 20

SEARCH_NEIGHBORS = 80

N_DFT_CANDIDATES = 4


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


TRANSITION_METALS = [
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
]


def minmax_scale(
    series,
    higher_is_better=True,
):

    minimum = series.min()

    maximum = series.max()

    if np.isclose(
        minimum,
        maximum,
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


def get_main_transition_metal(
    formula,
):

    try:

        composition = Composition(
            formula
        )

        amounts = (
            composition
            .get_el_amt_dict()
        )

        candidates = {}

        for metal in TRANSITION_METALS:

            if metal in amounts:

                candidates[
                    metal
                ] = amounts[
                    metal
                ]

        if len(
            candidates
        ) == 0:

            return "Other"

        main_metal = max(
            candidates,
            key=candidates.get,
        )

        return main_metal

    except Exception:

        return "Unknown"


def get_chemistry_family(
    formula,
):

    try:

        composition = Composition(
            formula
        )

        elements = set(
            composition
            .get_el_amt_dict()
            .keys()
        )

        if (
            "P" in elements
            and
            "O" in elements
        ):

            return "P-O"

        if (
            "S" in elements
            and
            "O" in elements
        ):

            return "S-O"

        if (
            "F" in elements
            and
            "O" not in elements
        ):

            return "Fluoride"

        if (
            "F" in elements
            and
            "O" in elements
        ):

            return "Oxyfluoride"

        if "O" in elements:

            return "Oxide"

        if "S" in elements:

            return "Sulfide"

        return "Other"

    except Exception:

        return "Unknown"


def build_feature_list(
    feature_df,
):

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

    return (
        matminer_features
        +
        REACTION_FEATURES
    )


def calculate_local_uncertainty(
    df,
    feature_columns,
):

    X = (
        df[
            feature_columns
        ]
        .astype(float)
        .copy()
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    n_search = min(
        SEARCH_NEIGHBORS,
        len(
            df
        ),
    )

    neighbor_model = NearestNeighbors(
        n_neighbors=n_search,
        metric="euclidean",
        n_jobs=-1,
    )

    neighbor_model.fit(
        X_scaled
    )

    distances, indices = (
        neighbor_model.kneighbors(
            X_scaled
        )
    )

    abs_oof_error = (
        df[
            "voltage_oof_error"
        ]
        .abs()
        .to_numpy()
    )

    frameworks = (
        df[
            "framework_formula"
        ]
        .astype(str)
        .to_numpy()
    )

    local_mae_list = []

    local_q90_list = []

    nearest_distance_list = []

    neighbor_count_list = []


    for i in range(
        len(
            df
        )
    ):

        selected_errors = []

        selected_distances = []

        current_framework = (
            frameworks[
                i
            ]
        )

        for distance, neighbor_index in zip(
            distances[
                i
            ],
            indices[
                i
            ],
        ):

            if neighbor_index == i:
                continue

            if (
                frameworks[
                    neighbor_index
                ]
                ==
                current_framework
            ):
                continue

            selected_errors.append(
                abs_oof_error[
                    neighbor_index
                ]
            )

            selected_distances.append(
                distance
            )

            if (
                len(
                    selected_errors
                )
                >=
                K_NEIGHBORS
            ):

                break


        if len(
            selected_errors
        ) == 0:

            local_mae = np.nan

            local_q90 = np.nan

            nearest_distance = np.nan

        else:

            selected_errors = np.array(
                selected_errors,
                dtype=float,
            )

            local_mae = float(
                np.mean(
                    selected_errors
                )
            )

            local_q90 = float(
                np.quantile(
                    selected_errors,
                    0.90,
                )
            )

            nearest_distance = float(
                selected_distances[
                    0
                ]
            )


        local_mae_list.append(
            local_mae
        )

        local_q90_list.append(
            local_q90
        )

        nearest_distance_list.append(
            nearest_distance
        )

        neighbor_count_list.append(
            len(
                selected_errors
            )
        )


    df[
        "local_voltage_mae"
    ] = local_mae_list

    df[
        "local_voltage_q90"
    ] = local_q90_list

    df[
        "nearest_framework_distance"
    ] = nearest_distance_list

    df[
        "local_neighbor_count"
    ] = neighbor_count_list


    df[
        "voltage_lcb"
    ] = (
        df[
            "voltage_oof_pred"
        ]
        -
        df[
            "local_voltage_q90"
        ]
    )


    return df


def add_confidence_labels(
    df,
):

    uncertainty_low = (
        df[
            "local_voltage_q90"
        ]
        .quantile(
            0.33
        )
    )

    uncertainty_high = (
        df[
            "local_voltage_q90"
        ]
        .quantile(
            0.67
        )
    )

    distance_low = (
        df[
            "nearest_framework_distance"
        ]
        .quantile(
            0.33
        )
    )

    distance_high = (
        df[
            "nearest_framework_distance"
        ]
        .quantile(
            0.67
        )
    )


    confidence_labels = []


    for _, row in df.iterrows():

        uncertainty = (
            row[
                "local_voltage_q90"
            ]
        )

        distance = (
            row[
                "nearest_framework_distance"
            ]
        )


        if (
            uncertainty
            <=
            uncertainty_low
            and
            distance
            <=
            distance_high
        ):

            label = "High"

        elif (
            uncertainty
            <=
            uncertainty_high
            and
            distance
            <=
            distance_high
        ):

            label = "Medium"

        else:

            label = "Low"


        confidence_labels.append(
            label
        )


    df[
        "confidence"
    ] = confidence_labels


    return df


def build_conservative_score(
    df,
):

    df[
        "score_voltage_conservative"
    ] = minmax_scale(
        df[
            "voltage_lcb"
        ],
        higher_is_better=True,
    )

    df[
        "score_capacity_conservative"
    ] = minmax_scale(
        df[
            "capacity_physics"
        ],
        higher_is_better=True,
    )

    df[
        "score_volume_conservative"
    ] = minmax_scale(
        df[
            "endpoint_delta_abs"
        ],
        higher_is_better=False,
    )

    df[
        "score_stability_conservative"
    ] = minmax_scale(
        df[
            "max_stability"
        ],
        higher_is_better=False,
    )


    df[
        "conservative_score"
    ] = (
        0.30
        *
        df[
            "score_voltage_conservative"
        ]
        +
        0.30
        *
        df[
            "score_capacity_conservative"
        ]
        +
        0.20
        *
        df[
            "score_volume_conservative"
        ]
        +
        0.20
        *
        df[
            "score_stability_conservative"
        ]
    )


    return df


def greedy_diverse_shortlist(
    df,
    n_candidates,
):

    selected_indices = []

    used_frameworks = set()

    used_chemistry_pairs = set()


    ranked_df = (
        df.sort_values(
            "conservative_score",
            ascending=False,
        )
        .copy()
    )


    for index, row in ranked_df.iterrows():

        framework = (
            row[
                "framework_formula"
            ]
        )

        chemistry_pair = (
            row[
                "main_transition_metal"
            ],
            row[
                "chemistry_family"
            ],
        )


        if framework in used_frameworks:
            continue


        if chemistry_pair in used_chemistry_pairs:
            continue


        selected_indices.append(
            index
        )

        used_frameworks.add(
            framework
        )

        used_chemistry_pairs.add(
            chemistry_pair
        )


        if (
            len(
                selected_indices
            )
            >=
            n_candidates
        ):

            break


    if (
        len(
            selected_indices
        )
        <
        n_candidates
    ):

        for index, row in ranked_df.iterrows():

            if index in selected_indices:
                continue

            framework = (
                row[
                    "framework_formula"
                ]
            )

            if framework in used_frameworks:
                continue

            selected_indices.append(
                index
            )

            used_frameworks.add(
                framework
            )

            if (
                len(
                    selected_indices
                )
                >=
                n_candidates
            ):

                break


    shortlist = (
        ranked_df.loc[
            selected_indices
        ]
        .copy()
        .sort_values(
            "conservative_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    shortlist[
        "dft_rank"
    ] = np.arange(
        1,
        len(
            shortlist
        ) + 1,
    )


    return shortlist


def main():

    screening_df = pd.read_csv(
        SCREENING_FILE
    )

    pareto_df = pd.read_csv(
        PARETO_FILE
    )

    feature_df = pd.read_csv(
        FEATURE_FILE
    )


    print(
        "Screening candidates:",
        screening_df.shape,
    )

    print(
        "Pareto candidates:",
        pareto_df.shape,
    )


    feature_columns = (
        build_feature_list(
            feature_df
        )
    )


    print(
        "Uncertainty feature dimensions:",
        len(
            feature_columns
        ),
    )


    screening_df = (
        calculate_local_uncertainty(
            screening_df,
            feature_columns,
        )
    )


    screening_df = (
        add_confidence_labels(
            screening_df
        )
    )


    print()
    print(
        "Global OOF voltage error:"
    )

    print(
        screening_df[
            "voltage_oof_error"
        ]
        .abs()
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
    )


    print()
    print(
        "Local uncertainty statistics:"
    )

    print(
        screening_df[
            [
                "local_voltage_mae",
                "local_voltage_q90",
                "nearest_framework_distance",
            ]
        ]
        .describe()
    )


    print()
    print(
        "Confidence distribution:"
    )

    print(
        screening_df[
            "confidence"
        ]
        .value_counts()
    )


    uncertainty_columns = [
        "battery_id",
        "local_voltage_mae",
        "local_voltage_q90",
        "nearest_framework_distance",
        "voltage_lcb",
        "confidence",
    ]


    pareto_df = pareto_df.merge(
        screening_df[
            uncertainty_columns
        ],
        on="battery_id",
        how="left",
    )


    pareto_df[
        "main_transition_metal"
    ] = (
        pareto_df[
            "framework_formula"
        ]
        .apply(
            get_main_transition_metal
        )
    )


    pareto_df[
        "chemistry_family"
    ] = (
        pareto_df[
            "framework_formula"
        ]
        .apply(
            get_chemistry_family
        )
    )


    pareto_df = (
        pareto_df
        .sort_values(
            "composite_score",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "framework_formula"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


    print()
    print(
        "Pareto candidates after framework deduplication:",
        len(
            pareto_df
        ),
    )


    pareto_df = (
        build_conservative_score(
            pareto_df
        )
    )


    conservative_candidates = (
        pareto_df[
            (
                pareto_df[
                    "voltage_lcb"
                ]
                >=
                3.0
            )
            &
            (
                pareto_df[
                    "confidence"
                ]
                !=
                "Low"
            )
        ]
        .copy()
    )


    print(
        "Candidates after uncertainty filtering:",
        len(
            conservative_candidates
        ),
    )


    if len(
        conservative_candidates
    ) < N_DFT_CANDIDATES:

        print(
            "Not enough High/Medium confidence candidates."
        )

        print(
            "Using all Pareto candidates with voltage LCB >= 3.0."
        )

        conservative_candidates = (
            pareto_df[
                pareto_df[
                    "voltage_lcb"
                ]
                >=
                3.0
            ]
            .copy()
        )


    shortlist = (
        greedy_diverse_shortlist(
            conservative_candidates,
            N_DFT_CANDIDATES,
        )
    )


    display_columns = [
        "dft_rank",
        "battery_id",
        "framework_formula",
        "main_transition_metal",
        "chemistry_family",
        "formula_charge",
        "formula_discharge",
        "voltage_oof_pred",
        "local_voltage_q90",
        "voltage_lcb",
        "average_voltage",
        "capacity_physics",
        "endpoint_delta_abs",
        "max_stability",
        "confidence",
        "conservative_score",
    ]


    print()
    print(
        "DFT shortlist:"
    )

    print(
        shortlist[
            display_columns
        ]
        .to_string(
            index=False
        )
    )


    print()
    print(
        "Top conservative Pareto candidates:"
    )

    print(
        pareto_df[
            [
                "battery_id",
                "framework_formula",
                "main_transition_metal",
                "chemistry_family",
                "voltage_oof_pred",
                "local_voltage_q90",
                "voltage_lcb",
                "capacity_physics",
                "endpoint_delta_abs",
                "max_stability",
                "confidence",
                "conservative_score",
            ]
        ]
        .sort_values(
            "conservative_score",
            ascending=False,
        )
        .head(
            20
        )
        .to_string(
            index=False
        )
    )


    OUTPUT_UNCERTAINTY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    screening_df.to_csv(
        OUTPUT_UNCERTAINTY,
        index=False,
    )

    pareto_df.to_csv(
        OUTPUT_PARETO,
        index=False,
    )

    shortlist.to_csv(
        OUTPUT_SHORTLIST,
        index=False,
    )


    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_UNCERTAINTY
    )

    print(
        OUTPUT_PARETO
    )

    print(
        OUTPUT_SHORTLIST
    )

    print()
    print(
        "STEP 24 completed."
    )


if __name__ == "__main__":

    main()