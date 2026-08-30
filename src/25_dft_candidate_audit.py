from pathlib import Path

import pandas as pd

from pymatgen.core import Structure


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHORTLIST_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "dft_shortlist.csv"
)

ENDPOINT_FILE = (
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
    / "results"
    / "metrics"
    / "dft_candidate_audit.csv"
)


def inspect_structure(
    file_path,
):

    structure = Structure.from_file(
        file_path
    )

    return {
        "formula":
            structure.composition
            .reduced_formula,

        "num_atoms":
            len(
                structure
            ),

        "volume":
            float(
                structure.volume
            ),

        "a":
            float(
                structure.lattice.a
            ),

        "b":
            float(
                structure.lattice.b
            ),

        "c":
            float(
                structure.lattice.c
            ),
    }


def main():

    shortlist = pd.read_csv(
        SHORTLIST_FILE
    )

    endpoint_df = pd.read_csv(
        ENDPOINT_FILE
    )

    print(
        "Shortlist shape:",
        shortlist.shape,
    )

    print(
        "Endpoint index shape:",
        endpoint_df.shape,
    )

    print()

    required_shortlist_columns = [
        "battery_id",
        "dft_rank",
        "framework_formula",
        "id_charge",
        "id_discharge",
        "voltage_oof_pred",
        "voltage_lcb",
        "capacity_physics",
        "endpoint_delta_abs",
        "confidence",
    ]

    missing_columns = [
        column
        for column in required_shortlist_columns
        if column not in shortlist.columns
    ]

    if missing_columns:

        raise KeyError(
            f"Missing shortlist columns: {missing_columns}"
        )

    endpoint_subset = (
        endpoint_df[
            [
                "battery_id",
                "charge_file",
                "discharge_file",
            ]
        ]
        .copy()
    )

    df = shortlist.merge(
        endpoint_subset,
        on="battery_id",
        how="left",
        validate="one_to_one",
    )

    print(
        "Merged dataset:",
        df.shape,
    )

    print()

    print(
        "Missing charge files:",
        df[
            "charge_file"
        ]
        .isna()
        .sum(),
    )

    print(
        "Missing discharge files:",
        df[
            "discharge_file"
        ]
        .isna()
        .sum(),
    )

    print()

    rows = []

    for _, row in df.iterrows():

        battery_id = str(
            row[
                "battery_id"
            ]
        )

        charge_file_name = (
            row[
                "charge_file"
            ]
        )

        discharge_file_name = (
            row[
                "discharge_file"
            ]
        )

        if pd.isna(
            charge_file_name
        ):

            raise FileNotFoundError(
                f"Missing charge CIF for {battery_id}"
            )

        if pd.isna(
            discharge_file_name
        ):

            raise FileNotFoundError(
                f"Missing discharge CIF for {battery_id}"
            )

        charge_file = (
            CHARGE_DIR
            /
            str(
                charge_file_name
            )
        )

        discharge_file = (
            DISCHARGE_DIR
            /
            str(
                discharge_file_name
            )
        )

        if not charge_file.exists():

            raise FileNotFoundError(
                f"Charge CIF not found: {charge_file}"
            )

        if not discharge_file.exists():

            raise FileNotFoundError(
                f"Discharge CIF not found: {discharge_file}"
            )

        charge = inspect_structure(
            charge_file
        )

        discharge = inspect_structure(
            discharge_file
        )

        rows.append(
            {
                "dft_rank":
                    int(
                        row[
                            "dft_rank"
                        ]
                    ),

                "battery_id":
                    battery_id,

                "framework_formula":
                    row[
                        "framework_formula"
                    ],

                "charge_mp_id":
                    row[
                        "id_charge"
                    ],

                "discharge_mp_id":
                    row[
                        "id_discharge"
                    ],

                "charge_formula":
                    charge[
                        "formula"
                    ],

                "discharge_formula":
                    discharge[
                        "formula"
                    ],

                "charge_atoms":
                    charge[
                        "num_atoms"
                    ],

                "discharge_atoms":
                    discharge[
                        "num_atoms"
                    ],

                "charge_volume":
                    charge[
                        "volume"
                    ],

                "discharge_volume":
                    discharge[
                        "volume"
                    ],

                "charge_a":
                    charge[
                        "a"
                    ],

                "charge_b":
                    charge[
                        "b"
                    ],

                "charge_c":
                    charge[
                        "c"
                    ],

                "discharge_a":
                    discharge[
                        "a"
                    ],

                "discharge_b":
                    discharge[
                        "b"
                    ],

                "discharge_c":
                    discharge[
                        "c"
                    ],

                "voltage_ml":
                    row[
                        "voltage_oof_pred"
                    ],

                "voltage_lcb":
                    row[
                        "voltage_lcb"
                    ],

                "capacity":
                    row[
                        "capacity_physics"
                    ],

                "volume_change":
                    row[
                        "endpoint_delta_abs"
                    ],

                "confidence":
                    row[
                        "confidence"
                    ],
            }
        )

    result_df = pd.DataFrame(
        rows
    )

    result_df = (
        result_df
        .sort_values(
            "dft_rank"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "DFT candidate audit:"
    )

    print()

    print(
        result_df[
            [
                "dft_rank",
                "battery_id",
                "framework_formula",
                "charge_formula",
                "discharge_formula",
                "charge_atoms",
                "discharge_atoms",
                "charge_mp_id",
                "discharge_mp_id",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "Atom statistics:"
    )

    print(
        result_df[
            [
                "charge_atoms",
                "discharge_atoms",
            ]
        ]
        .describe()
    )

    print()

    print(
        "Candidate details:"
    )

    print(
        result_df[
            [
                "dft_rank",
                "framework_formula",
                "voltage_ml",
                "voltage_lcb",
                "capacity",
                "volume_change",
                "confidence",
            ]
        ]
        .to_string(
            index=False
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
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
        "STEP 25 candidate audit completed."
    )


if __name__ == "__main__":

    main()