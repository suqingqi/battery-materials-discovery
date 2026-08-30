from pathlib import Path
import os
import time

import pandas as pd

from mp_api.client import MPRester


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_model.csv"
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

INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "endpoint_structure_index.csv"
)


BATCH_SIZE = 100

SLEEP_SECONDS = 1.0


def save_structure(
    structure,
    output_file,
):

    structure.to(
        filename=str(
            output_file
        ),
        fmt="cif",
    )


def download_batch(
    mpr,
    material_ids,
):

    docs = (
        mpr
        .materials
        .summary
        .search(
            material_ids=material_ids,
            fields=[
                "material_id",
                "structure",
            ],
        )
    )

    structure_map = {}

    for doc in docs:

        material_id = str(
            doc.material_id
        )

        structure_map[
            material_id
        ] = (
            doc.structure
        )

    return structure_map


def main():

    CHARGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DISCHARGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Dataset shape:",
        df.shape,
    )

    api_key = os.getenv(
        "MP_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "MP_API_KEY is not available."
        )

    endpoint_records = []

    all_ids = set()

    for _, row in df.iterrows():

        all_ids.add(
            str(
                row[
                    "id_charge"
                ]
            )
        )

        all_ids.add(
            str(
                row[
                    "id_discharge"
                ]
            )
        )

    all_ids = sorted(
        all_ids
    )

    print(
        "Unique endpoint IDs:",
        len(
            all_ids
        ),
    )

    existing_ids = set()

    for _, row in df.iterrows():

        battery_id = str(
            row[
                "battery_id"
            ]
        )

        charge_id = str(
            row[
                "id_charge"
            ]
        )

        discharge_id = str(
            row[
                "id_discharge"
            ]
        )

        charge_file = (
            CHARGE_DIR
            / f"{battery_id}_{charge_id}.cif"
        )

        discharge_file = (
            DISCHARGE_DIR
            / f"{battery_id}_{discharge_id}.cif"
        )

        if charge_file.exists():

            existing_ids.add(
                charge_id
            )

        if discharge_file.exists():

            existing_ids.add(
                discharge_id
            )

    ids_to_download = [
        material_id
        for material_id in all_ids
        if material_id
        not in existing_ids
    ]

    print(
        "Already available IDs:",
        len(
            existing_ids
        ),
    )

    print(
        "IDs to download:",
        len(
            ids_to_download
        ),
    )

    downloaded_structures = {}

    with MPRester(
        api_key
    ) as mpr:

        total_batches = (
            len(
                ids_to_download
            )
            +
            BATCH_SIZE
            -
            1
        ) // BATCH_SIZE

        for batch_number, start in enumerate(
            range(
                0,
                len(
                    ids_to_download
                ),
                BATCH_SIZE,
            ),
            start=1,
        ):

            batch_ids = (
                ids_to_download[
                    start:
                    start + BATCH_SIZE
                ]
            )

            print()
            print(
                "Batch:",
                batch_number,
                "/",
                total_batches,
            )

            print(
                "Requested IDs:",
                len(
                    batch_ids
                ),
            )

            try:

                structure_map = (
                    download_batch(
                        mpr,
                        batch_ids,
                    )
                )

                downloaded_structures.update(
                    structure_map
                )

                print(
                    "Received:",
                    len(
                        structure_map
                    ),
                )

            except Exception as error:

                print(
                    "Batch failed:"
                )

                print(
                    type(
                        error
                    ).__name__,
                    str(
                        error
                    ),
                )

            time.sleep(
                SLEEP_SECONDS
            )

    success_charge = 0

    success_discharge = 0

    missing_charge = 0

    missing_discharge = 0

    for _, row in df.iterrows():

        battery_id = str(
            row[
                "battery_id"
            ]
        )

        charge_id = str(
            row[
                "id_charge"
            ]
        )

        discharge_id = str(
            row[
                "id_discharge"
            ]
        )

        charge_file = (
            CHARGE_DIR
            / f"{battery_id}_{charge_id}.cif"
        )

        discharge_file = (
            DISCHARGE_DIR
            / f"{battery_id}_{discharge_id}.cif"
        )

        charge_status = None
        discharge_status = None

        if charge_file.exists():

            charge_status = (
                "existing"
            )

            success_charge += 1

        elif charge_id in downloaded_structures:

            save_structure(
                downloaded_structures[
                    charge_id
                ],
                charge_file,
            )

            charge_status = (
                "downloaded"
            )

            success_charge += 1

        else:

            charge_status = (
                "missing"
            )

            missing_charge += 1

        if discharge_file.exists():

            discharge_status = (
                "existing"
            )

            success_discharge += 1

        elif discharge_id in downloaded_structures:

            save_structure(
                downloaded_structures[
                    discharge_id
                ],
                discharge_file,
            )

            discharge_status = (
                "downloaded"
            )

            success_discharge += 1

        else:

            discharge_status = (
                "missing"
            )

            missing_discharge += 1

        endpoint_records.append(
            {
                "battery_id":
                    battery_id,
                "framework_formula":
                    row[
                        "framework_formula"
                    ],
                "id_charge":
                    charge_id,
                "id_discharge":
                    discharge_id,
                "charge_file":
                    charge_file.name
                    if charge_file.exists()
                    else None,
                "discharge_file":
                    discharge_file.name
                    if discharge_file.exists()
                    else None,
                "charge_status":
                    charge_status,
                "discharge_status":
                    discharge_status,
            }
        )

    index_df = pd.DataFrame(
        endpoint_records
    )

    index_df.to_csv(
        INDEX_FILE,
        index=False,
    )

    print()
    print(
        "Final summary"
    )

    print(
        "Charge structures available:",
        success_charge,
    )

    print(
        "Discharge structures available:",
        success_discharge,
    )

    print(
        "Missing charge structures:",
        missing_charge,
    )

    print(
        "Missing discharge structures:",
        missing_discharge,
    )

    print()
    print(
        "Charge directory:"
    )

    print(
        CHARGE_DIR
    )

    print()
    print(
        "Discharge directory:"
    )

    print(
        DISCHARGE_DIR
    )

    print()
    print(
        "Index file:"
    )

    print(
        INDEX_FILE
    )

    print()
    print(
        "STEP 20 completed."
    )


if __name__ == "__main__":

    main()