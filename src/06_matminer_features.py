from pathlib import Path

import pandas as pd

from pymatgen.core import Composition

from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_model.csv"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "matminer_batches"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_features.csv"
)


BATCH_SIZE = 100


def main():

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_FILE)

    print("Input shape:", df.shape)
    print("Batch size:", BATCH_SIZE)

    total_rows = len(df)

    total_batches = (
        total_rows + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print("Total batches:", total_batches)

    magpie = ElementProperty.from_preset(
        "magpie"
    )

    stoichiometry = Stoichiometry()

    valence = ValenceOrbital(
        props=["avg"]
    )

    magpie.set_n_jobs(1)
    stoichiometry.set_n_jobs(1)
    valence.set_n_jobs(1)

    for batch_index in range(total_batches):

        start = (
            batch_index
            * BATCH_SIZE
        )

        end = min(
            start + BATCH_SIZE,
            total_rows,
        )

        batch_number = (
            batch_index + 1
        )

        batch_file = (
            INTERIM_DIR
            / f"batch_{batch_number:03d}.csv"
        )

        if batch_file.exists():

            print()
            print(
                f"Batch "
                f"{batch_number}/{total_batches} "
                f"already exists. Skipping."
            )

            continue

        print()
        print(
            f"Processing batch "
            f"{batch_number}/{total_batches}"
        )

        print(
            f"Rows: {start} to {end - 1}"
        )

        batch_df = (
            df.iloc[start:end]
            .copy()
            .reset_index(drop=True)
        )

        batch_df["composition"] = (
            batch_df[
                "framework_formula"
            ]
            .apply(Composition)
        )

        print(
            "Generating Magpie features..."
        )

        batch_df = (
            magpie.featurize_dataframe(
                batch_df,
                col_id="composition",
                ignore_errors=True,
                pbar=False,
            )
        )

        print(
            "Generating stoichiometry features..."
        )

        batch_df = (
            stoichiometry.featurize_dataframe(
                batch_df,
                col_id="composition",
                ignore_errors=True,
                pbar=False,
            )
        )

        print(
            "Generating valence features..."
        )

        batch_df = (
            valence.featurize_dataframe(
                batch_df,
                col_id="composition",
                ignore_errors=True,
                pbar=False,
            )
        )

        batch_df = batch_df.drop(
            columns=["composition"]
        )

        batch_df.to_csv(
            batch_file,
            index=False,
        )

        print(
            f"Saved: {batch_file.name}"
        )

    print()
    print("Combining all batches...")

    batch_files = sorted(
        INTERIM_DIR.glob(
            "batch_*.csv"
        )
    )

    if len(batch_files) != total_batches:

        raise RuntimeError(
            f"Expected {total_batches} "
            f"batch files, "
            f"but found {len(batch_files)}."
        )

    all_batches = []

    for batch_file in batch_files:

        batch_df = pd.read_csv(
            batch_file
        )

        all_batches.append(
            batch_df
        )

    final_df = pd.concat(
        all_batches,
        ignore_index=True,
    )

    print(
        "Combined shape:",
        final_df.shape,
    )

    if len(final_df) != len(df):

        raise RuntimeError(
            "Final row count does not "
            "match input row count."
        )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    original_columns = set(
        df.columns
    )

    new_columns = [
        column
        for column in final_df.columns
        if column not in original_columns
    ]

    print()
    print(
        "Generated matminer features:",
        len(new_columns),
    )

    numeric_new_features = (
        final_df[
            new_columns
        ]
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    print(
        "Numeric matminer features:",
        len(numeric_new_features),
    )

    missing_counts = (
        final_df[
            numeric_new_features
        ]
        .isna()
        .sum()
    )

    missing_counts = (
        missing_counts[
            missing_counts > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    print()
    print(
        "Features with missing values:"
    )

    if len(missing_counts) == 0:

        print("None")

    else:

        print(
            missing_counts.head(30)
        )

    print()
    print(
        "Saved final feature dataset:"
    )

    print(OUTPUT_FILE)

    print()
    print("STEP 06 completed.")


if __name__ == "__main__":
    main()