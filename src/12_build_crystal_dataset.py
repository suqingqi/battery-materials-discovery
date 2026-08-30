from pathlib import Path

import pandas as pd
import torch

from pymatgen.core import Structure
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "li_cathode_model.csv"
)

STRUCTURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "structures"
)

GRAPH_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graphs"
)

INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph_dataset_index.csv"
)


CUTOFF = 5.0

MAX_ATOMS = 300

MAX_EDGES = 30000

REPORT_INTERVAL = 25


def build_graph(
    structure,
    row,
):

    if len(structure) > MAX_ATOMS:
        raise ValueError(
            f"Too many atoms: {len(structure)}"
        )

    atomic_numbers = []

    node_features = []

    for site in structure:

        element = site.specie

        atomic_numbers.append(
            int(element.Z)
        )

        atomic_mass = float(
            element.atomic_mass
        )

        electronegativity = (
            float(element.X)
            if element.X is not None
            else 0.0
        )

        node_features.append(
            [
                atomic_mass,
                electronegativity,
            ]
        )

    z = torch.tensor(
        atomic_numbers,
        dtype=torch.long,
    )

    x = torch.tensor(
        node_features,
        dtype=torch.float32,
    )

    (
        center_indices,
        neighbor_indices,
        offsets,
        distances,
    ) = structure.get_neighbor_list(
        r=CUTOFF
    )

    num_edges = len(
        center_indices
    )

    if num_edges == 0:
        raise ValueError(
            "No neighbor edges found."
        )

    if num_edges > MAX_EDGES:
        raise ValueError(
            f"Too many edges: {num_edges}"
        )

    edge_index = torch.tensor(
        [
            center_indices,
            neighbor_indices,
        ],
        dtype=torch.long,
    )

    edge_attr = torch.tensor(
        distances.reshape(-1, 1),
        dtype=torch.float32,
    )

    y_voltage = torch.tensor(
        [
            float(
                row[
                    "average_voltage"
                ]
            )
        ],
        dtype=torch.float32,
    )

    y_capacity = torch.tensor(
        [
            float(
                row[
                    "capacity_grav"
                ]
            )
        ],
        dtype=torch.float32,
    )

    y_volume = torch.tensor(
        [
            float(
                row[
                    "max_delta_volume"
                ]
            )
        ],
        dtype=torch.float32,
    )

    graph = Data(
        z=z,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_voltage=y_voltage,
        y_capacity=y_capacity,
        y_volume=y_volume,
    )

    graph.battery_id = str(
        row[
            "battery_id"
        ]
    )

    graph.framework_formula = str(
        row[
            "framework_formula"
        ]
    )

    graph.num_atoms_original = int(
        len(structure)
    )

    return graph


def main():

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        "Dataset shape:",
        df.shape,
    )

    print(
        "Cutoff:",
        CUTOFF,
    )

    print(
        "Max atoms:",
        MAX_ATOMS,
    )

    print(
        "Max edges:",
        MAX_EDGES,
    )

    index_rows = []

    success_count = 0

    skipped_count = 0

    error_count = 0

    total_samples = len(df)

    for row_number, row in df.iterrows():

        battery_id = str(
            row[
                "battery_id"
            ]
        )

        framework_formula = str(
            row[
                "framework_formula"
            ]
        )

        cif_file = (
            STRUCTURE_DIR
            / f"{battery_id}.cif"
        )

        graph_file = (
            GRAPH_DIR
            / f"{battery_id}.pt"
        )

        if graph_file.exists():

            skipped_count += 1

            try:

                graph = torch.load(
                    graph_file,
                    weights_only=False,
                )

                num_atoms = int(
                    graph.num_nodes
                )

                num_edges = int(
                    graph.num_edges
                )

            except Exception:

                num_atoms = None

                num_edges = None

            index_rows.append(
                {
                    "battery_id":
                        battery_id,
                    "framework_formula":
                        framework_formula,
                    "graph_file":
                        graph_file.name,
                    "status":
                        "existing",
                    "num_atoms":
                        num_atoms,
                    "num_edges":
                        num_edges,
                }
            )

        elif not cif_file.exists():

            print()
            print(
                "Missing CIF:",
                battery_id,
            )

            error_count += 1

            index_rows.append(
                {
                    "battery_id":
                        battery_id,
                    "framework_formula":
                        framework_formula,
                    "graph_file":
                        None,
                    "status":
                        "missing_cif",
                    "num_atoms":
                        None,
                    "num_edges":
                        None,
                }
            )

        else:

            try:

                structure = (
                    Structure.from_file(
                        cif_file
                    )
                )

                graph = build_graph(
                    structure,
                    row,
                )

                torch.save(
                    graph,
                    graph_file,
                )

                success_count += 1

                index_rows.append(
                    {
                        "battery_id":
                            battery_id,
                        "framework_formula":
                            framework_formula,
                        "graph_file":
                            graph_file.name,
                        "status":
                            "success",
                        "num_atoms":
                            int(
                                graph.num_nodes
                            ),
                        "num_edges":
                            int(
                                graph.num_edges
                            ),
                    }
                )

            except Exception as error:

                print()
                print(
                    "Graph failed:",
                    battery_id,
                )

                print(
                    type(error).__name__,
                    str(error),
                )

                error_count += 1

                index_rows.append(
                    {
                        "battery_id":
                            battery_id,
                        "framework_formula":
                            framework_formula,
                        "graph_file":
                            None,
                        "status":
                            "error",
                        "num_atoms":
                            None,
                        "num_edges":
                            None,
                    }
                )

        processed = (
            row_number + 1
        )

        if (
            processed
            % REPORT_INTERVAL
            == 0
            or
            processed
            == total_samples
        ):

            print()
            print(
                "Processed:",
                processed,
                "/",
                total_samples,
            )

            print(
                "New graphs:",
                success_count,
            )

            print(
                "Existing graphs:",
                skipped_count,
            )

            print(
                "Errors:",
                error_count,
            )

            current_index_df = pd.DataFrame(
                index_rows
            )

            current_index_df.to_csv(
                INDEX_FILE,
                index=False,
            )

    index_df = pd.DataFrame(
        index_rows
    )

    index_df.to_csv(
        INDEX_FILE,
        index=False,
    )

    valid_index = (
        index_df[
            index_df[
                "status"
            ].isin(
                [
                    "success",
                    "existing",
                ]
            )
        ]
        .copy()
    )

    print()
    print("Final summary")

    print(
        "Total samples:",
        total_samples,
    )

    print(
        "New graphs:",
        success_count,
    )

    print(
        "Existing graphs:",
        skipped_count,
    )

    print(
        "Errors:",
        error_count,
    )

    print(
        "Valid graphs:",
        len(valid_index),
    )

    if len(valid_index) > 0:

        print()
        print(
            "Atom count statistics:"
        )

        print(
            valid_index[
                "num_atoms"
            ].describe()
        )

        print()
        print(
            "Edge count statistics:"
        )

        print(
            valid_index[
                "num_edges"
            ].describe()
        )

        print()
        print(
            "Largest structures:"
        )

        print(
            valid_index[
                [
                    "battery_id",
                    "framework_formula",
                    "num_atoms",
                    "num_edges",
                ]
            ]
            .sort_values(
                "num_atoms",
                ascending=False,
            )
            .head(10)
        )

        print()
        print(
            "Largest graphs:"
        )

        print(
            valid_index[
                [
                    "battery_id",
                    "framework_formula",
                    "num_atoms",
                    "num_edges",
                ]
            ]
            .sort_values(
                "num_edges",
                ascending=False,
            )
            .head(10)
        )

    if error_count > 0:

        print()
        print(
            "Failed samples:"
        )

        print(
            index_df.loc[
                index_df[
                    "status"
                ].isin(
                    [
                        "error",
                        "missing_cif",
                    ]
                ),
                [
                    "battery_id",
                    "framework_formula",
                    "status",
                ],
            ]
        )

    print()
    print(
        "Graph directory:"
    )

    print(
        GRAPH_DIR
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
        "STEP 12 completed."
    )


if __name__ == "__main__":
    main()