from pathlib import Path

import torch

from pymatgen.core import Structure
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STRUCTURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "structures"
    / "BAT_00001.cif"
)


CUTOFF = 5.0


structure = Structure.from_file(
    STRUCTURE_FILE
)


print("Structure loaded:")
print(structure.composition)

print()
print("Number of atoms:")
print(len(structure))

print()
print("Lattice:")
print(structure.lattice)


node_features = []


for site in structure:

    element = site.specie

    atomic_number = element.Z

    atomic_mass = float(
        element.atomic_mass
    )

    electronegativity = (
        element.X
        if element.X is not None
        else 0.0
    )

    node_features.append(
        [
            atomic_number,
            atomic_mass,
            electronegativity,
        ]
    )


x = torch.tensor(
    node_features,
    dtype=torch.float32,
)


edge_sources = []
edge_targets = []
edge_distances = []


for atom_index, site in enumerate(
    structure
):

    neighbors = structure.get_neighbors(
        site,
        r=CUTOFF,
    )

    for neighbor in neighbors:

        neighbor_index = (
            neighbor.index
        )

        distance = float(
            neighbor.nn_distance
        )

        edge_sources.append(
            atom_index
        )

        edge_targets.append(
            neighbor_index
        )

        edge_distances.append(
            [distance]
        )


edge_index = torch.tensor(
    [
        edge_sources,
        edge_targets,
    ],
    dtype=torch.long,
)


edge_attr = torch.tensor(
    edge_distances,
    dtype=torch.float32,
)


graph = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
)


print()
print("Graph:")
print(graph)

print()
print("Node feature shape:")
print(graph.x.shape)

print()
print("Edge index shape:")
print(graph.edge_index.shape)

print()
print("Edge attribute shape:")
print(graph.edge_attr.shape)


print()
print("First 5 node features:")
print(graph.x[:5])


print()
print("First 10 edges:")
print(
    graph.edge_index[:, :10]
)


print()
print("First 10 distances:")
print(
    graph.edge_attr[:10]
)


print()
print("STEP 11 test completed.")