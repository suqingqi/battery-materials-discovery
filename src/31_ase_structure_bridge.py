from pathlib import Path

from ase.io import write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CIF_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "structures_charge"
    / "BAT_01569_mp-689940.cif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ase_bridge"
)

ASE_OUTPUT_FILE = (
    OUTPUT_DIR
    / "CoPO4_from_ase.xyz"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CIF_FILE.exists():
        raise FileNotFoundError(
            f"CIF file not found:\n{CIF_FILE}"
        )

    structure = Structure.from_file(
        CIF_FILE
    )

    adaptor = AseAtomsAdaptor()

    atoms = adaptor.get_atoms(
        structure
    )

    restored_structure = adaptor.get_structure(
        atoms
    )

    print(
        "Original pymatgen structure:"
    )

    print(
        structure.composition
    )

    print()

    print(
        "ASE Atoms:"
    )

    print(
        atoms
    )

    print()

    print(
        "Number of atoms:"
    )

    print(
        len(atoms)
    )

    print()

    print(
        "Chemical formula:"
    )

    print(
        atoms.get_chemical_formula()
    )

    print()

    print(
        "Cell:"
    )

    print(
        atoms.cell
    )

    print()

    print(
        "Periodic boundary conditions:"
    )

    print(
        atoms.pbc
    )

    print()

    print(
        "Restored pymatgen structure:"
    )

    print(
        restored_structure.composition
    )

    same_atom_count = (
        len(structure)
        == len(restored_structure)
    )

    same_formula = (
        structure.composition.reduced_formula
        == restored_structure.composition.reduced_formula
    )

    print()

    print(
        "Atom count preserved:"
    )

    print(
        same_atom_count
    )

    print()

    print(
        "Formula preserved:"
    )

    print(
        same_formula
    )

    write(
        ASE_OUTPUT_FILE,
        atoms,
    )

    print()

    print(
        "ASE structure written to:"
    )

    print(
        ASE_OUTPUT_FILE
    )

    print()

    if (
        same_atom_count
        and same_formula
    ):
        print(
            "pymatgen -> ASE -> pymatgen conversion passed."
        )
    else:
        raise RuntimeError(
            "Structure conversion validation failed."
        )


if __name__ == "__main__":
    main()