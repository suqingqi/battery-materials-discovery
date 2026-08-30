from pathlib import Path

from pymatgen.core import Structure


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CIF_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "structures_charge"
    / "BAT_01569_mp-689940.cif"
)

PSEUDO_DIR = (
    PROJECT_ROOT
    / "dft"
    / "pseudo"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dft"
    / "co_po4"
    / "scf"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "CoPO4_scf.in"
)


PSEUDO_MAP = {
    "Co":
        "Co.nc.pbe.z_17.oncvpsp4.spms.v1.upf",

    "P":
        "P.us.pbe.z_5.ld1.psl.v1.0.0-high.upf",

    "O":
        "O.paw.pbe.z_6.ld1.psl.v0.1.upf",
}


ATOMIC_MASSES = {
    "Co": 58.933194,
    "P": 30.973762,
    "O": 15.999,
}


ECUTWFC = 55

ECUTRHO = 540

KPOINT_GRID = (
    3,
    3,
    3,
)


def main():

    structure = Structure.from_file(
        CIF_FILE
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    species = sorted(
        {
            site.specie.symbol
            for site in structure
        }
    )

    lines = []

    lines.append(
        "&CONTROL"
    )

    lines.append(
        "  calculation = 'scf',"
    )

    lines.append(
        "  prefix = 'CoPO4',"
    )

    lines.append(
        f"  pseudo_dir = '{PSEUDO_DIR}',"
    )

    lines.append(
        "  outdir = './tmp',"
    )

    lines.append(
        "  verbosity = 'high',"
    )

    lines.append(
        "/"
    )

    lines.append(
        ""
    )

    lines.append(
        "&SYSTEM"
    )

    lines.append(
        "  ibrav = 0,"
    )

    lines.append(
        f"  nat = {len(structure)},"
    )

    lines.append(
        f"  ntyp = {len(species)},"
    )

    lines.append(
        f"  ecutwfc = {ECUTWFC},"
    )

    lines.append(
        f"  ecutrho = {ECUTRHO},"
    )

    lines.append(
        "  nspin = 2,"
    )

    co_index = (
        species.index(
            "Co"
        )
        +
        1
    )

    lines.append(
        f"  starting_magnetization({co_index}) = 0.5,"
    )

    lines.append(
        "  occupations = 'smearing',"
    )

    lines.append(
        "  smearing = 'mv',"
    )

    lines.append(
        "  degauss = 0.01,"
    )

    lines.append(
        "/"
    )

    lines.append(
        ""
    )

    lines.append(
        "&ELECTRONS"
    )

    lines.append(
        "  conv_thr = 1.0d-8,"
    )

    lines.append(
        "  mixing_beta = 0.3,"
    )

    lines.append(
        "  electron_maxstep = 200,"
    )

    lines.append(
        "/"
    )

    lines.append(
        ""
    )

    lines.append(
        "ATOMIC_SPECIES"
    )

    for symbol in species:

        lines.append(
            f"{symbol} "
            f"{ATOMIC_MASSES[symbol]} "
            f"{PSEUDO_MAP[symbol]}"
        )

    lines.append(
        ""
    )

    lines.append(
        "CELL_PARAMETERS angstrom"
    )

    for vector in (
        structure
        .lattice
        .matrix
    ):

        lines.append(
            f"{vector[0]:.12f} "
            f"{vector[1]:.12f} "
            f"{vector[2]:.12f}"
        )

    lines.append(
        ""
    )

    lines.append(
        "ATOMIC_POSITIONS crystal"
    )

    for site in structure:

        symbol = (
            site.specie.symbol
        )

        x, y, z = (
            site.frac_coords
        )

        lines.append(
            f"{symbol} "
            f"{x:.12f} "
            f"{y:.12f} "
            f"{z:.12f}"
        )

    lines.append(
        ""
    )

    lines.append(
        "K_POINTS automatic"
    )

    lines.append(
        f"{KPOINT_GRID[0]} "
        f"{KPOINT_GRID[1]} "
        f"{KPOINT_GRID[2]} "
        "0 0 0"
    )

    OUTPUT_FILE.write_text(
        "\n".join(
            lines
        )
    )

    print(
        "Structure:",
        structure.composition
    )

    print(
        "Atoms:",
        len(
            structure
        ),
    )

    print(
        "Generated:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()