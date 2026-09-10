from pathlib import Path

import pandas as pd
from ase.io import read


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COPO4_OUTPUT = (
    PROJECT_ROOT
    / "dft"
    / "co_po4"
    / "relax"
    / "CoPO4_final_scf.out"
)

LICOPO4_OUTPUT = (
    PROJECT_ROOT
    / "dft"
    / "li_copo4"
    / "relax"
    / "LiCoPO4_final_scf.out"
)

LI_METAL_OUTPUT = (
    PROJECT_ROOT
    / "dft"
    / "li_metal"
    / "kpoint"
    / "Li_k14.out"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "qe_parser"
)

RESULT_FILE = (
    RESULT_DIR
    / "qe_energy_summary.csv"
)

VOLTAGE_FILE = (
    RESULT_DIR
    / "qe_voltage_validation.csv"
)

N_LI = 4

V_ML_OOF = 4.513853
V_MP = 4.301843


def check_qe_finished(output_file):

    text = output_file.read_text(
        errors="ignore"
    )

    return "JOB DONE." in text


def read_qe_result(
    name,
    output_file,
):

    if not output_file.exists():
        raise FileNotFoundError(
            f"QE output not found:\n{output_file}"
        )

    finished = check_qe_finished(
        output_file
    )

    atoms = read(
        output_file,
        format="espresso-out",
        index=-1,
    )

    energy_ev = (
        atoms.get_potential_energy()
    )

    formula = (
        atoms.get_chemical_formula()
    )

    atom_count = len(
        atoms
    )

    volume = (
        atoms.get_volume()
        if atoms.cell.volume > 0
        else None
    )

    return {
        "system": name,
        "formula": formula,
        "n_atoms": atom_count,
        "energy_eV": energy_ev,
        "volume_A3": volume,
        "job_done": finished,
        "output_file": str(output_file),
    }


def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    systems = [
        (
            "CoPO4",
            COPO4_OUTPUT,
        ),
        (
            "LiCoPO4",
            LICOPO4_OUTPUT,
        ),
        (
            "Li_metal",
            LI_METAL_OUTPUT,
        ),
    ]

    records = []

    for name, output_file in systems:

        result = read_qe_result(
            name,
            output_file,
        )

        records.append(
            result
        )

        print()
        print(
            f"System: {name}"
        )

        print(
            f"Formula: {result['formula']}"
        )

        print(
            f"Atoms: {result['n_atoms']}"
        )

        print(
            f"Energy: {result['energy_eV']:.8f} eV"
        )

        print(
            f"JOB DONE: {result['job_done']}"
        )

    energy_df = pd.DataFrame(
        records
    )

    energy_df.to_csv(
        RESULT_FILE,
        index=False,
    )

    energy_map = {
        row["system"]: row["energy_eV"]
        for row in records
    }

    e_charge = (
        energy_map["CoPO4"]
    )

    e_discharge = (
        energy_map["LiCoPO4"]
    )

    e_li = (
        energy_map["Li_metal"]
    )

    delta_e_ev = (
        e_discharge
        - e_charge
        - N_LI * e_li
    )

    delta_e_per_li_ev = (
        delta_e_ev
        / N_LI
    )

    voltage_dft = (
        -delta_e_per_li_ev
    )

    print()
    print(
        "Battery voltage calculation"
    )

    print(
        f"E(CoPO4 cell)   = {e_charge:.8f} eV"
    )

    print(
        f"E(LiCoPO4 cell) = {e_discharge:.8f} eV"
    )

    print(
        f"E(Li metal)     = {e_li:.8f} eV"
    )

    print()

    print(
        f"Reaction dE     = {delta_e_ev:.8f} eV"
    )

    print(
        f"dE per Li       = {delta_e_per_li_ev:.8f} eV"
    )

    print()

    print(
        f"DFT voltage     = {voltage_dft:.4f} V"
    )

    print(
        f"ML OOF voltage  = {V_ML_OOF:.4f} V"
    )

    print(
        f"MP reference    = {V_MP:.4f} V"
    )

    voltage_df = pd.DataFrame(
        {
            "method": [
                "DFT-PBE automated",
                "ML-OOF",
                "Materials Project",
            ],
            "voltage_V": [
                voltage_dft,
                V_ML_OOF,
                V_MP,
            ],
            "error_vs_MP_V": [
                voltage_dft - V_MP,
                V_ML_OOF - V_MP,
                0.0,
            ],
        }
    )

    voltage_df.to_csv(
        VOLTAGE_FILE,
        index=False,
    )

    print()
    print(
        voltage_df
    )

    print()

    print(
        "Energy summary saved to:"
    )

    print(
        RESULT_FILE
    )

    print()

    print(
        "Voltage validation saved to:"
    )

    print(
        VOLTAGE_FILE
    )


if __name__ == "__main__":
    main()