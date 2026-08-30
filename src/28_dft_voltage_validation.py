from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RY_TO_EV = 13.605693

E_CHARGE_CELL_RY = -1856.45435286
E_DISCHARGE_CELL_RY = -1915.41470790
E_LI_ATOM_RY = -14.47195990

N_FORMULA_UNITS = 4
N_LI = 4

V_ML_OOF = 4.513853
V_MP = 4.301843


E_CHARGE_FU_RY = (
    E_CHARGE_CELL_RY
    / N_FORMULA_UNITS
)

E_DISCHARGE_FU_RY = (
    E_DISCHARGE_CELL_RY
    / N_FORMULA_UNITS
)

DELTA_E_CELL_RY = (
    E_DISCHARGE_CELL_RY
    - E_CHARGE_CELL_RY
    - N_LI * E_LI_ATOM_RY
)

DELTA_E_PER_LI_EV = (
    DELTA_E_CELL_RY
    * RY_TO_EV
    / N_LI
)

V_DFT = -DELTA_E_PER_LI_EV


print(
    f"E(CoPO4 cell)       = "
    f"{E_CHARGE_CELL_RY:.8f} Ry"
)

print(
    f"E(LiCoPO4 cell)     = "
    f"{E_DISCHARGE_CELL_RY:.8f} Ry"
)

print(
    f"E(Li metal atom)    = "
    f"{E_LI_ATOM_RY:.8f} Ry"
)

print()

print(
    f"E(CoPO4 per f.u.)   = "
    f"{E_CHARGE_FU_RY:.8f} Ry"
)

print(
    f"E(LiCoPO4 per f.u.) = "
    f"{E_DISCHARGE_FU_RY:.8f} Ry"
)

print()

print(
    f"Reaction dE cell    = "
    f"{DELTA_E_CELL_RY:.8f} Ry"
)

print(
    f"Reaction dE / Li    = "
    f"{DELTA_E_PER_LI_EV:.6f} eV"
)

print()

print(
    f"DFT-PBE voltage     = "
    f"{V_DFT:.4f} V"
)

print(
    f"ML OOF voltage      = "
    f"{V_ML_OOF:.4f} V"
)

print(
    f"MP reference        = "
    f"{V_MP:.4f} V"
)

print()

print(
    f"DFT vs MP error     = "
    f"{V_DFT - V_MP:+.4f} V"
)

print(
    f"ML vs MP error      = "
    f"{V_ML_OOF - V_MP:+.4f} V"
)


result = pd.DataFrame(
    {
        "method": [
            "DFT-PBE",
            "ML-OOF",
            "Materials Project",
        ],
        "voltage_V": [
            V_DFT,
            V_ML_OOF,
            V_MP,
        ],
        "error_vs_MP_V": [
            V_DFT - V_MP,
            V_ML_OOF - V_MP,
            0.0,
        ],
    }
)

output_path = (
    OUTPUT_DIR
    / "dft_voltage_validation.csv"
)

result.to_csv(
    output_path,
    index=False,
)

print()
print(result)

print()
print(
    f"Saved: "
    f"{output_path}"
)