import os
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
STRUCTURE_DIR = RAW_DIR / "structures"

RAW_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)


api_key = os.getenv("MP_API_KEY")

if not api_key:
    raise RuntimeError("MP_API_KEY was not found.")


fields = [
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
    "num_steps",
    "nelements",
    "chemsys",
    "elements",
    "id_charge",
    "id_discharge",
    "material_ids",
    "host_structure",
    "warnings",
]


print("Downloading Li insertion electrode data...")


with MPRester(api_key) as mpr:
    docs = mpr.materials.insertion_electrodes.search(
        working_ion="Li",
        fields=fields,
    )


print("Downloaded documents:", len(docs))


rows = []
structure_count = 0


for index, doc in enumerate(docs):

    battery_id = f"BAT_{index + 1:05d}"

    elements = None

    if doc.elements is not None:
        elements = ",".join(str(element) for element in doc.elements)

    material_ids = None

    if doc.material_ids is not None:
        material_ids = ",".join(str(mp_id) for mp_id in doc.material_ids)

    warnings = None

    if doc.warnings is not None:
        warnings = " | ".join(str(item) for item in doc.warnings)

    row = {
        "battery_id": battery_id,
        "battery_formula": doc.battery_formula,
        "framework_formula": doc.framework_formula,
        "formula_charge": doc.formula_charge,
        "formula_discharge": doc.formula_discharge,
        "working_ion": str(doc.working_ion),
        "average_voltage": doc.average_voltage,
        "capacity_grav": doc.capacity_grav,
        "capacity_vol": doc.capacity_vol,
        "energy_grav": doc.energy_grav,
        "energy_vol": doc.energy_vol,
        "max_delta_volume": doc.max_delta_volume,
        "stability_charge": doc.stability_charge,
        "stability_discharge": doc.stability_discharge,
        "num_steps": doc.num_steps,
        "nelements": doc.nelements,
        "chemsys": doc.chemsys,
        "elements": elements,
        "id_charge": str(doc.id_charge),
        "id_discharge": str(doc.id_discharge),
        "material_ids": material_ids,
        "warnings": warnings,
    }

    rows.append(row)

    if doc.host_structure is not None:

        cif_path = STRUCTURE_DIR / f"{battery_id}.cif"

        doc.host_structure.to(
            filename=str(cif_path),
            fmt="cif",
        )

        structure_count += 1


df = pd.DataFrame(rows)


csv_path = RAW_DIR / "li_insertion_electrodes_raw.csv"

df.to_csv(
    csv_path,
    index=False,
)


print()
print("Dataset shape:", df.shape)
print("Saved CSV:", csv_path)
print("Saved structures:", structure_count)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("First 5 rows:")
print(
    df[
        [
            "battery_id",
            "battery_formula",
            "average_voltage",
            "capacity_grav",
            "max_delta_volume",
        ]
    ].head()
)

print()
print("Missing values:")
print(
    df[
        [
            "average_voltage",
            "capacity_grav",
            "max_delta_volume",
            "stability_charge",
            "stability_discharge",
        ]
    ].isna().sum()
)

print()
print("Download completed.")