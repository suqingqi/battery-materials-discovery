from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


stages = [
    "Modeling dataset",
    "Endpoint structures available",
    "Physics + stability constraints",
    "Pareto-optimal candidates",
    "Framework deduplication",
    "Uncertainty-filtered candidates",
    "DFT shortlist",
    "DFT validated",
]

counts = [
    1858,
    1855,
    629,
    69,
    53,
    20,
    4,
    1,
]


fig, ax = plt.subplots(
    figsize=(9, 5.8)
)

bars = ax.barh(
    stages,
    counts,
)

ax.invert_yaxis()


for bar, count in zip(
    bars,
    counts,
):
    ax.text(
        bar.get_width() + 15,
        bar.get_y() + bar.get_height() / 2,
        f"{count:,}",
        va="center",
        fontsize=10,
    )


ax.set_xlabel(
    "Number of Candidate Records"
)

ax.set_title(
    "Battery Electrode Discovery Pipeline: "
    "From Database to DFT Validation"
)

ax.set_xscale("log")

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3,
)


ax.text(
    0.5,
    -0.13,
    (
        "Filtering combines application-domain constraints, "
        "physics-based capacity and volume metrics, "
        "Pareto optimization, framework-aware uncertainty, "
        "and first-principles validation."
    ),
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=9,
)


fig.tight_layout()


output_path = (
    OUTPUT_DIR
    / "candidate_screening_funnel.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print("Pipeline counts:")

for stage, count in zip(
    stages,
    counts,
):
    print(
        f"{stage}: {count}"
    )

print()

print(
    f"Saved: {output_path}"
)