"""Alternative CTV citrus map: central/south India split by strain class.

This version uses the same source-noted demo CTV records but presents them as
a focused surveillance map:
    - central and south India zoom
    - two panels for Mild and Severe strain classes
    - cultivar groups shown by marker shape
    - state boundaries from GADM ADM1

Run:
    python3 india_ctv_citrus_map_v2.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from india_ctv_citrus_disease_layer_map import (
    CULTIVAR_MARKERS,
    STRAIN_COLORS,
    make_demo_ctv_records,
)
from india_real_boundary_map import ADM1_BASENAME, ensure_gadm_files, read_adm1_polygons


OUTPUT_DIR = Path("india_ctv_citrus_map_v2_output")

FOCUS_STATES = {
    "Maharashtra",
    "Madhya Pradesh",
    "Telangana",
    "Andhra Pradesh",
    "Karnataka",
    "Tamil Nadu",
    "Kerala",
    "Goa",
}


def draw_state_base(ax: plt.Axes, parts, focus_states: set[str]) -> None:
    """Draw central/south India state base map."""
    for _, row in parts.iterrows():
        is_focus = row["name"] in focus_states
        patch = Polygon(
            np.asarray(row["points"]),
            closed=True,
            facecolor="#f1f1df" if is_focus else "#323232",
            edgecolor="#777777" if is_focus else "#555555",
            linewidth=0.45 if is_focus else 0.25,
            alpha=1.0 if is_focus else 0.28,
            antialiased=True,
        )
        ax.add_patch(patch)


def draw_state_labels(ax: plt.Axes) -> None:
    """Add readable labels for focus states."""
    labels = [
        ("Maharashtra", 76.4, 19.2),
        ("Madhya Pradesh", 78.3, 23.5),
        ("Telangana", 79.0, 17.6),
        ("Andhra Pradesh", 80.2, 15.4),
        ("Karnataka", 76.2, 14.5),
        ("Tamil Nadu", 78.4, 11.2),
        ("Kerala", 76.2, 10.2),
        ("Goa", 74.0, 15.3),
    ]
    for label, lon, lat in labels:
        ax.text(lon, lat, label, fontsize=7.2, color="#303030", ha="center", va="center", alpha=0.8)


def draw_ctv_points(ax: plt.Axes, ctv_data, strain_class: str) -> None:
    """Draw records for one strain class."""
    subset = ctv_data[ctv_data["strain_class"] == strain_class]
    for cultivar_group, marker in CULTIVAR_MARKERS.items():
        cultivar_subset = subset[subset["cultivar_group"] == cultivar_group]
        if cultivar_subset.empty:
            continue
        ax.scatter(
            cultivar_subset["longitude"],
            cultivar_subset["latitude"],
            s=80 + cultivar_subset["demo_incidence_percent"] * 5,
            marker=marker,
            color=STRAIN_COLORS[strain_class],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.92,
            zorder=10,
        )

    for _, row in subset.iterrows():
        label = f"{row['site']}\n{row['cultivar_group']}"
        ax.annotate(
            label,
            (row["longitude"], row["latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#111111",
            zorder=11,
        )


def draw_panel(ax: plt.Axes, parts, ctv_data, strain_class: str) -> None:
    """Draw one mild/severe panel."""
    draw_state_base(ax, parts, FOCUS_STATES)
    draw_state_labels(ax)
    draw_ctv_points(ax, ctv_data, strain_class)

    ax.set_title(f"{strain_class} CTV Strain Records", color="#f0f0f0", fontsize=13, pad=8)
    ax.set_xlim(72.0, 82.5)
    ax.set_ylim(7.3, 24.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#050505")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_split_strain_map(parts, ctv_data, output_path: Path) -> None:
    """Create two-panel strain map."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 9.0), facecolor="black")
    for ax, strain_class in zip(axes, ["Mild", "Severe"]):
        draw_panel(ax, parts, ctv_data, strain_class)

    cultivar_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            label=group,
            markerfacecolor="#bdbdbd",
            markeredgecolor="black",
            markersize=9,
        )
        for group, marker in CULTIVAR_MARKERS.items()
    ]
    strain_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=strain,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=9,
        )
        for strain, color in STRAIN_COLORS.items()
    ]

    legend = fig.legend(
        handles=strain_handles + cultivar_handles,
        loc="lower center",
        ncol=6,
        frameon=True,
        bbox_to_anchor=(0.5, 0.035),
    )
    legend.get_frame().set_facecolor("#111111")
    legend.get_frame().set_edgecolor("#777777")
    for text in legend.get_texts():
        text.set_color("#f0f0f0")

    fig.suptitle(
        "Citrus Tristeza Virus in Central and South India: Mild vs Severe Demo Layers",
        color="#f0f0f0",
        fontsize=15,
        y=0.965,
    )
    fig.text(
        0.5,
        0.012,
        "Demo CTV records informed by literature; points are illustrative and not measured field surveillance.",
        ha="center",
        va="bottom",
        color="#cfcfcf",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ensure_gadm_files()

    parts = read_adm1_polygons(ADM1_BASENAME)
    ctv_data = make_demo_ctv_records()
    output_path = OUTPUT_DIR / "india_ctv_citrus_mild_vs_severe_zoom_map.png"
    data_path = OUTPUT_DIR / "demo_ctv_citrus_points_v2.csv"

    ctv_data.to_csv(data_path, index=False)
    plot_split_strain_map(parts, ctv_data, output_path)

    print("CTV citrus map version 2 complete.")
    print(f"Demo records: {len(ctv_data)}")
    print(f"Mild records: {(ctv_data['strain_class'] == 'Mild').sum()}")
    print(f"Severe records: {(ctv_data['strain_class'] == 'Severe').sum()}")
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print(f"  - {output_path.name}")
    print(f"  - {data_path.name}")


if __name__ == "__main__":
    main()
