"""Generate full-India CTV incidence and severity maps.

This script uses the same source-noted demo Citrus tristeza virus records as
the central/south maps, but renders the full India boundary.

Run:
    python3 india_ctv_full_india_maps.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from india_ctv_citrus_disease_layer_map import CULTIVAR_MARKERS, STRAIN_COLORS, make_demo_ctv_records
from india_ctv_incidence_severity_maps import add_severity_fields, summarize_state_disease
from india_real_boundary_map import ADM1_BASENAME, ensure_gadm_files, read_adm1_polygons


OUTPUT_DIR = Path("india_ctv_full_india_maps_output")


def full_extent(parts) -> tuple[float, float, float, float]:
    """Return full India plotting extent from all polygon parts."""
    all_points = np.vstack([np.asarray(points) for points in parts["points"]])
    return (
        float(all_points[:, 0].min() - 1.3),
        float(all_points[:, 0].max() + 1.2),
        float(all_points[:, 1].min() - 1.0),
        float(all_points[:, 1].max() + 1.0),
    )


def draw_full_india_base(ax, parts, focus_states: set[str] | None = None) -> None:
    """Draw full India base map with central/south CTV states highlighted."""
    focus_states = focus_states or set()
    for _, row in parts.iterrows():
        is_focus = row["name"] in focus_states
        patch = Polygon(
            np.asarray(row["points"]),
            closed=True,
            facecolor="#f3f1df" if is_focus else "#d8d8cc",
            edgecolor="#616161",
            linewidth=0.42 if is_focus else 0.28,
            alpha=1.0 if is_focus else 0.62,
            antialiased=True,
        )
        ax.add_patch(patch)


def setup_full_axis(ax, parts, title: str) -> None:
    """Apply full India map styling."""
    x_min, x_max, y_min, y_max = full_extent(parts)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#050505")
    ax.set_title(title, color="#f2f2f2", fontsize=13, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def style_legend(legend) -> None:
    """Apply dark legend styling."""
    legend.get_frame().set_facecolor("#111111")
    legend.get_frame().set_edgecolor("#777777")
    if legend.get_title():
        legend.get_title().set_color("#f2f2f2")
    for text in legend.get_texts():
        text.set_color("#f2f2f2")


def add_footer(fig) -> None:
    """Add demo-data note."""
    fig.text(
        0.5,
        0.012,
        "Demo CTV incidence/severity values are illustrative; boundaries are GADM ADM1.",
        ha="center",
        va="bottom",
        color="#cfcfcf",
        fontsize=8,
    )


def plot_full_incidence_map(parts, data, output_path: Path) -> None:
    """Create full India incidence bubble map."""
    focus_states = set(data["state"])
    fig, ax = plt.subplots(figsize=(9.2, 9.6), facecolor="black")
    draw_full_india_base(ax, parts, focus_states)

    for strain_class, color in STRAIN_COLORS.items():
        subset = data[data["strain_class"] == strain_class]
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            s=45 + subset["demo_incidence_percent"] * 5.5,
            color=color,
            edgecolor="white",
            linewidth=0.75,
            alpha=0.92,
            label=strain_class,
            zorder=10,
        )

    for _, row in data.iterrows():
        ax.annotate(
            f"{row['site']} {row['demo_incidence_percent']}%",
            (row["longitude"], row["latitude"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=5.5,
            color="#f0f0f0",
            zorder=11,
        )

    legend = ax.legend(title="Strain class", loc="lower left", frameon=True)
    style_legend(legend)
    setup_full_axis(ax, parts, "Full India CTV Site Incidence")
    add_footer(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_full_severity_cultivar_map(parts, data, output_path: Path) -> None:
    """Create full India severity/cultivar map."""
    focus_states = set(data["state"])
    fig, ax = plt.subplots(figsize=(9.2, 9.6), facecolor="black")
    draw_full_india_base(ax, parts, focus_states)

    for cultivar_group, marker in CULTIVAR_MARKERS.items():
        for strain_class, color in STRAIN_COLORS.items():
            subset = data[(data["cultivar_group"] == cultivar_group) & (data["strain_class"] == strain_class)]
            if subset.empty:
                continue
            ax.scatter(
                subset["longitude"],
                subset["latitude"],
                s=150,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.75,
                alpha=0.94,
                zorder=10,
            )

    handles = [
        Line2D([0], [0], marker="o", color="none", label=strain, markerfacecolor=color, markeredgecolor="white", markersize=8)
        for strain, color in STRAIN_COLORS.items()
    ]
    handles.extend(
        [
            Line2D([0], [0], marker=marker, color="none", label=group, markerfacecolor="#bbbbbb", markeredgecolor="white", markersize=8)
            for group, marker in CULTIVAR_MARKERS.items()
        ]
    )
    legend = ax.legend(handles=handles, loc="lower left", ncol=2, frameon=True)
    style_legend(legend)
    setup_full_axis(ax, parts, "Full India CTV Severity and Citrus Cultivar")
    add_footer(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_full_state_pressure_map(parts, data, state_summary, output_path: Path) -> None:
    """Create full India state-level disease pressure choropleth."""
    pressure_lookup = dict(zip(state_summary["state"], state_summary["mean_disease_pressure_index"]))
    values = list(pressure_lookup.values())
    norm = plt.Normalize(vmin=min(values), vmax=max(values))
    cmap = plt.get_cmap("PuRd")

    fig, ax = plt.subplots(figsize=(9.2, 9.6), facecolor="black")
    for _, row in parts.iterrows():
        state = row["name"]
        if state in pressure_lookup:
            facecolor = cmap(norm(pressure_lookup[state]))
            alpha = 0.96
        else:
            facecolor = "#d8d8cc"
            alpha = 0.35
        patch = Polygon(
            np.asarray(row["points"]),
            closed=True,
            facecolor=facecolor,
            edgecolor="#616161",
            linewidth=0.34,
            alpha=alpha,
            antialiased=True,
        )
        ax.add_patch(patch)

    ax.scatter(
        data["longitude"],
        data["latitude"],
        s=42,
        color="#111111",
        edgecolor="white",
        linewidth=0.55,
        zorder=10,
    )

    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Incidence x severity score", color="#f2f2f2")
    colorbar.ax.yaxis.set_tick_params(color="#f2f2f2")
    plt.setp(colorbar.ax.get_yticklabels(), color="#f2f2f2")

    setup_full_axis(ax, parts, "Full India CTV Disease Pressure by State")
    add_footer(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ensure_gadm_files()

    parts = read_adm1_polygons(ADM1_BASENAME)
    ctv_data = add_severity_fields(make_demo_ctv_records())
    state_summary = summarize_state_disease(ctv_data)

    ctv_data_path = OUTPUT_DIR / "full_india_ctv_site_data.csv"
    state_summary_path = OUTPUT_DIR / "full_india_ctv_state_summary.csv"
    ctv_data.to_csv(ctv_data_path, index=False)
    state_summary.to_csv(state_summary_path, index=False)

    plot_full_incidence_map(parts, ctv_data, OUTPUT_DIR / "full_india_map_1_ctv_incidence.png")
    plot_full_severity_cultivar_map(parts, ctv_data, OUTPUT_DIR / "full_india_map_2_ctv_severity_cultivar.png")
    plot_full_state_pressure_map(parts, ctv_data, state_summary, OUTPUT_DIR / "full_india_map_3_ctv_state_pressure.png")

    print("Full India CTV maps complete.")
    print(f"Site records: {len(ctv_data)}")
    print(f"States with CTV demo records: {len(state_summary)}")
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print("  - full_india_ctv_site_data.csv")
    print("  - full_india_ctv_state_summary.csv")
    print("  - full_india_map_1_ctv_incidence.png")
    print("  - full_india_map_2_ctv_severity_cultivar.png")
    print("  - full_india_map_3_ctv_state_pressure.png")


if __name__ == "__main__":
    main()
