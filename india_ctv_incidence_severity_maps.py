"""Generate multiple CTV incidence and severity maps.

This script reuses the source-noted demo Citrus tristeza virus records and
creates several map products:
    - incidence bubble map
    - severity class map
    - state mean incidence choropleth
    - state severity index choropleth
    - cultivar group small-multiple maps

Run:
    python3 india_ctv_incidence_severity_maps.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from india_ctv_citrus_disease_layer_map import (
    CULTIVAR_MARKERS,
    STRAIN_COLORS,
    make_demo_ctv_records,
)
from india_real_boundary_map import ADM1_BASENAME, ensure_gadm_files, read_adm1_polygons


OUTPUT_DIR = Path("india_ctv_incidence_severity_maps_output")

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

MAP_EXTENT = (72.0, 82.6, 7.2, 24.4)


def add_severity_fields(ctv_data: pd.DataFrame) -> pd.DataFrame:
    """Add numeric severity score fields for mapping and summarization."""
    data = ctv_data.copy()
    data["severity_score"] = np.where(data["strain_class"] == "Severe", 3, 1)
    data["severity_label"] = np.where(data["strain_class"] == "Severe", "Severe strain", "Mild strain")
    data["disease_pressure_index"] = (data["demo_incidence_percent"] * data["severity_score"]).round(1)
    return data


def summarize_state_disease(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize incidence and severity by state."""
    return (
        data.groupby("state", as_index=False)
        .agg(
            records=("site", "count"),
            mean_incidence_percent=("demo_incidence_percent", "mean"),
            max_incidence_percent=("demo_incidence_percent", "max"),
            mean_severity_score=("severity_score", "mean"),
            severe_records=("strain_class", lambda values: int((values == "Severe").sum())),
            mild_records=("strain_class", lambda values: int((values == "Mild").sum())),
            mean_disease_pressure_index=("disease_pressure_index", "mean"),
        )
        .round(2)
        .sort_values("mean_disease_pressure_index", ascending=False)
    )


def state_value_lookup(summary: pd.DataFrame, value_col: str) -> dict[str, float]:
    """Return state to value lookup from a summary table."""
    return dict(zip(summary["state"], summary[value_col]))


def draw_base_boundaries(
    ax: plt.Axes,
    parts: pd.DataFrame,
    fill_lookup: dict[str, float] | None = None,
    cmap_name: str = "YlOrRd",
    value_range: tuple[float, float] | None = None,
):
    """Draw state boundaries and optional focus-state choropleth fill."""
    cmap = plt.get_cmap(cmap_name)
    norm = None
    if fill_lookup:
        values = list(fill_lookup.values())
        vmin, vmax = value_range if value_range else (min(values), max(values))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for _, row in parts.iterrows():
        state = row["name"]
        is_focus = state in FOCUS_STATES
        if fill_lookup and state in fill_lookup:
            facecolor = cmap(norm(fill_lookup[state]))
            alpha = 0.96
        elif is_focus:
            facecolor = "#f3f1df"
            alpha = 1.0
        else:
            facecolor = "#2f2f2f"
            alpha = 0.22

        patch = Polygon(
            np.asarray(row["points"]),
            closed=True,
            facecolor=facecolor,
            edgecolor="#6a6a6a" if is_focus else "#4e4e4e",
            linewidth=0.38 if is_focus else 0.22,
            alpha=alpha,
            antialiased=True,
        )
        ax.add_patch(patch)

    return cmap, norm


def draw_focus_labels(ax: plt.Axes) -> None:
    """Add compact state labels for central and south India."""
    labels = [
        ("Maharashtra", 76.3, 19.2),
        ("Madhya Pradesh", 78.2, 23.5),
        ("Telangana", 79.0, 17.6),
        ("Andhra Pradesh", 80.2, 15.3),
        ("Karnataka", 76.1, 14.5),
        ("Tamil Nadu", 78.4, 11.0),
        ("Kerala", 76.2, 10.1),
        ("Goa", 74.0, 15.3),
    ]
    for label, lon, lat in labels:
        ax.text(lon, lat, label, fontsize=6.4, color="#222222", ha="center", va="center", alpha=0.72)


def setup_axis(ax: plt.Axes, title: str) -> None:
    """Apply common map styling."""
    x_min, x_max, y_min, y_max = MAP_EXTENT
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#050505")
    ax.set_title(title, color="#f2f2f2", fontsize=12, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_incidence_bubble_map(parts: pd.DataFrame, data: pd.DataFrame, output_path: Path) -> None:
    """Map incidence as bubble size and strain class as color."""
    fig, ax = plt.subplots(figsize=(8.8, 9.2), facecolor="black")
    draw_base_boundaries(ax, parts)
    draw_focus_labels(ax)

    for strain_class, color in STRAIN_COLORS.items():
        subset = data[data["strain_class"] == strain_class]
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            s=40 + subset["demo_incidence_percent"] * 6.2,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            label=strain_class,
            zorder=10,
        )

    for _, row in data.iterrows():
        ax.annotate(
            f"{row['site']}\n{row['demo_incidence_percent']}%",
            (row["longitude"], row["latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.4,
            color="#111111",
            zorder=11,
        )

    legend = ax.legend(title="Strain class", loc="lower left", frameon=True)
    style_legend(legend)
    setup_axis(ax, "CTV Disease Incidence by Site")
    add_footer(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_severity_class_map(parts: pd.DataFrame, data: pd.DataFrame, output_path: Path) -> None:
    """Map strain severity and cultivar marker shape."""
    fig, ax = plt.subplots(figsize=(8.8, 9.2), facecolor="black")
    draw_base_boundaries(ax, parts)
    draw_focus_labels(ax)

    for cultivar_group, marker in CULTIVAR_MARKERS.items():
        for strain_class, color in STRAIN_COLORS.items():
            subset = data[(data["cultivar_group"] == cultivar_group) & (data["strain_class"] == strain_class)]
            if subset.empty:
                continue
            ax.scatter(
                subset["longitude"],
                subset["latitude"],
                s=155,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.94,
                zorder=10,
            )

    for _, row in data.iterrows():
        ax.annotate(
            f"{row['site']}\n{row['cultivar_group']}",
            (row["longitude"], row["latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.2,
            color="#111111",
            zorder=11,
        )

    handles = make_combined_legend_handles()
    legend = ax.legend(handles=handles, loc="lower left", ncol=2, frameon=True)
    style_legend(legend)
    setup_axis(ax, "CTV Severity Class and Citrus Cultivar")
    add_footer(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_state_choropleth(
    parts: pd.DataFrame,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    value_col: str,
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap_name: str,
) -> None:
    """Draw a state-level choropleth with survey points."""
    fig, ax = plt.subplots(figsize=(8.8, 9.2), facecolor="black")
    fill_lookup = state_value_lookup(summary, value_col)
    cmap, norm = draw_base_boundaries(ax, parts, fill_lookup=fill_lookup, cmap_name=cmap_name)

    ax.scatter(
        data["longitude"],
        data["latitude"],
        s=60,
        color="#111111",
        edgecolor="white",
        linewidth=0.65,
        alpha=0.9,
        zorder=10,
    )

    for _, row in summary.iterrows():
        state_points = data[data["state"] == row["state"]]
        if state_points.empty:
            continue
        ax.text(
            state_points["longitude"].mean(),
            state_points["latitude"].mean() + 0.35,
            f"{row[value_col]:.1f}",
            fontsize=7,
            color="#111111",
            ha="center",
            va="center",
            zorder=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )

    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label(colorbar_label, color="#f2f2f2")
    colorbar.ax.yaxis.set_tick_params(color="#f2f2f2")
    plt.setp(colorbar.ax.get_yticklabels(), color="#f2f2f2")

    setup_axis(ax, title)
    add_footer(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cultivar_small_multiples(parts: pd.DataFrame, data: pd.DataFrame, output_path: Path) -> None:
    """Create small maps split by cultivar group."""
    cultivar_groups = list(CULTIVAR_MARKERS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), facecolor="black")
    axes = axes.ravel()

    for ax, cultivar_group in zip(axes, cultivar_groups):
        draw_base_boundaries(ax, parts)
        subset = data[data["cultivar_group"] == cultivar_group]
        for strain_class, color in STRAIN_COLORS.items():
            strain_subset = subset[subset["strain_class"] == strain_class]
            if strain_subset.empty:
                continue
            ax.scatter(
                strain_subset["longitude"],
                strain_subset["latitude"],
                s=60 + strain_subset["demo_incidence_percent"] * 5,
                marker=CULTIVAR_MARKERS[cultivar_group],
                color=color,
                edgecolor="white",
                linewidth=0.75,
                alpha=0.92,
                zorder=10,
            )
        for _, row in subset.iterrows():
            ax.annotate(
                row["site"],
                (row["longitude"], row["latitude"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.2,
                color="#111111",
                zorder=11,
            )
        setup_axis(ax, cultivar_group)

    legend = fig.legend(handles=make_strain_handles(), loc="lower center", ncol=2, frameon=True)
    style_legend(legend)
    fig.suptitle("CTV Incidence and Severity by Citrus Cultivar Group", color="#f2f2f2", fontsize=15, y=0.965)
    add_footer(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_strain_handles() -> list[Line2D]:
    """Legend handles for strain classes."""
    return [
        Line2D([0], [0], marker="o", color="none", label=strain, markerfacecolor=color, markeredgecolor="white", markersize=8)
        for strain, color in STRAIN_COLORS.items()
    ]


def make_combined_legend_handles() -> list[Line2D]:
    """Legend handles for strain color and cultivar shape."""
    strain_handles = make_strain_handles()
    cultivar_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            label=group,
            markerfacecolor="#bdbdbd",
            markeredgecolor="white",
            markersize=8,
        )
        for group, marker in CULTIVAR_MARKERS.items()
    ]
    return strain_handles + cultivar_handles


def style_legend(legend) -> None:
    """Apply dark legend styling."""
    legend.get_frame().set_facecolor("#111111")
    legend.get_frame().set_edgecolor("#777777")
    if legend.get_title():
        legend.get_title().set_color("#f2f2f2")
    for text in legend.get_texts():
        text.set_color("#f2f2f2")


def add_footer(fig: plt.Figure) -> None:
    """Add demo-data note."""
    fig.text(
        0.5,
        0.01,
        "Demo CTV layer: incidence and severity values are illustrative; boundaries are GADM ADM1.",
        ha="center",
        va="bottom",
        color="#cfcfcf",
        fontsize=7.5,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ensure_gadm_files()

    parts = read_adm1_polygons(ADM1_BASENAME)
    ctv_data = add_severity_fields(make_demo_ctv_records())
    state_summary = summarize_state_disease(ctv_data)

    ctv_data_path = OUTPUT_DIR / "ctv_incidence_severity_site_data.csv"
    state_summary_path = OUTPUT_DIR / "ctv_incidence_severity_state_summary.csv"
    ctv_data.to_csv(ctv_data_path, index=False)
    state_summary.to_csv(state_summary_path, index=False)

    plot_incidence_bubble_map(
        parts,
        ctv_data,
        OUTPUT_DIR / "map_1_ctv_site_incidence_bubbles.png",
    )
    plot_severity_class_map(
        parts,
        ctv_data,
        OUTPUT_DIR / "map_2_ctv_strain_severity_cultivar_points.png",
    )
    plot_state_choropleth(
        parts,
        ctv_data,
        state_summary,
        value_col="mean_incidence_percent",
        title="State Mean CTV Incidence",
        colorbar_label="Mean incidence (%)",
        output_path=OUTPUT_DIR / "map_3_state_mean_incidence_choropleth.png",
        cmap_name="YlOrRd",
    )
    plot_state_choropleth(
        parts,
        ctv_data,
        state_summary,
        value_col="mean_disease_pressure_index",
        title="State CTV Disease Pressure Index",
        colorbar_label="Incidence x severity score",
        output_path=OUTPUT_DIR / "map_4_state_disease_pressure_choropleth.png",
        cmap_name="PuRd",
    )
    plot_cultivar_small_multiples(
        parts,
        ctv_data,
        OUTPUT_DIR / "map_5_cultivar_group_small_multiples.png",
    )

    print("Multiple CTV incidence and severity maps complete.")
    print(f"Site records: {len(ctv_data)}")
    print(f"States summarized: {len(state_summary)}")
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print("  - ctv_incidence_severity_site_data.csv")
    print("  - ctv_incidence_severity_state_summary.csv")
    print("  - map_1_ctv_site_incidence_bubbles.png")
    print("  - map_2_ctv_strain_severity_cultivar_points.png")
    print("  - map_3_state_mean_incidence_choropleth.png")
    print("  - map_4_state_disease_pressure_choropleth.png")
    print("  - map_5_cultivar_group_small_multiples.png")


if __name__ == "__main__":
    main()
