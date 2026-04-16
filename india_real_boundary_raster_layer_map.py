"""Render India ADM1 boundaries over a different raster layer.

This script uses the real GADM India ADM1 shapefile and creates a synthetic
annual rainfall raster layer. The raster is clipped to India polygons and
state/UT boundaries are drawn on top.

Run:
    python3 india_real_boundary_raster_layer_map.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Polygon

from india_real_boundary_map import (
    ADM1_BASENAME,
    EXTERNAL_LABELS,
    SHORT_LABELS,
    ensure_gadm_files,
    make_label_positions,
    polygon_area,
    polygon_centroid,
    read_adm1_polygons,
)


OUTPUT_DIR = Path("india_real_boundary_raster_layer_output")
RASTER_PATH = OUTPUT_DIR / "demo_india_annual_rainfall_raster.asc"


def write_rainfall_ascii_raster(path: Path) -> None:
    """Write a demo annual rainfall raster as ESRI ASCII grid."""
    ncols = 280
    nrows = 290
    xllcorner = 66.0
    yllcorner = 5.0
    cellsize = 0.12
    nodata = -9999

    lon = xllcorner + (np.arange(ncols) + 0.5) * cellsize
    lat = yllcorner + (np.arange(nrows) + 0.5) * cellsize
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    western_ghats = 1250 * np.exp(-(((lon_grid - 75.0) / 1.3) ** 2 + ((lat_grid - 13.5) / 5.0) ** 2))
    northeast = 1600 * np.exp(-(((lon_grid - 92.0) / 3.8) ** 2 + ((lat_grid - 26.0) / 2.9) ** 2))
    east_coast = 650 * np.exp(-(((lon_grid - 85.0) / 3.8) ** 2 + ((lat_grid - 20.5) / 4.8) ** 2))
    himalaya = 700 * np.exp(-((lat_grid - 31.0) / 2.8) ** 2) * np.exp(-((lon_grid - 80.0) / 7.0) ** 2)
    thar_dry_zone = -750 * np.exp(-(((lon_grid - 72.0) / 4.2) ** 2 + ((lat_grid - 27.0) / 3.8) ** 2))
    deccan_rain_shadow = -360 * np.exp(-(((lon_grid - 77.0) / 4.2) ** 2 + ((lat_grid - 17.0) / 4.2) ** 2))
    south_gradient = 260 * np.exp(-((lat_grid - 9.5) / 3.0) ** 2)

    rainfall = 850 + western_ghats + northeast + east_coast + himalaya + south_gradient + thar_dry_zone + deccan_rain_shadow
    rainfall = np.clip(rainfall, 150, 2800)

    with path.open("w", encoding="ascii") as raster:
        raster.write(f"ncols {ncols}\n")
        raster.write(f"nrows {nrows}\n")
        raster.write(f"xllcorner {xllcorner}\n")
        raster.write(f"yllcorner {yllcorner}\n")
        raster.write(f"cellsize {cellsize}\n")
        raster.write(f"NODATA_value {nodata}\n")
        for row in rainfall[::-1]:
            raster.write(" ".join(f"{value:.1f}" for value in row) + "\n")


def read_ascii_raster(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    """Read an ESRI ASCII grid raster."""
    header = {}
    with path.open("r", encoding="ascii") as raster:
        for _ in range(6):
            key, value = raster.readline().split()
            header[key.lower()] = float(value)
        data = np.loadtxt(raster)
    return data, header


def make_compound_clip_patch(parts: pd.DataFrame, ax: plt.Axes) -> PathPatch:
    """Create one clip path from all polygon parts."""
    vertices = []
    codes = []
    for points in parts["points"]:
        xy = np.asarray(points)
        if len(xy) < 3:
            continue
        vertices.extend(xy.tolist())
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])

    path = MplPath(np.asarray(vertices), codes)
    return PathPatch(path, transform=ax.transData)


def draw_boundaries(parts: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw state/UT boundaries on top of the raster."""
    for _, row in parts.iterrows():
        xy = np.asarray(row["points"])
        patch = Polygon(
            xy,
            closed=True,
            facecolor="none",
            edgecolor="#1b1b1b",
            linewidth=0.35,
            antialiased=True,
        )
        ax.add_patch(patch)


def draw_labels(parts: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw labels on the raster map."""
    labels = make_label_positions(parts)
    centroid_lookup = {
        name: polygon_centroid(max(subset["points"], key=lambda points: abs(polygon_area(points))))
        for name, subset in parts.groupby("name")
    }
    for _, row in labels.iterrows():
        label = SHORT_LABELS.get(row["name"], row["name"])
        is_external = row["name"] in EXTERNAL_LABELS
        color = "#f0f0f0" if is_external else "#151515"
        fontsize = 5.3 if len(label.replace("\n", "")) > 13 else 6.1
        ax.text(
            row["longitude"],
            row["latitude"],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            clip_on=False,
        )
        if is_external:
            start = centroid_lookup[row["name"]]
            ax.plot(
                [start[0], row["longitude"]],
                [start[1], row["latitude"]],
                color="#d8d8d8",
                linewidth=0.45,
                alpha=0.85,
            )


def summarize_raster_by_admin(parts: pd.DataFrame, raster: np.ndarray, header: dict[str, float]) -> pd.DataFrame:
    """Summarize raster values by ADM1 using point-in-polygon sampling."""
    nrows, ncols = raster.shape
    x_min = header["xllcorner"]
    y_min = header["yllcorner"]
    cellsize = header["cellsize"]
    lon = x_min + (np.arange(ncols) + 0.5) * cellsize
    lat = y_min + (np.arange(nrows) + 0.5) * cellsize
    lon_grid, lat_grid = np.meshgrid(lon, lat[::-1])
    sample_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    raster_values = raster.ravel()

    rows = []
    for name, subset in parts.groupby("name"):
        mask = np.zeros(sample_points.shape[0], dtype=bool)
        for points in subset["points"]:
            mask |= MplPath(np.asarray(points)).contains_points(sample_points)
        values = raster_values[mask]
        if values.size == 0:
            continue
        rows.append(
            {
                "name": name,
                "sampled_cells": int(values.size),
                "mean_annual_rainfall_mm": round(float(values.mean()), 1),
                "min_annual_rainfall_mm": round(float(values.min()), 1),
                "max_annual_rainfall_mm": round(float(values.max()), 1),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("mean_annual_rainfall_mm", ascending=False)


def plot_raster_boundary_map(
    parts: pd.DataFrame,
    raster: np.ndarray,
    header: dict[str, float],
    output_path: Path,
    show_labels: bool,
) -> None:
    """Plot raster layer clipped to India with real ADM1 boundaries."""
    x_min = header["xllcorner"]
    y_min = header["yllcorner"]
    x_max = x_min + header["ncols"] * header["cellsize"]
    y_max = y_min + header["nrows"] * header["cellsize"]

    fig, ax = plt.subplots(figsize=(10.5, 10.0), facecolor="black")
    ax.set_facecolor("black")

    image = ax.imshow(
        raster,
        extent=(x_min, x_max, y_min, y_max),
        origin="upper",
        cmap="terrain_r",
        vmin=150,
        vmax=2800,
        interpolation="bilinear",
    )
    image.set_clip_path(make_compound_clip_patch(parts, ax))

    draw_boundaries(parts, ax)
    if show_labels:
        draw_labels(parts, ax)

    all_points = np.vstack([np.asarray(points) for points in parts["points"]])
    ax.set_xlim(all_points[:, 0].min() - 1.7, all_points[:, 0].max() + 1.2)
    ax.set_ylim(all_points[:, 1].min() - 1.0, all_points[:, 1].max() + 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.034, pad=0.02)
    colorbar.set_label("Demo annual rainfall (mm)", color="#f0f0f0")
    colorbar.ax.yaxis.set_tick_params(color="#f0f0f0")
    plt.setp(colorbar.ax.get_yticklabels(), color="#f0f0f0")

    fig.tight_layout(pad=0.05)
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ensure_gadm_files()

    parts = read_adm1_polygons(ADM1_BASENAME)
    write_rainfall_ascii_raster(RASTER_PATH)
    raster, header = read_ascii_raster(RASTER_PATH)

    labeled_map_path = OUTPUT_DIR / "india_real_boundary_rainfall_raster_map.png"
    clean_map_path = OUTPUT_DIR / "india_real_boundary_rainfall_raster_map_no_labels.png"
    raster_summary_path = OUTPUT_DIR / "rainfall_raster_summary_by_state.csv"

    raster_summary = summarize_raster_by_admin(parts, raster, header)
    raster_summary.to_csv(raster_summary_path, index=False)
    plot_raster_boundary_map(parts, raster, header, labeled_map_path, show_labels=True)
    plot_raster_boundary_map(parts, raster, header, clean_map_path, show_labels=False)

    print("India real-boundary raster layer map complete.")
    print(f"Input boundary shapefile: {ADM1_BASENAME.with_suffix('.shp')}")
    print(f"Raster layer: {RASTER_PATH}")
    print(f"Raster shape: {raster.shape[0]} rows x {raster.shape[1]} columns")
    print(f"Administrative names rendered: {parts['name'].nunique()}")
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print(f"  - {labeled_map_path.name}")
    print(f"  - {clean_map_path.name}")
    print(f"  - {RASTER_PATH.name}")
    print(f"  - {raster_summary_path.name}")
    print("\nNote: rainfall raster values are synthetic demo data; boundaries are from GADM 4.1.")


if __name__ == "__main__":
    main()
