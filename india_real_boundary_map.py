"""Render India using a real ADM1 shapefile boundary dataset.

This script reads the downloaded GADM India ADM1 shapefile directly, without
geopandas, and creates a high-resolution pastel state/UT map.

Input expected:
    gadm41_IND_shp/gadm41_IND_1.shp
    gadm41_IND_shp/gadm41_IND_1.dbf

Run:
    python3 india_real_boundary_map.py
"""

from __future__ import annotations

import os
import struct
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from india_shp_raster_map_demo import read_dbf


GADM_ZIP = Path("gadm41_IND_shp.zip")
GADM_DIR = Path("gadm41_IND_shp")
ADM1_BASENAME = GADM_DIR / "gadm41_IND_1"
OUTPUT_DIR = Path("india_real_boundary_map_output")


PASTEL_COLORS = [
    "#cfe8bf",
    "#fff8b8",
    "#f8cfd0",
    "#ead2ef",
    "#d8edcf",
    "#fff2a9",
    "#f6c9d0",
    "#e7caef",
    "#c8e3c0",
    "#fff6c0",
]


SHORT_LABELS = {
    "Andaman and Nicobar": "Andaman\nand Nicobar",
    "Arunachal Pradesh": "Arunachal\nPradesh",
    "Dadra and Nagar Haveli": "Dadra and\nNagar Haveli",
    "Daman and Diu": "Daman\nand Diu",
    "Himachal Pradesh": "Himachal\nPradesh",
    "Jammu and Kashmir": "Jammu and\nKashmir",
    "Madhya Pradesh": "Madhya\nPradesh",
    "NCT of Delhi": "Delhi",
    "Uttar Pradesh": "Uttar\nPradesh",
    "Andhra Pradesh": "Andhra\nPradesh",
}


MANUAL_LABEL_POSITIONS = {
    "Chandigarh": (76.85, 30.95),
    "Dadra and Nagar Haveli": (70.0, 20.3),
    "Daman and Diu": (70.0, 19.65),
    "Goa": (72.4, 15.0),
    "Lakshadweep": (71.0, 10.4),
    "NCT of Delhi": (77.15, 28.55),
    "Puducherry": (82.3, 11.9),
}


EXTERNAL_LABELS = {
    "Dadra and Nagar Haveli",
    "Daman and Diu",
    "Goa",
    "Lakshadweep",
    "Puducherry",
}


def ensure_gadm_files() -> None:
    """Extract GADM shapefiles if the zip is present and the folder is missing."""
    required = [
        ADM1_BASENAME.with_suffix(".shp"),
        ADM1_BASENAME.with_suffix(".shx"),
        ADM1_BASENAME.with_suffix(".dbf"),
        ADM1_BASENAME.with_suffix(".prj"),
    ]
    if all(path.exists() for path in required):
        return

    if not GADM_ZIP.exists():
        raise FileNotFoundError(
            "Missing GADM India shapefile. Expected gadm41_IND_shp.zip or "
            "gadm41_IND_shp/gadm41_IND_1.shp."
        )

    GADM_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(GADM_ZIP) as archive:
        archive.extractall(GADM_DIR)


def read_adm1_polygons(basename: Path) -> pd.DataFrame:
    """Read ADM1 polygon parts and attributes from a shapefile."""
    attributes = read_dbf(basename.with_suffix(".dbf")).reset_index(drop=True)
    records = []

    with basename.with_suffix(".shp").open("rb") as shp:
        shp.seek(100)
        record_index = 0
        while True:
            header = shp.read(8)
            if not header:
                break

            _, content_length_words = struct.unpack(">2i", header)
            content = shp.read(content_length_words * 2)
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                record_index += 1
                continue
            if shape_type != 5:
                raise ValueError(f"Expected polygon shape type 5, found {shape_type}")

            num_parts, num_points = struct.unpack("<2i", content[36:44])
            part_offsets = list(struct.unpack(f"<{num_parts}i", content[44 : 44 + 4 * num_parts]))
            point_start = 44 + 4 * num_parts
            points = [
                struct.unpack("<2d", content[point_start + point_idx * 16 : point_start + (point_idx + 1) * 16])
                for point_idx in range(num_points)
            ]
            part_offsets.append(num_points)

            attr = attributes.iloc[record_index].to_dict()
            for part_index, (start, end) in enumerate(zip(part_offsets[:-1], part_offsets[1:])):
                part_points = points[start:end]
                if len(part_points) < 3:
                    continue
                records.append(
                    {
                        "name": attr["NAME_1"],
                        "type": attr["ENGTYPE_1"],
                        "record_index": record_index,
                        "part_index": part_index,
                        "points": part_points,
                    }
                )

            record_index += 1

    return pd.DataFrame.from_records(records)


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Return signed polygon area in coordinate units."""
    xy = np.asarray(points)
    x = xy[:, 0]
    y = xy[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Return area-weighted centroid, falling back to mean for tiny rings."""
    xy = np.asarray(points)
    area = polygon_area(points)
    if abs(area) < 1e-10:
        return float(xy[:, 0].mean()), float(xy[:, 1].mean())

    x = xy[:, 0]
    y = xy[:, 1]
    factor = x * np.roll(y, -1) - np.roll(x, -1) * y
    cx = np.sum((x + np.roll(x, -1)) * factor) / (6 * area)
    cy = np.sum((y + np.roll(y, -1)) * factor) / (6 * area)
    return float(cx), float(cy)


def make_label_positions(parts: pd.DataFrame) -> pd.DataFrame:
    """Choose one readable label position per state/UT."""
    rows = []
    for name, subset in parts.groupby("name"):
        if name in MANUAL_LABEL_POSITIONS:
            lon, lat = MANUAL_LABEL_POSITIONS[name]
        else:
            largest = max(subset["points"], key=lambda points: abs(polygon_area(points)))
            lon, lat = polygon_centroid(largest)
        rows.append({"name": name, "longitude": lon, "latitude": lat})
    return pd.DataFrame.from_records(rows)


def plot_real_india_map(parts: pd.DataFrame, output_path: Path, show_labels: bool = True) -> None:
    """Render the real ADM1 boundary map."""
    names = sorted(parts["name"].unique())
    color_lookup = {name: PASTEL_COLORS[idx % len(PASTEL_COLORS)] for idx, name in enumerate(names)}

    fig, ax = plt.subplots(figsize=(10.5, 10.0), facecolor="black")
    ax.set_facecolor("black")

    for _, row in parts.iterrows():
        points = np.asarray(row["points"])
        patch = Polygon(
            points,
            closed=True,
            facecolor=color_lookup[row["name"]],
            edgecolor="#6f6f6f",
            linewidth=0.28,
            antialiased=True,
        )
        ax.add_patch(patch)

    if show_labels:
        labels = make_label_positions(parts)
        centroid_lookup = {
            name: polygon_centroid(max(subset["points"], key=lambda points: abs(polygon_area(points))))
            for name, subset in parts.groupby("name")
        }
        for _, row in labels.iterrows():
            label = SHORT_LABELS.get(row["name"], row["name"])
            is_external = row["name"] in EXTERNAL_LABELS
            fontsize = 5.4 if len(label.replace("\n", "")) > 13 else 6.2
            color = "#e8e8e8" if is_external else "#202020"
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
                    alpha=0.8,
                )

    all_points = np.vstack([np.asarray(points) for points in parts["points"]])
    ax.set_xlim(all_points[:, 0].min() - 1.7, all_points[:, 0].max() + 1.2)
    ax.set_ylim(all_points[:, 1].min() - 1.0, all_points[:, 1].max() + 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.05)
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ensure_gadm_files()

    parts = read_adm1_polygons(ADM1_BASENAME)
    labels = make_label_positions(parts)
    output_path = OUTPUT_DIR / "india_real_state_boundary_map.png"
    no_label_output_path = OUTPUT_DIR / "india_real_state_boundary_map_no_labels.png"
    parts_summary_path = OUTPUT_DIR / "india_adm1_boundary_summary.csv"
    labels_path = OUTPUT_DIR / "india_adm1_label_positions.csv"

    summary = (
        parts.assign(abs_area=parts["points"].map(lambda points: abs(polygon_area(points))))
        .groupby(["name", "type"], as_index=False)
        .agg(parts=("part_index", "count"), approx_area_degrees=("abs_area", "sum"))
        .sort_values("name")
    )
    summary.to_csv(parts_summary_path, index=False)
    labels.to_csv(labels_path, index=False)
    plot_real_india_map(parts, output_path, show_labels=True)
    plot_real_india_map(parts, no_label_output_path, show_labels=False)

    print("Real India boundary map complete.")
    print(f"Input shapefile: {ADM1_BASENAME.with_suffix('.shp')}")
    print(f"Administrative names rendered: {parts['name'].nunique()}")
    print(f"Polygon parts rendered: {len(parts)}")
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print(f"  - {output_path.name}")
    print(f"  - {no_label_output_path.name}")
    print(f"  - {parts_summary_path.name}")
    print(f"  - {labels_path.name}")
    print("\nSource: GADM 4.1 India ADM1 boundaries. Check GADM license before reuse.")


if __name__ == "__main__":
    main()
