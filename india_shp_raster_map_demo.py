"""Create an India map by importing demo shapefile and raster files.

This script demonstrates a lightweight GIS workflow without geopandas:
    - imports an India outline polygon shapefile
    - imports a point shapefile with demo plant disease coordinates
    - creates and imports an ESRI ASCII raster grid
    - overlays demo coordinates on the raster and shapefile map

Run:
    python3 india_shp_raster_map_demo.py
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from create_india_plant_disease_shapefiles import main as create_demo_shapefiles


OUTPUT_DIR = Path("india_shp_raster_demo_output")
SOURCE_SHAPE_DIR = Path("plant_disease_map_output") / "shapefiles"
RASTER_PATH = OUTPUT_DIR / "demo_india_disease_risk_raster.asc"
DEMO_COORDINATE_PATH = OUTPUT_DIR / "demo_map_coordinates.csv"


def ensure_demo_shapefiles() -> None:
    """Create the source shapefiles if they are not already available."""
    required_files = [
        SOURCE_SHAPE_DIR / "india_approx_outline.shp",
        SOURCE_SHAPE_DIR / "india_approx_outline.dbf",
        SOURCE_SHAPE_DIR / "india_plant_disease_incidence_points.shp",
        SOURCE_SHAPE_DIR / "india_plant_disease_incidence_points.dbf",
    ]
    if not all(path.exists() for path in required_files):
        create_demo_shapefiles()


def read_dbf(path: Path) -> pd.DataFrame:
    """Read simple dBASE III attributes used by the demo shapefiles."""
    with path.open("rb") as dbf:
        header = dbf.read(32)
        n_records = struct.unpack("<I", header[4:8])[0]
        header_length = struct.unpack("<H", header[8:10])[0]
        record_length = struct.unpack("<H", header[10:12])[0]

        fields = []
        while dbf.tell() < header_length - 1:
            descriptor = dbf.read(32)
            name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii")
            field_type = chr(descriptor[11])
            width = descriptor[16]
            decimals = descriptor[17]
            fields.append((name, field_type, width, decimals))

        dbf.read(1)
        records = []
        for _ in range(n_records):
            raw_record = dbf.read(record_length)
            if not raw_record or raw_record[0:1] == b"*":
                continue

            offset = 1
            record = {}
            for name, field_type, width, decimals in fields:
                raw_value = raw_record[offset : offset + width].decode("ascii", errors="ignore").strip()
                offset += width
                if field_type == "N" and raw_value:
                    record[name] = float(raw_value) if decimals else int(raw_value)
                else:
                    record[name] = raw_value
            records.append(record)

    return pd.DataFrame.from_records(records)


def read_point_shapefile(basename: Path) -> pd.DataFrame:
    """Read a point shapefile and join attributes from the DBF table."""
    points = []
    with basename.with_suffix(".shp").open("rb") as shp:
        shp.seek(100)
        while True:
            record_header = shp.read(8)
            if not record_header:
                break
            record_number, content_length_words = struct.unpack(">2i", record_header)
            content = shp.read(content_length_words * 2)
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type != 1:
                raise ValueError(f"Expected point shape type 1, found {shape_type}")
            x, y = struct.unpack("<2d", content[4:20])
            points.append({"record_number": record_number, "longitude": x, "latitude": y})

    attributes = read_dbf(basename.with_suffix(".dbf"))
    geometry = pd.DataFrame.from_records(points)
    return pd.concat([geometry, attributes], axis=1)


def read_polygon_shapefile(basename: Path) -> list[pd.DataFrame]:
    """Read polygon parts from a shapefile."""
    polygons = []
    with basename.with_suffix(".shp").open("rb") as shp:
        shp.seek(100)
        while True:
            record_header = shp.read(8)
            if not record_header:
                break
            _, content_length_words = struct.unpack(">2i", record_header)
            content = shp.read(content_length_words * 2)
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type != 5:
                raise ValueError(f"Expected polygon shape type 5, found {shape_type}")

            num_parts, num_points = struct.unpack("<2i", content[36:44])
            part_offsets = list(struct.unpack(f"<{num_parts}i", content[44 : 44 + 4 * num_parts]))
            point_start = 44 + 4 * num_parts
            points = [
                struct.unpack("<2d", content[point_start + i * 16 : point_start + (i + 1) * 16])
                for i in range(num_points)
            ]

            part_offsets.append(num_points)
            for start, end in zip(part_offsets[:-1], part_offsets[1:]):
                polygons.append(pd.DataFrame(points[start:end], columns=["longitude", "latitude"]))

    return polygons


def make_demo_coordinates() -> pd.DataFrame:
    """Create separate demo coordinates to show user-supplied point overlay."""
    records = [
        {"label": "Delhi demo station", "longitude": 77.2090, "latitude": 28.6139, "sample_value": 32},
        {"label": "Mumbai demo station", "longitude": 72.8777, "latitude": 19.0760, "sample_value": 46},
        {"label": "Bengaluru demo station", "longitude": 77.5946, "latitude": 12.9716, "sample_value": 41},
        {"label": "Kolkata demo station", "longitude": 88.3639, "latitude": 22.5726, "sample_value": 55},
        {"label": "Guwahati demo station", "longitude": 91.7362, "latitude": 26.1445, "sample_value": 61},
        {"label": "Ahmedabad demo station", "longitude": 72.5714, "latitude": 23.0225, "sample_value": 38},
    ]
    return pd.DataFrame.from_records(records)


def write_demo_ascii_raster(path: Path) -> None:
    """Write a synthetic ESRI ASCII raster over India in lon/lat degrees."""
    ncols = 125
    nrows = 120
    xllcorner = 67.0
    yllcorner = 6.0
    cellsize = 0.25
    nodata = -9999

    lon = xllcorner + (np.arange(ncols) + 0.5) * cellsize
    lat = yllcorner + (np.arange(nrows) + 0.5) * cellsize
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    east_hotspot = 35 * np.exp(-(((lon_grid - 88.5) / 4.8) ** 2 + ((lat_grid - 23.5) / 4.0) ** 2))
    south_hotspot = 28 * np.exp(-(((lon_grid - 77.0) / 3.6) ** 2 + ((lat_grid - 12.0) / 3.4) ** 2))
    north_band = 18 * np.exp(-((lat_grid - 29.0) / 3.2) ** 2)
    west_drydown = -10 * np.exp(-(((lon_grid - 71.5) / 4.0) ** 2 + ((lat_grid - 25.0) / 5.0) ** 2))
    risk = np.clip(18 + east_hotspot + south_hotspot + north_band + west_drydown, 0, 80)

    # ESRI ASCII grid writes the northernmost row first.
    with path.open("w", encoding="ascii") as raster:
        raster.write(f"ncols {ncols}\n")
        raster.write(f"nrows {nrows}\n")
        raster.write(f"xllcorner {xllcorner}\n")
        raster.write(f"yllcorner {yllcorner}\n")
        raster.write(f"cellsize {cellsize}\n")
        raster.write(f"NODATA_value {nodata}\n")
        for row in risk[::-1]:
            raster.write(" ".join(f"{value:.2f}" for value in row) + "\n")


def read_ascii_raster(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    """Import an ESRI ASCII raster grid."""
    header = {}
    with path.open("r", encoding="ascii") as raster:
        for _ in range(6):
            key, value = raster.readline().split()
            header[key.lower()] = float(value)
        data = np.loadtxt(raster)

    return data, header


def plot_shapefile_raster_map(
    polygons: list[pd.DataFrame],
    shapefile_points: pd.DataFrame,
    demo_coordinates: pd.DataFrame,
    raster: np.ndarray,
    raster_header: dict[str, float],
    output_path: Path,
) -> None:
    """Save a map with imported shapefile, raster, and demo coordinate overlays."""
    x_min = raster_header["xllcorner"]
    y_min = raster_header["yllcorner"]
    x_max = x_min + raster_header["ncols"] * raster_header["cellsize"]
    y_max = y_min + raster_header["nrows"] * raster_header["cellsize"]

    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    image = ax.imshow(
        raster,
        extent=(x_min, x_max, y_min, y_max),
        origin="upper",
        cmap="YlGnBu",
        alpha=0.78,
        vmin=0,
        vmax=80,
    )

    for polygon in polygons:
        ax.plot(polygon["longitude"], polygon["latitude"], color="#222222", linewidth=1.4)

    shapefile_scatter = ax.scatter(
        shapefile_points["longitude"],
        shapefile_points["latitude"],
        c=shapefile_points["inc_pct"],
        cmap="YlOrRd",
        vmin=0,
        vmax=70,
        s=48 + shapefile_points["inc_pct"] * 2.3,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.88,
        label="Imported point shapefile",
    )

    ax.scatter(
        demo_coordinates["longitude"],
        demo_coordinates["latitude"],
        marker="*",
        s=190,
        color="#111111",
        edgecolor="white",
        linewidth=0.8,
        label="Demo coordinates",
    )
    for _, row in demo_coordinates.iterrows():
        ax.annotate(
            row["label"].replace(" demo station", ""),
            (row["longitude"], row["latitude"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="#111111",
        )

    raster_colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    raster_colorbar.set_label("Demo raster risk index")
    point_colorbar = fig.colorbar(shapefile_scatter, ax=ax, fraction=0.035, pad=0.09)
    point_colorbar.set_label("Point incidence (%)")

    ax.set_title("India Demo Map: Imported Shapefile, Raster, and Coordinates")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(67, 98)
    ax.set_ylim(6, 36)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#ffffff", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ensure_demo_shapefiles()

    outline_basename = SOURCE_SHAPE_DIR / "india_approx_outline"
    point_basename = SOURCE_SHAPE_DIR / "india_plant_disease_incidence_points"

    write_demo_ascii_raster(RASTER_PATH)
    demo_coordinates = make_demo_coordinates()
    demo_coordinates.to_csv(DEMO_COORDINATE_PATH, index=False)

    polygons = read_polygon_shapefile(outline_basename)
    shapefile_points = read_point_shapefile(point_basename)
    raster, raster_header = read_ascii_raster(RASTER_PATH)

    imported_points_path = OUTPUT_DIR / "imported_shapefile_points.csv"
    imported_points = shapefile_points[
        ["site", "state", "crop", "disease", "latitude", "longitude", "inc_pct"]
    ].copy()
    imported_points.to_csv(imported_points_path, index=False)

    map_path = OUTPUT_DIR / "india_shapefile_raster_coordinate_map.png"
    plot_shapefile_raster_map(
        polygons=polygons,
        shapefile_points=shapefile_points,
        demo_coordinates=demo_coordinates,
        raster=raster,
        raster_header=raster_header,
        output_path=map_path,
    )

    print("India shapefile and raster map demo complete.")
    print(f"Imported polygon shapefile: {outline_basename.with_suffix('.shp')}")
    print(f"Imported point shapefile: {point_basename.with_suffix('.shp')}")
    print(f"Imported raster file: {RASTER_PATH}")
    print(f"Raster shape: {raster.shape[0]} rows x {raster.shape[1]} columns")
    print(f"Imported shapefile points: {len(shapefile_points)}")
    print(f"Demo coordinates: {len(demo_coordinates)}")
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print(f"  - {RASTER_PATH.name}")
    print(f"  - {DEMO_COORDINATE_PATH.name}")
    print(f"  - {imported_points_path.name}")
    print(f"  - {map_path.name}")


if __name__ == "__main__":
    main()
