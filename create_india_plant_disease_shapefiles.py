"""Create shapefiles for the India plant disease demo dataset.

This script writes:
    - point shapefile: plant disease incidence locations
    - polygon shapefile: approximate India outline used by the demo map

It does not require geopandas, fiona, or pyshp.

Run:
    python3 create_india_plant_disease_shapefiles.py
"""

from __future__ import annotations

import datetime as dt
import struct
import zipfile
from pathlib import Path
from typing import Iterable

from plant_disease_india_map_demo import (
    india_outline_coordinates,
    make_demo_disease_data,
)


OUTPUT_DIR = Path("plant_disease_map_output")
SHAPE_DIR = OUTPUT_DIR / "shapefiles"
WGS84_PRJ = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)


def write_shp_header(
    file_obj,
    shape_type: int,
    file_length_words: int,
    bbox: tuple[float, float, float, float],
) -> None:
    """Write the fixed 100-byte shapefile or shx header."""
    xmin, ymin, xmax, ymax = bbox
    file_obj.write(struct.pack(">i", 9994))
    file_obj.write(struct.pack(">5i", 0, 0, 0, 0, 0))
    file_obj.write(struct.pack(">i", file_length_words))
    file_obj.write(struct.pack("<i", 1000))
    file_obj.write(struct.pack("<i", shape_type))
    file_obj.write(struct.pack("<4d", xmin, ymin, xmax, ymax))
    file_obj.write(struct.pack("<4d", 0.0, 0.0, 0.0, 0.0))


def write_dbf(
    path: Path,
    fields: list[tuple[str, str, int, int]],
    records: list[dict[str, object]],
) -> None:
    """Write a small dBASE III table for shapefile attributes."""
    today = dt.date.today()
    header_length = 32 + len(fields) * 32 + 1
    record_length = 1 + sum(field[2] for field in fields)

    with path.open("wb") as dbf:
        dbf.write(
            struct.pack(
                "<BBBBIHH20x",
                0x03,
                today.year - 1900,
                today.month,
                today.day,
                len(records),
                header_length,
                record_length,
            )
        )

        for name, field_type, width, decimals in fields:
            encoded_name = name.encode("ascii")[:10]
            dbf.write(encoded_name.ljust(11, b"\x00"))
            dbf.write(field_type.encode("ascii"))
            dbf.write(b"\x00\x00\x00\x00")
            dbf.write(struct.pack("BB", width, decimals))
            dbf.write(b"\x00" * 14)

        dbf.write(b"\r")

        for record in records:
            dbf.write(b" ")
            for name, field_type, width, decimals in fields:
                value = record.get(name, "")
                if field_type == "N":
                    if isinstance(value, float):
                        text = f"{value:>{width}.{decimals}f}"
                    else:
                        text = f"{int(value):>{width}d}"
                else:
                    text = str(value)[:width].ljust(width)
                dbf.write(text.encode("ascii", errors="replace"))

        dbf.write(b"\x1a")


def write_prj(path: Path) -> None:
    path.write_text(WGS84_PRJ + "\n", encoding="ascii")


def write_point_shapefile(
    basename: Path,
    points: list[dict[str, object]],
    fields: list[tuple[str, str, int, int]],
) -> None:
    """Write a WGS84 point shapefile."""
    shape_type = 1
    xs = [float(point["lon"]) for point in points]
    ys = [float(point["lat"]) for point in points]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    record_content_length_words = 10
    shp_file_length_words = 50 + len(points) * (4 + record_content_length_words)
    shx_file_length_words = 50 + len(points) * 4

    with basename.with_suffix(".shp").open("wb") as shp, basename.with_suffix(".shx").open("wb") as shx:
        write_shp_header(shp, shape_type, shp_file_length_words, bbox)
        write_shp_header(shx, shape_type, shx_file_length_words, bbox)

        offset_words = 50
        for record_number, point in enumerate(points, start=1):
            x = float(point["lon"])
            y = float(point["lat"])

            shp.write(struct.pack(">2i", record_number, record_content_length_words))
            shp.write(struct.pack("<i2d", shape_type, x, y))

            shx.write(struct.pack(">2i", offset_words, record_content_length_words))
            offset_words += 4 + record_content_length_words

    write_dbf(basename.with_suffix(".dbf"), fields, points)
    write_prj(basename.with_suffix(".prj"))


def write_polygon_shapefile(
    basename: Path,
    polygons: list[dict[str, object]],
    fields: list[tuple[str, str, int, int]],
) -> None:
    """Write a WGS84 polygon shapefile with single-part polygons."""
    shape_type = 5
    all_points = [point for polygon in polygons for point in polygon["points"]]
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    content_lengths_words = []
    for polygon in polygons:
        num_points = len(polygon["points"])
        content_bytes = 4 + 32 + 4 + 4 + 4 + num_points * 16
        content_lengths_words.append(content_bytes // 2)

    shp_file_length_words = 50 + sum(4 + length for length in content_lengths_words)
    shx_file_length_words = 50 + len(polygons) * 4

    with basename.with_suffix(".shp").open("wb") as shp, basename.with_suffix(".shx").open("wb") as shx:
        write_shp_header(shp, shape_type, shp_file_length_words, bbox)
        write_shp_header(shx, shape_type, shx_file_length_words, bbox)

        offset_words = 50
        for record_number, (polygon, content_length_words) in enumerate(
            zip(polygons, content_lengths_words),
            start=1,
        ):
            points = polygon["points"]
            poly_xs = [point[0] for point in points]
            poly_ys = [point[1] for point in points]
            poly_bbox = (min(poly_xs), min(poly_ys), max(poly_xs), max(poly_ys))

            shp.write(struct.pack(">2i", record_number, content_length_words))
            shp.write(struct.pack("<i", shape_type))
            shp.write(struct.pack("<4d", *poly_bbox))
            shp.write(struct.pack("<2i", 1, len(points)))
            shp.write(struct.pack("<i", 0))
            for x, y in points:
                shp.write(struct.pack("<2d", x, y))

            shx.write(struct.pack(">2i", offset_words, content_length_words))
            offset_words += 4 + content_length_words

    write_dbf(basename.with_suffix(".dbf"), fields, polygons)
    write_prj(basename.with_suffix(".prj"))


def zip_shapefile(basename: Path) -> Path:
    zip_path = basename.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for suffix in (".shp", ".shx", ".dbf", ".prj"):
            path = basename.with_suffix(suffix)
            archive.write(path, arcname=path.name)
    return zip_path


def make_point_records() -> list[dict[str, object]]:
    data = make_demo_disease_data()
    records = []
    for _, row in data.iterrows():
        records.append(
            {
                "site": row["site"],
                "state": row["state"],
                "loctype": row["location_type"],
                "crop": row["crop"],
                "disease": row["disease"],
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "inc_pct": int(row["incidence_percent"]),
                "note": row["data_note"],
            }
        )
    return records


def make_outline_records() -> list[dict[str, object]]:
    longitudes, latitudes = india_outline_coordinates()
    points = list(zip(longitudes, latitudes))
    if points[0] != points[-1]:
        points.append(points[0])
    return [
        {
            "name": "India outline",
            "note": "approximate demo outline",
            "points": points,
        }
    ]


def print_created_files(paths: Iterable[Path]) -> None:
    for path in paths:
        print(f"  - {path}")


def main() -> None:
    SHAPE_DIR.mkdir(parents=True, exist_ok=True)

    points_basename = SHAPE_DIR / "india_plant_disease_incidence_points"
    outline_basename = SHAPE_DIR / "india_approx_outline"

    point_fields = [
        ("site", "C", 24, 0),
        ("state", "C", 24, 0),
        ("loctype", "C", 12, 0),
        ("crop", "C", 20, 0),
        ("disease", "C", 32, 0),
        ("lat", "N", 11, 6),
        ("lon", "N", 11, 6),
        ("inc_pct", "N", 5, 0),
        ("note", "C", 40, 0),
    ]
    outline_fields = [
        ("name", "C", 24, 0),
        ("note", "C", 40, 0),
    ]

    write_point_shapefile(points_basename, make_point_records(), point_fields)
    write_polygon_shapefile(outline_basename, make_outline_records(), outline_fields)

    point_zip = zip_shapefile(points_basename)
    outline_zip = zip_shapefile(outline_basename)

    created = sorted(SHAPE_DIR.glob("*"))
    print("Shapefiles created.")
    print(f"Output folder: {SHAPE_DIR.resolve()}")
    print("\nCreated files:")
    print_created_files(created)
    print("\nZIP bundles:")
    print(f"  - {point_zip}")
    print(f"  - {outline_zip}")


if __name__ == "__main__":
    main()
