"""Create a pastel India state-style map from demo coordinates.

This script makes a reference-style India map with:
    - black background
    - pastel state-like regions
    - thin internal boundaries
    - state labels
    - Lakshadweep and Andaman/Nicobar demo island markers

The India outline is imported from the existing demo shapefile. Internal
regions are illustrative Voronoi cells generated from demo state centroids,
so they are not official administrative boundaries.

Run:
    python3 india_state_style_map_demo.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path as MplPath
from scipy.spatial import Voronoi

from india_shp_raster_map_demo import ensure_demo_shapefiles, read_polygon_shapefile


OUTPUT_DIR = Path("india_state_style_map_output")
SOURCE_SHAPE_DIR = Path("plant_disease_map_output") / "shapefiles"


def make_demo_state_centroids() -> pd.DataFrame:
    """Create approximate state/UT label coordinates for a demo India map."""
    records = [
        ("Jammu and Kashmir", 76.6, 33.4),
        ("Himachal Pradesh", 77.2, 31.7),
        ("Punjab", 75.3, 30.9),
        ("Uttarakhand", 79.0, 30.1),
        ("Haryana", 76.2, 29.1),
        ("Delhi", 77.2, 28.6),
        ("Rajasthan", 73.7, 26.6),
        ("Uttar Pradesh", 80.9, 27.2),
        ("Bihar", 85.6, 25.8),
        ("Sikkim", 88.5, 27.5),
        ("Arunachal Pradesh", 94.5, 28.1),
        ("Assam", 92.0, 26.2),
        ("Meghalaya", 91.3, 25.5),
        ("Nagaland", 94.3, 26.1),
        ("Manipur", 93.9, 24.8),
        ("Mizoram", 92.8, 23.3),
        ("Tripura", 91.7, 23.8),
        ("West Bengal", 87.9, 23.3),
        ("Jharkhand", 85.3, 23.7),
        ("Odisha", 84.5, 20.5),
        ("Chhattisgarh", 82.0, 21.4),
        ("Madhya Pradesh", 78.2, 23.7),
        ("Gujarat", 71.9, 22.8),
        ("Maharashtra", 75.3, 19.4),
        ("Telangana", 79.1, 17.8),
        ("Andhra Pradesh", 80.0, 15.5),
        ("Karnataka", 76.2, 14.7),
        ("Goa", 74.0, 15.4),
        ("Kerala", 76.4, 10.3),
        ("Tamil Nadu", 78.5, 11.2),
    ]
    return pd.DataFrame(records, columns=["state", "longitude", "latitude"])


def voronoi_finite_polygons_2d(vor: Voronoi, radius: float = 100.0) -> tuple[list[list[int]], np.ndarray]:
    """Convert infinite Voronoi regions to finite regions for plotting."""
    if vor.points.shape[1] != 2:
        raise ValueError("This helper supports only 2D input.")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for point_index, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(vertex >= 0 for vertex in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[point_index]
        new_region = [vertex for vertex in vertices if vertex >= 0]

        for neighbor_index, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            tangent = vor.points[neighbor_index] - vor.points[point_index]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[point_index, neighbor_index]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        polygon = np.asarray([new_vertices[vertex] for vertex in new_region])
        centroid = polygon.mean(axis=0)
        angles = np.arctan2(polygon[:, 1] - centroid[1], polygon[:, 0] - centroid[0])
        new_region = [vertex for _, vertex in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def polygon_to_clip_patch(outline: pd.DataFrame) -> PathPatch:
    """Create a reusable clipping patch from the imported India outline."""
    vertices = outline[["longitude", "latitude"]].to_numpy()
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 2) + [MplPath.CLOSEPOLY]
    path = MplPath(vertices, codes)
    return PathPatch(path, transform=plt.gca().transData)


def plot_state_style_map(
    outline: pd.DataFrame,
    states: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a pastel India map similar to the provided reference image."""
    points = states[["longitude", "latitude"]].to_numpy()
    regions, vertices = voronoi_finite_polygons_2d(Voronoi(points), radius=80)

    pastel_colors = [
        "#cfe8bf",
        "#fff8b8",
        "#f8cfd0",
        "#ead2ef",
        "#d1e6c4",
        "#fff4a8",
        "#f6c5c7",
        "#e8c6ee",
    ]

    fig, ax = plt.subplots(figsize=(9.2, 8.6), facecolor="black")
    ax.set_facecolor("black")

    clip_patch = polygon_to_clip_patch(outline)
    ax.add_patch(clip_patch)
    clip_patch.set_visible(False)

    for idx, region in enumerate(regions):
        polygon_points = vertices[region]
        patch = Polygon(
            polygon_points,
            closed=True,
            facecolor=pastel_colors[idx % len(pastel_colors)],
            edgecolor="#7a7a7a",
            linewidth=0.65,
            alpha=1.0,
        )
        patch.set_clip_path(clip_patch)
        ax.add_patch(patch)

    ax.plot(
        outline["longitude"],
        outline["latitude"],
        color="#a5d39a",
        linewidth=1.15,
        solid_joinstyle="round",
    )

    island_groups = {
        "Lakshadweep": [(72.1, 10.5), (72.3, 11.0), (72.0, 11.6), (73.0, 8.3)],
        "Andaman and Nicobar": [(92.8, 13.5), (92.9, 12.0), (93.0, 10.7), (93.2, 9.2), (93.5, 7.2)],
    }
    for islands in island_groups.values():
        xs, ys = zip(*islands)
        ax.scatter(xs, ys, s=11, color="#cfe8bf", edgecolor="#a5d39a", linewidth=0.35)

    for _, row in states.iterrows():
        label = row["state"]
        fontsize = 5.8 if len(label) > 13 else 6.6
        ax.text(
            row["longitude"],
            row["latitude"],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#1f1f1f",
        )

    ax.text(72.5, 9.5, "Lakshadweep", ha="left", va="center", fontsize=5.8, color="#cfe8bf")
    ax.text(92.0, 11.2, "Andaman\nand Nicobar", ha="right", va="center", fontsize=5.8, color="#cfe8bf")

    ax.set_xlim(66.5, 97.8)
    ax.set_ylim(5.5, 36.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ensure_demo_shapefiles()

    outline_basename = SOURCE_SHAPE_DIR / "india_approx_outline"
    outline = read_polygon_shapefile(outline_basename)[0]
    states = make_demo_state_centroids()

    state_data_path = OUTPUT_DIR / "demo_state_label_coordinates.csv"
    map_path = OUTPUT_DIR / "india_pastel_state_style_map.png"

    states.to_csv(state_data_path, index=False)
    plot_state_style_map(outline, states, map_path)

    print("India state-style map demo complete.")
    print(f"Imported outline shapefile: {outline_basename.with_suffix('.shp')}")
    print(f"Demo state labels: {len(states)}")
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print(f"  - {state_data_path.name}")
    print(f"  - {map_path.name}")
    print("\nNote: internal state regions are illustrative demo polygons, not official boundaries.")


if __name__ == "__main__":
    main()
