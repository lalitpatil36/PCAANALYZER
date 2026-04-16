"""Map citrus tristeza virus demo layers over real India boundaries.

This script creates a plant disease GIS layer focused on Citrus tristeza virus
(CTV) in central and south India. It uses:
    - real GADM India ADM1 boundaries
    - a synthetic CTV risk raster layer
    - source-noted demo survey points by citrus cultivar and mild/severe strain

The point data are illustrative training/demo records informed by published
CTV and Indian citrus literature. They are not field measurements.

Run:
    python3 india_ctv_citrus_disease_layer_map.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Polygon

from india_real_boundary_map import ADM1_BASENAME, ensure_gadm_files, read_adm1_polygons


OUTPUT_DIR = Path("india_ctv_citrus_disease_layer_output")
RASTER_PATH = OUTPUT_DIR / "demo_ctv_citrus_risk_raster.asc"


STRAIN_COLORS = {
    "Mild": "#2f80ed",
    "Severe": "#d7191c",
}


CULTIVAR_MARKERS = {
    "Mandarin": "o",
    "Sweet orange": "s",
    "Acid lime": "^",
    "Lemon": "D",
}


SOURCE_NOTES = [
    {
        "topic": "CTV strain severity",
        "note": "CTV strains range from mild/asymptomatic to severe syndromes such as quick decline, stem pitting, seedling yellows, and vein clearing.",
        "source": "Frontiers in Plant Science, 2017",
        "url": "https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2017.01419/full",
    },
    {
        "topic": "India CTV diversity",
        "note": "Indian CTV isolates have been sampled from Vidarbha, Bangalore, Darjeeling Hills, and Delhi; many Indian isolates grouped with VT severe stem-pitting clades.",
        "source": "USDA ARS publication summary",
        "url": "https://www.ars.usda.gov/research/publications/publication/?seqNo115=276584",
    },
    {
        "topic": "Indian CTV locations",
        "note": "CTV-infected samples were collected from Bangalore, Delhi, Nagpur, and Pune and maintained in Kagzi lime; isolates produced vein clearing and flecking symptoms.",
        "source": "Roy et al., Archives of Virology, PubMed abstract",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12664295/",
    },
    {
        "topic": "Cultivars and regions",
        "note": "Nagpur mandarin is centered in Vidarbha and adjoining Madhya Pradesh; acid lime is grown in Maharashtra, Tamil Nadu, Karnataka, Telangana, and Andhra Pradesh; Mosambi is grown in Marathwada.",
        "source": "ICAR Central Citrus Research Institute QRT document",
        "url": "https://www.icar.gov.in/sites/default/files/2022-07/QRT-NAGPUR-01.pdf",
    },
    {
        "topic": "CTV symptoms and management",
        "note": "Acid lime is susceptible and can show vein flecks; mild-strain preimmunization is mentioned for acid lime management.",
        "source": "TNAU Agritech Portal",
        "url": "https://agritech.tnau.ac.in/crop_protection/citrus_diseases_4.html",
    },
]


def make_demo_ctv_records() -> pd.DataFrame:
    """Create source-noted demo CTV survey records for central and south India."""
    records = [
        {
            "site": "Nagpur",
            "state": "Maharashtra",
            "zone": "Central India",
            "latitude": 21.1458,
            "longitude": 79.0882,
            "cultivar": "Nagpur mandarin",
            "cultivar_group": "Mandarin",
            "strain_class": "Severe",
            "dominant_syndrome": "stem pitting / decline risk",
            "demo_incidence_percent": 38,
            "evidence_basis": "Indian CTV isolate location and Vidarbha mandarin belt; demo severity assignment.",
        },
        {
            "site": "Amravati",
            "state": "Maharashtra",
            "zone": "Central India",
            "latitude": 20.9374,
            "longitude": 77.7796,
            "cultivar": "Nagpur mandarin",
            "cultivar_group": "Mandarin",
            "strain_class": "Mild",
            "dominant_syndrome": "mild vein clearing",
            "demo_incidence_percent": 22,
            "evidence_basis": "Vidarbha citrus belt; illustrative mild CTV layer point.",
        },
        {
            "site": "Aurangabad",
            "state": "Maharashtra",
            "zone": "Central India",
            "latitude": 19.8762,
            "longitude": 75.3433,
            "cultivar": "Mosambi sweet orange",
            "cultivar_group": "Sweet orange",
            "strain_class": "Severe",
            "dominant_syndrome": "quick decline risk on susceptible rootstock",
            "demo_incidence_percent": 34,
            "evidence_basis": "Mosambi region from ICAR-CCRI document; severe CTV syndrome modeled for demo.",
        },
        {
            "site": "Pune",
            "state": "Maharashtra",
            "zone": "Central India",
            "latitude": 18.5204,
            "longitude": 73.8567,
            "cultivar": "Kagzi acid lime",
            "cultivar_group": "Acid lime",
            "strain_class": "Severe",
            "dominant_syndrome": "vein clearing / stem pitting",
            "demo_incidence_percent": 45,
            "evidence_basis": "Indian CTV isolate location; acid lime susceptibility noted in extension literature.",
        },
        {
            "site": "Chhindwara",
            "state": "Madhya Pradesh",
            "zone": "Central India",
            "latitude": 22.0574,
            "longitude": 78.9382,
            "cultivar": "Nagpur mandarin",
            "cultivar_group": "Mandarin",
            "strain_class": "Mild",
            "dominant_syndrome": "mild or latent infection",
            "demo_incidence_percent": 18,
            "evidence_basis": "Adjoining Nagpur mandarin belt; illustrative mild CTV layer point.",
        },
        {
            "site": "Nanded",
            "state": "Maharashtra",
            "zone": "Central India",
            "latitude": 19.1383,
            "longitude": 77.3210,
            "cultivar": "Mosambi sweet orange",
            "cultivar_group": "Sweet orange",
            "strain_class": "Mild",
            "dominant_syndrome": "mild vein flecking",
            "demo_incidence_percent": 20,
            "evidence_basis": "Marathwada sweet orange region; illustrative mild CTV layer point.",
        },
        {
            "site": "Nalgonda",
            "state": "Telangana",
            "zone": "South India",
            "latitude": 17.0575,
            "longitude": 79.2684,
            "cultivar": "Sathgudi sweet orange",
            "cultivar_group": "Sweet orange",
            "strain_class": "Severe",
            "dominant_syndrome": "decline / stem pitting risk",
            "demo_incidence_percent": 41,
            "evidence_basis": "South-central sweet orange belt; severe CTV syndrome modeled for demo.",
        },
        {
            "site": "Kadapa",
            "state": "Andhra Pradesh",
            "zone": "South India",
            "latitude": 14.4673,
            "longitude": 78.8242,
            "cultivar": "Sathgudi sweet orange",
            "cultivar_group": "Sweet orange",
            "strain_class": "Severe",
            "dominant_syndrome": "stem pitting / decline risk",
            "demo_incidence_percent": 44,
            "evidence_basis": "Andhra Pradesh citrus region; severe CTV syndrome modeled for demo.",
        },
        {
            "site": "Anantapur",
            "state": "Andhra Pradesh",
            "zone": "South India",
            "latitude": 14.6819,
            "longitude": 77.6006,
            "cultivar": "Acid lime",
            "cultivar_group": "Acid lime",
            "strain_class": "Mild",
            "dominant_syndrome": "mild vein flecking",
            "demo_incidence_percent": 24,
            "evidence_basis": "Acid lime region from ICAR-CCRI document; illustrative mild point.",
        },
        {
            "site": "Bengaluru",
            "state": "Karnataka",
            "zone": "South India",
            "latitude": 12.9716,
            "longitude": 77.5946,
            "cultivar": "Acid lime",
            "cultivar_group": "Acid lime",
            "strain_class": "Severe",
            "dominant_syndrome": "vein clearing / stem pitting",
            "demo_incidence_percent": 48,
            "evidence_basis": "Indian CTV isolate location; acid lime susceptibility noted in extension literature.",
        },
        {
            "site": "Kodagu",
            "state": "Karnataka",
            "zone": "South India",
            "latitude": 12.3375,
            "longitude": 75.8069,
            "cultivar": "Coorg mandarin",
            "cultivar_group": "Mandarin",
            "strain_class": "Mild",
            "dominant_syndrome": "mild or latent infection",
            "demo_incidence_percent": 17,
            "evidence_basis": "Coorg mandarin region from Indian citrus cultivar literature; illustrative mild point.",
        },
        {
            "site": "Coimbatore",
            "state": "Tamil Nadu",
            "zone": "South India",
            "latitude": 11.0168,
            "longitude": 76.9558,
            "cultivar": "Acid lime",
            "cultivar_group": "Acid lime",
            "strain_class": "Severe",
            "dominant_syndrome": "vein flecks / yield reduction",
            "demo_incidence_percent": 52,
            "evidence_basis": "Tamil Nadu acid lime region; acid lime susceptibility noted by TNAU.",
        },
        {
            "site": "Dindigul",
            "state": "Tamil Nadu",
            "zone": "South India",
            "latitude": 10.3673,
            "longitude": 77.9803,
            "cultivar": "Lemon",
            "cultivar_group": "Lemon",
            "strain_class": "Mild",
            "dominant_syndrome": "mild vein clearing",
            "demo_incidence_percent": 19,
            "evidence_basis": "South India citrus production area; illustrative mild CTV layer point.",
        },
        {
            "site": "Palakkad",
            "state": "Kerala",
            "zone": "South India",
            "latitude": 10.7867,
            "longitude": 76.6548,
            "cultivar": "Lemon",
            "cultivar_group": "Lemon",
            "strain_class": "Mild",
            "dominant_syndrome": "mild or latent infection",
            "demo_incidence_percent": 15,
            "evidence_basis": "South India citrus production area; illustrative mild CTV layer point.",
        },
    ]
    return pd.DataFrame.from_records(records)


def write_ctv_ascii_raster(path: Path, ctv_data: pd.DataFrame) -> None:
    """Write a synthetic CTV risk raster as ESRI ASCII grid."""
    ncols = 300
    nrows = 310
    xllcorner = 66.0
    yllcorner = 5.0
    cellsize = 0.12
    nodata = -9999

    lon = xllcorner + (np.arange(ncols) + 0.5) * cellsize
    lat = yllcorner + (np.arange(nrows) + 0.5) * cellsize
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    risk = np.full_like(lon_grid, 8.0, dtype=float)
    for _, row in ctv_data.iterrows():
        severity_weight = 1.35 if row["strain_class"] == "Severe" else 0.72
        cultivar_weight = {"Acid lime": 1.25, "Sweet orange": 1.05, "Mandarin": 0.85, "Lemon": 0.75}[row["cultivar_group"]]
        amplitude = row["demo_incidence_percent"] * severity_weight * cultivar_weight
        spread_lon = 1.25 if row["zone"] == "Central India" else 1.05
        spread_lat = 1.05
        hotspot = amplitude * np.exp(
            -(
                ((lon_grid - row["longitude"]) / spread_lon) ** 2
                + ((lat_grid - row["latitude"]) / spread_lat) ** 2
            )
        )
        risk += hotspot

    central_citrus_belt = 14 * np.exp(-(((lon_grid - 78.5) / 4.8) ** 2 + ((lat_grid - 20.5) / 3.4) ** 2))
    south_citrus_belt = 12 * np.exp(-(((lon_grid - 77.8) / 3.8) ** 2 + ((lat_grid - 13.0) / 3.4) ** 2))
    risk = np.clip(risk + central_citrus_belt + south_citrus_belt, 0, 100)

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
    """Read an ESRI ASCII raster grid."""
    header = {}
    with path.open("r", encoding="ascii") as raster:
        for _ in range(6):
            key, value = raster.readline().split()
            header[key.lower()] = float(value)
        data = np.loadtxt(raster)
    return data, header


def make_compound_clip_patch(parts: pd.DataFrame, ax: plt.Axes) -> PathPatch:
    """Create a compound clip path from all India polygon parts."""
    vertices = []
    codes = []
    for points in parts["points"]:
        xy = np.asarray(points)
        if len(xy) < 3:
            continue
        vertices.extend(xy.tolist())
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(np.asarray(vertices), codes), transform=ax.transData)


def draw_boundaries(parts: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw state/UT boundaries."""
    for _, row in parts.iterrows():
        patch = Polygon(
            np.asarray(row["points"]),
            closed=True,
            facecolor="none",
            edgecolor="#232323",
            linewidth=0.32,
            antialiased=True,
        )
        ax.add_patch(patch)


def plot_ctv_map(
    parts: pd.DataFrame,
    ctv_data: pd.DataFrame,
    raster: np.ndarray,
    raster_header: dict[str, float],
    output_path: Path,
) -> None:
    """Plot the CTV raster layer and differentiated cultivar/strain points."""
    x_min = raster_header["xllcorner"]
    y_min = raster_header["yllcorner"]
    x_max = x_min + raster_header["ncols"] * raster_header["cellsize"]
    y_max = y_min + raster_header["nrows"] * raster_header["cellsize"]

    fig, ax = plt.subplots(figsize=(10.5, 10.0), facecolor="black")
    ax.set_facecolor("black")

    image = ax.imshow(
        raster,
        extent=(x_min, x_max, y_min, y_max),
        origin="upper",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        interpolation="bilinear",
        alpha=0.88,
    )
    image.set_clip_path(make_compound_clip_patch(parts, ax))

    draw_boundaries(parts, ax)

    focus_states = {
        "Maharashtra",
        "Madhya Pradesh",
        "Telangana",
        "Andhra Pradesh",
        "Karnataka",
        "Tamil Nadu",
        "Kerala",
    }
    focus_parts = parts[parts["name"].isin(focus_states)]
    for _, row in focus_parts.iterrows():
        patch = Polygon(
            np.asarray(row["points"]),
            closed=True,
            facecolor="none",
            edgecolor="#ffffff",
            linewidth=0.85,
            antialiased=True,
        )
        ax.add_patch(patch)

    for cultivar_group, marker in CULTIVAR_MARKERS.items():
        for strain_class, color in STRAIN_COLORS.items():
            subset = ctv_data[
                (ctv_data["cultivar_group"] == cultivar_group)
                & (ctv_data["strain_class"] == strain_class)
            ]
            if subset.empty:
                continue
            ax.scatter(
                subset["longitude"],
                subset["latitude"],
                s=55 + subset["demo_incidence_percent"] * 3.1,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.75,
                alpha=0.94,
                zorder=10,
            )

    for _, row in ctv_data.iterrows():
        label = f"{row['site']}\n{row['cultivar'].replace(' sweet orange', '')}"
        ax.annotate(
            label,
            (row["longitude"], row["latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=5.8,
            color="#f2f2f2",
            zorder=11,
        )

    strain_handles = [
        Line2D([0], [0], marker="o", color="none", label=strain, markerfacecolor=color, markeredgecolor="white", markersize=8)
        for strain, color in STRAIN_COLORS.items()
    ]
    cultivar_handles = [
        Line2D([0], [0], marker=marker, color="#f0f0f0", label=group, markerfacecolor="#999999", markeredgecolor="white", markersize=8, linestyle="none")
        for group, marker in CULTIVAR_MARKERS.items()
    ]
    first_legend = ax.legend(handles=strain_handles, title="CTV strain class", loc="lower left", frameon=True)
    first_legend.get_frame().set_facecolor("#111111")
    first_legend.get_frame().set_edgecolor("#777777")
    first_legend.get_title().set_color("#f2f2f2")
    for text in first_legend.get_texts():
        text.set_color("#f2f2f2")
    ax.add_artist(first_legend)

    second_legend = ax.legend(handles=cultivar_handles, title="Cultivar group", loc="lower right", frameon=True)
    second_legend.get_frame().set_facecolor("#111111")
    second_legend.get_frame().set_edgecolor("#777777")
    second_legend.get_title().set_color("#f2f2f2")
    for text in second_legend.get_texts():
        text.set_color("#f2f2f2")

    colorbar = fig.colorbar(image, ax=ax, fraction=0.034, pad=0.02)
    colorbar.set_label("Demo CTV risk index", color="#f2f2f2")
    colorbar.ax.yaxis.set_tick_params(color="#f2f2f2")
    plt.setp(colorbar.ax.get_yticklabels(), color="#f2f2f2")

    ax.set_title(
        "Citrus Tristeza Virus Demo Layer: Central and South India",
        color="#f2f2f2",
        fontsize=13,
        pad=10,
    )
    ax.text(
        67.0,
        5.25,
        "Demo layer: points and raster are illustrative; boundaries are GADM ADM1.",
        color="#d8d8d8",
        fontsize=7,
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


def summarize_ctv_data(ctv_data: pd.DataFrame) -> pd.DataFrame:
    """Summarize demo CTV records by cultivar group and strain class."""
    return (
        ctv_data.groupby(["cultivar_group", "strain_class"], as_index=False)
        .agg(
            sites=("site", "count"),
            mean_demo_incidence_percent=("demo_incidence_percent", "mean"),
            max_demo_incidence_percent=("demo_incidence_percent", "max"),
        )
        .round(2)
        .sort_values(["strain_class", "cultivar_group"])
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ensure_gadm_files()

    parts = read_adm1_polygons(ADM1_BASENAME)
    ctv_data = make_demo_ctv_records()
    source_notes = pd.DataFrame.from_records(SOURCE_NOTES)

    ctv_data_path = OUTPUT_DIR / "demo_ctv_citrus_survey_points.csv"
    source_notes_path = OUTPUT_DIR / "ctv_source_notes.csv"
    summary_path = OUTPUT_DIR / "ctv_summary_by_cultivar_and_strain.csv"
    map_path = OUTPUT_DIR / "india_ctv_citrus_strain_raster_map.png"

    ctv_data.to_csv(ctv_data_path, index=False)
    source_notes.to_csv(source_notes_path, index=False)
    summarize_ctv_data(ctv_data).to_csv(summary_path, index=False)

    write_ctv_ascii_raster(RASTER_PATH, ctv_data)
    raster, raster_header = read_ascii_raster(RASTER_PATH)
    plot_ctv_map(parts, ctv_data, raster, raster_header, map_path)

    print("India CTV citrus disease map layer complete.")
    print(f"Input boundary shapefile: {ADM1_BASENAME.with_suffix('.shp')}")
    print(f"Demo CTV survey records: {len(ctv_data)}")
    print(f"Severe records: {(ctv_data['strain_class'] == 'Severe').sum()}")
    print(f"Mild records: {(ctv_data['strain_class'] == 'Mild').sum()}")
    print(f"Raster shape: {raster.shape[0]} rows x {raster.shape[1]} columns")
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print(f"  - {map_path.name}")
    print(f"  - {RASTER_PATH.name}")
    print(f"  - {ctv_data_path.name}")
    print(f"  - {summary_path.name}")
    print(f"  - {source_notes_path.name}")
    print("\nNote: CTV points and raster values are demo data informed by literature, not field measurements.")


if __name__ == "__main__":
    main()
