"""Create a demo map of plant disease incidence across India.

The script uses real Indian place coordinates with synthetic disease
incidence values for demonstration, then saves a CSV dataset and PNG map.

Run:
    python3 plant_disease_india_map_demo.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("plant_disease_map_output")


def make_demo_disease_data() -> pd.DataFrame:
    """Create synthetic incidence data for real Indian locations."""
    records = [
        {
            "site": "Ludhiana",
            "state": "Punjab",
            "location_type": "city",
            "crop": "Wheat",
            "disease": "Leaf rust",
            "latitude": 30.9010,
            "longitude": 75.8573,
            "incidence_percent": 18,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Karnal",
            "state": "Haryana",
            "location_type": "city",
            "crop": "Wheat",
            "disease": "Leaf rust",
            "latitude": 29.6857,
            "longitude": 76.9905,
            "incidence_percent": 22,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Lucknow",
            "state": "Uttar Pradesh",
            "location_type": "city",
            "crop": "Rice",
            "disease": "Bacterial leaf blight",
            "latitude": 26.8467,
            "longitude": 80.9462,
            "incidence_percent": 31,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Patna",
            "state": "Bihar",
            "location_type": "city",
            "crop": "Rice",
            "disease": "Sheath blight",
            "latitude": 25.5941,
            "longitude": 85.1376,
            "incidence_percent": 44,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Guwahati",
            "state": "Assam",
            "location_type": "city",
            "crop": "Rice",
            "disease": "Blast",
            "latitude": 26.1445,
            "longitude": 91.7362,
            "incidence_percent": 57,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Kolkata",
            "state": "West Bengal",
            "location_type": "city",
            "crop": "Rice",
            "disease": "Blast",
            "latitude": 22.5726,
            "longitude": 88.3639,
            "incidence_percent": 49,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Bhubaneswar",
            "state": "Odisha",
            "location_type": "city",
            "crop": "Rice",
            "disease": "Brown spot",
            "latitude": 20.2961,
            "longitude": 85.8245,
            "incidence_percent": 36,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Raipur",
            "state": "Chhattisgarh",
            "location_type": "city",
            "crop": "Rice",
            "disease": "Sheath blight",
            "latitude": 21.2514,
            "longitude": 81.6296,
            "incidence_percent": 41,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Nagpur",
            "state": "Maharashtra",
            "location_type": "city",
            "crop": "Cotton",
            "disease": "Bacterial blight",
            "latitude": 21.1458,
            "longitude": 79.0882,
            "incidence_percent": 29,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Hyderabad",
            "state": "Telangana",
            "location_type": "city",
            "crop": "Cotton",
            "disease": "Wilt",
            "latitude": 17.3850,
            "longitude": 78.4867,
            "incidence_percent": 52,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Bengaluru",
            "state": "Karnataka",
            "location_type": "city",
            "crop": "Tomato",
            "disease": "Early blight",
            "latitude": 12.9716,
            "longitude": 77.5946,
            "incidence_percent": 47,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Coimbatore",
            "state": "Tamil Nadu",
            "location_type": "city",
            "crop": "Banana",
            "disease": "Sigatoka leaf spot",
            "latitude": 11.0168,
            "longitude": 76.9558,
            "incidence_percent": 63,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Kochi",
            "state": "Kerala",
            "location_type": "city",
            "crop": "Pepper",
            "disease": "Phytophthora foot rot",
            "latitude": 9.9312,
            "longitude": 76.2673,
            "incidence_percent": 58,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Ahmedabad",
            "state": "Gujarat",
            "location_type": "city",
            "crop": "Cotton",
            "disease": "Leaf curl",
            "latitude": 23.0225,
            "longitude": 72.5714,
            "incidence_percent": 34,
            "data_note": "real coordinate, demo incidence",
        },
        {
            "site": "Jaipur",
            "state": "Rajasthan",
            "location_type": "city",
            "crop": "Mustard",
            "disease": "Alternaria blight",
            "latitude": 26.9124,
            "longitude": 75.7873,
            "incidence_percent": 25,
            "data_note": "real coordinate, demo incidence",
        },
    ]

    return pd.DataFrame.from_records(records)


def india_outline_coordinates() -> tuple[list[float], list[float]]:
    """Return a coarse outline of mainland India for demo visualization."""
    outline = [
        (68.1, 23.7),
        (69.2, 22.3),
        (70.1, 20.9),
        (72.0, 19.0),
        (72.8, 15.8),
        (74.0, 13.0),
        (75.7, 10.4),
        (77.4, 8.1),
        (79.1, 9.5),
        (80.3, 12.2),
        (80.2, 15.8),
        (81.8, 18.0),
        (84.1, 19.2),
        (86.8, 20.8),
        (88.1, 22.0),
        (89.9, 22.1),
        (92.0, 24.0),
        (94.5, 25.2),
        (96.0, 27.0),
        (94.4, 28.1),
        (91.5, 27.8),
        (88.9, 26.9),
        (88.1, 27.9),
        (86.0, 27.9),
        (83.8, 28.7),
        (81.0, 30.1),
        (78.8, 32.2),
        (76.0, 34.3),
        (74.5, 34.8),
        (73.1, 33.6),
        (72.0, 31.0),
        (70.6, 28.5),
        (69.4, 26.1),
        (68.1, 23.7),
    ]
    longitudes, latitudes = zip(*outline)
    return list(longitudes), list(latitudes)


def plot_disease_map(data: pd.DataFrame, output_path: Path) -> None:
    """Create and save a plant disease incidence map."""
    outline_lons, outline_lats = india_outline_coordinates()

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.fill(outline_lons, outline_lats, color="#f3f0df", edgecolor="#555555", linewidth=1.3)

    point_sizes = 45 + data["incidence_percent"] * 4
    scatter = ax.scatter(
        data["longitude"],
        data["latitude"],
        c=data["incidence_percent"],
        s=point_sizes,
        cmap="YlOrRd",
        vmin=0,
        vmax=70,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.9,
    )

    for _, row in data.iterrows():
        ax.annotate(
            row["site"],
            (row["longitude"], row["latitude"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Disease incidence (%)")

    ax.set_title("Demo Plant Disease Incidence Across India")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(67, 98)
    ax.set_ylim(6, 36)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#cccccc", linestyle="--", linewidth=0.6, alpha=0.6)

    note = "Real place coordinates; incidence values are synthetic demo data."
    ax.text(67.2, 6.4, note, fontsize=8, color="#555555")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize_incidence(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean incidence by crop and disease."""
    return (
        data.groupby(["crop", "disease"], as_index=False)
        .agg(
            sites=("site", "count"),
            mean_incidence_percent=("incidence_percent", "mean"),
            max_incidence_percent=("incidence_percent", "max"),
        )
        .sort_values("mean_incidence_percent", ascending=False)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = make_demo_disease_data()
    summary = summarize_incidence(data)

    data_path = OUTPUT_DIR / "demo_plant_disease_incidence_india.csv"
    summary_path = OUTPUT_DIR / "incidence_summary_by_crop_disease.csv"
    map_path = OUTPUT_DIR / "india_plant_disease_incidence_map.png"

    data.to_csv(data_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_disease_map(data, map_path)

    print("Plant disease incidence map complete.")
    print(f"Sites mapped: {len(data)}")
    print(f"Mean incidence: {data['incidence_percent'].mean():.1f}%")
    print(f"Highest incidence: {data['incidence_percent'].max()}%")
    print("\nTop incidence records:")
    top_records = data.sort_values("incidence_percent", ascending=False).head(5)
    for _, row in top_records.iterrows():
        print(
            f"  {row['site']}, {row['state']}: "
            f"{row['crop']} - {row['disease']} ({row['incidence_percent']}%)"
        )

    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print(f"  - {data_path.name}")
    print(f"  - {summary_path.name}")
    print(f"  - {map_path.name}")


if __name__ == "__main__":
    main()
