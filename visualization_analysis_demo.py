"""Demo exploratory analysis and visualization workflow.

This script creates synthetic plant disease monitoring data, calculates
summary tables, and saves visualization-ready CSV files plus PNG charts.

Run:
    python3 visualization_analysis_demo.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED = 42
OUTPUT_DIR = Path("visualization_analysis_demo_output")


def make_demo_data() -> pd.DataFrame:
    """Create synthetic weekly disease observations for visualization analysis."""
    rng = np.random.default_rng(RANDOM_SEED)

    treatments = {
        "Control": {"base": 18.0, "weekly_growth": 5.2, "yield_penalty": 0.55},
        "Bio-control": {"base": 13.0, "weekly_growth": 3.4, "yield_penalty": 0.38},
        "Fungicide": {"base": 10.0, "weekly_growth": 2.5, "yield_penalty": 0.28},
        "Resistant variety": {"base": 8.0, "weekly_growth": 1.8, "yield_penalty": 0.18},
    }
    regions = {
        "North": {"humidity": 68, "temperature": 27.0},
        "South": {"humidity": 76, "temperature": 29.5},
        "East": {"humidity": 81, "temperature": 28.2},
        "West": {"humidity": 63, "temperature": 30.1},
    }

    records = []
    for region, climate in regions.items():
        for treatment, spec in treatments.items():
            for plot in range(1, 7):
                plot_effect = rng.normal(0, 2.2)
                for week in range(1, 7):
                    humidity = climate["humidity"] + rng.normal(0, 3.5)
                    temperature = climate["temperature"] + rng.normal(0, 1.1)
                    disease_pressure = max(0, humidity - 65) * 0.16 + max(0, temperature - 28) * 0.75
                    disease_index = (
                        spec["base"]
                        + spec["weekly_growth"] * week
                        + disease_pressure
                        + plot_effect
                        + rng.normal(0, 2.5)
                    )
                    disease_index = float(np.clip(disease_index, 0, 100))
                    yield_kg = (
                        72
                        - spec["yield_penalty"] * disease_index
                        + 0.18 * (100 - humidity)
                        + rng.normal(0, 3.0)
                    )

                    records.append(
                        {
                            "region": region,
                            "treatment": treatment,
                            "plot_id": f"{region[:1]}-{treatment[:3]}-{plot:02d}",
                            "week": week,
                            "humidity_percent": round(float(humidity), 1),
                            "temperature_c": round(float(temperature), 1),
                            "disease_index": round(disease_index, 2),
                            "yield_kg_per_plot": round(float(yield_kg), 2),
                        }
                    )

    return pd.DataFrame.from_records(records)


def summarize_by_treatment(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize final-week disease and yield by treatment."""
    final_week = data[data["week"] == data["week"].max()]
    return (
        final_week.groupby("treatment")
        .agg(
            plots=("plot_id", "nunique"),
            mean_disease_index=("disease_index", "mean"),
            sd_disease_index=("disease_index", "std"),
            mean_yield_kg_per_plot=("yield_kg_per_plot", "mean"),
            sd_yield_kg_per_plot=("yield_kg_per_plot", "std"),
        )
        .reset_index()
        .round(2)
    )


def summarize_weekly_trends(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize disease progression by treatment and week."""
    return (
        data.groupby(["treatment", "week"])["disease_index"]
        .agg(mean="mean", sd="std")
        .reset_index()
        .round(2)
    )


def summarize_region_treatment_heatmap(data: pd.DataFrame) -> pd.DataFrame:
    """Return final-week disease means for region by treatment heatmap plotting."""
    final_week = data[data["week"] == data["week"].max()]
    return (
        final_week.pivot_table(
            index="region",
            columns="treatment",
            values="disease_index",
            aggfunc="mean",
        )
        .round(2)
        .reset_index()
    )


def calculate_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations among numeric demo variables."""
    numeric_cols = ["humidity_percent", "temperature_c", "disease_index", "yield_kg_per_plot"]
    corr = data[numeric_cols].corr().round(3)
    return corr.reset_index(names="variable")


def plot_disease_trend(weekly_summary: pd.DataFrame, output_path: Path) -> None:
    """Save mean disease progression by treatment."""
    colors = {
        "Control": "#d62728",
        "Bio-control": "#2ca02c",
        "Fungicide": "#1f77b4",
        "Resistant variety": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for treatment, subset in weekly_summary.groupby("treatment"):
        week = subset["week"].to_numpy()
        mean = subset["mean"].to_numpy()
        sd = subset["sd"].to_numpy()
        ax.plot(week, mean, marker="o", linewidth=2.4, color=colors[treatment], label=treatment)
        ax.fill_between(week, mean - sd, mean + sd, color=colors[treatment], alpha=0.14)

    ax.set_title("Demo Disease Progression by Treatment")
    ax.set_xlabel("Week")
    ax.set_ylabel("Disease index")
    ax.set_xticks(sorted(weekly_summary["week"].unique()))
    ax.set_ylim(0, max(weekly_summary["mean"] + weekly_summary["sd"]) * 1.12)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_final_treatment_summary(treatment_summary: pd.DataFrame, output_path: Path) -> None:
    """Save final-week disease and yield comparison by treatment."""
    ordered = treatment_summary.sort_values("mean_disease_index", ascending=False)
    x = np.arange(len(ordered))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    disease_bars = ax1.bar(
        x - width / 2,
        ordered["mean_disease_index"],
        width,
        yerr=ordered["sd_disease_index"],
        color="#e45756",
        edgecolor="black",
        linewidth=0.7,
        capsize=4,
        label="Disease index",
    )
    yield_bars = ax2.bar(
        x + width / 2,
        ordered["mean_yield_kg_per_plot"],
        width,
        yerr=ordered["sd_yield_kg_per_plot"],
        color="#4c78a8",
        edgecolor="black",
        linewidth=0.7,
        capsize=4,
        label="Yield",
    )

    ax1.set_title("Final Week Disease and Yield")
    ax1.set_xlabel("Treatment")
    ax1.set_ylabel("Disease index")
    ax2.set_ylabel("Yield (kg per plot)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ordered["treatment"], rotation=15, ha="right")
    ax1.legend([disease_bars, yield_bars], ["Disease index", "Yield"], frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_disease_yield_scatter(data: pd.DataFrame, output_path: Path) -> None:
    """Save disease-yield relationship for final-week observations."""
    final_week = data[data["week"] == data["week"].max()]
    colors = {
        "Control": "#d62728",
        "Bio-control": "#2ca02c",
        "Fungicide": "#1f77b4",
        "Resistant variety": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(8, 5.8))
    for treatment, subset in final_week.groupby("treatment"):
        ax.scatter(
            subset["disease_index"],
            subset["yield_kg_per_plot"],
            s=70,
            alpha=0.82,
            color=colors[treatment],
            edgecolor="black",
            linewidth=0.5,
            label=treatment,
        )

    slope, intercept = np.polyfit(final_week["disease_index"], final_week["yield_kg_per_plot"], deg=1)
    x_line = np.linspace(final_week["disease_index"].min(), final_week["disease_index"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#333333", linewidth=2, linestyle="--")

    ax.set_title("Disease Severity vs Yield")
    ax.set_xlabel("Final disease index")
    ax.set_ylabel("Yield (kg per plot)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_region_treatment_heatmap(heatmap_summary: pd.DataFrame, output_path: Path) -> None:
    """Save a region by treatment heatmap for final-week disease."""
    matrix = heatmap_summary.set_index("region")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(matrix.to_numpy(), cmap="YlOrRd", aspect="auto")

    ax.set_title("Final Disease Index by Region and Treatment")
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Region")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", color="#222222", fontsize=9)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Disease index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    demo_data = make_demo_data()
    treatment_summary = summarize_by_treatment(demo_data)
    weekly_summary = summarize_weekly_trends(demo_data)
    heatmap_summary = summarize_region_treatment_heatmap(demo_data)
    correlations = calculate_correlations(demo_data)

    demo_data.to_csv(OUTPUT_DIR / "demo_visualization_data.csv", index=False)
    treatment_summary.to_csv(OUTPUT_DIR / "treatment_summary.csv", index=False)
    weekly_summary.to_csv(OUTPUT_DIR / "weekly_disease_summary.csv", index=False)
    heatmap_summary.to_csv(OUTPUT_DIR / "region_treatment_heatmap.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "numeric_correlations.csv", index=False)

    plot_disease_trend(weekly_summary, OUTPUT_DIR / "disease_progression_by_treatment.png")
    plot_final_treatment_summary(treatment_summary, OUTPUT_DIR / "final_disease_yield_by_treatment.png")
    plot_disease_yield_scatter(demo_data, OUTPUT_DIR / "disease_vs_yield_scatter.png")
    plot_region_treatment_heatmap(heatmap_summary, OUTPUT_DIR / "region_treatment_heatmap.png")

    final_week = demo_data[demo_data["week"] == demo_data["week"].max()]
    disease_yield_corr = final_week["disease_index"].corr(final_week["yield_kg_per_plot"])

    print("Visualization analysis demo complete.")
    print(f"Rows: {len(demo_data)}")
    print(f"Regions: {demo_data['region'].nunique()}")
    print(f"Treatments: {demo_data['treatment'].nunique()}")
    print(f"Final-week disease/yield correlation: {disease_yield_corr:.3f}")
    print("\nFinal-week treatment summary:")
    print(treatment_summary.to_string(index=False))
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print("  - demo_visualization_data.csv")
    print("  - treatment_summary.csv")
    print("  - weekly_disease_summary.csv")
    print("  - region_treatment_heatmap.csv")
    print("  - numeric_correlations.csv")
    print("  - disease_progression_by_treatment.png")
    print("  - final_disease_yield_by_treatment.png")
    print("  - disease_vs_yield_scatter.png")
    print("  - region_treatment_heatmap.png")


if __name__ == "__main__":
    main()
