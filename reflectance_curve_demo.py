"""Reflectance curve demo for healthy and infected plants.

This script creates synthetic hyperspectral-style reflectance data for
healthy and infected leaves, then saves a CSV dataset and reflectance plots.

Run:
    python3 reflectance_curve_demo.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_REPLICATES_PER_GROUP = 12
OUTPUT_DIR = Path("reflectance_curve_output")


def make_demo_reflectance_data() -> pd.DataFrame:
    """Create synthetic plant reflectance spectra with disease-related changes."""
    rng = np.random.default_rng(RANDOM_SEED)
    wavelengths = np.arange(400, 901, 5)

    healthy_curve = (
        0.05
        + 0.08 * np.exp(-((wavelengths - 550) / 55) ** 2)
        - 0.035 * np.exp(-((wavelengths - 680) / 30) ** 2)
        + 0.44 / (1 + np.exp(-(wavelengths - 720) / 16))
    )
    infected_curve = (
        0.07
        + 0.10 * np.exp(-((wavelengths - 560) / 65) ** 2)
        - 0.018 * np.exp(-((wavelengths - 675) / 34) ** 2)
        + 0.30 / (1 + np.exp(-(wavelengths - 735) / 22))
    )

    records = []
    for condition, base_curve in {"Healthy": healthy_curve, "Infected": infected_curve}.items():
        for replicate in range(1, N_REPLICATES_PER_GROUP + 1):
            replicate_shift = rng.normal(0, 0.008)
            noise = rng.normal(0, 0.006, size=wavelengths.size)
            reflectance = np.clip(base_curve + replicate_shift + noise, 0.01, 0.75)

            for wavelength, value in zip(wavelengths, reflectance):
                records.append(
                    {
                        "sample_id": f"{condition[:3]}_{replicate:02d}",
                        "condition": condition,
                        "wavelength_nm": wavelength,
                        "reflectance": value,
                    }
                )

    return pd.DataFrame.from_records(records)


def summarize_reflectance(reflectance_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and standard deviation for each condition and wavelength."""
    return (
        reflectance_df.groupby(["condition", "wavelength_nm"])["reflectance"]
        .agg(["mean", "std"])
        .reset_index()
    )


def calculate_demo_indices(reflectance_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple vegetation indices from demo red and NIR bands."""
    wide_df = reflectance_df.pivot_table(
        index=["sample_id", "condition"],
        columns="wavelength_nm",
        values="reflectance",
    ).reset_index()

    red = wide_df[680]
    green = wide_df[550]
    nir = wide_df[800]
    wide_df["NDVI_demo"] = (nir - red) / (nir + red)
    wide_df["green_red_ratio_demo"] = green / red

    return wide_df[["sample_id", "condition", "NDVI_demo", "green_red_ratio_demo"]]


def plot_mean_reflectance(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Save mean reflectance curves with one-standard-deviation bands."""
    colors = {"Healthy": "#2ca02c", "Infected": "#d62728"}

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for condition, subset in summary_df.groupby("condition"):
        wavelength = subset["wavelength_nm"].to_numpy()
        mean = subset["mean"].to_numpy()
        std = subset["std"].to_numpy()

        ax.plot(wavelength, mean, color=colors[condition], linewidth=2.5, label=condition)
        ax.fill_between(wavelength, mean - std, mean + std, color=colors[condition], alpha=0.18)

    ax.axvspan(630, 690, color="#f2c94c", alpha=0.18, label="Red absorption")
    ax.axvspan(760, 860, color="#56ccf2", alpha=0.14, label="NIR plateau")
    ax.set_title("Demo Plant Reflectance Curves")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_xlim(400, 900)
    ax.set_ylim(0, 0.6)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_replicate_reflectance(reflectance_df: pd.DataFrame, output_path: Path) -> None:
    """Save all individual sample curves to show replicate variation."""
    colors = {"Healthy": "#2ca02c", "Infected": "#d62728"}

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for (condition, sample_id), subset in reflectance_df.groupby(["condition", "sample_id"]):
        ax.plot(
            subset["wavelength_nm"],
            subset["reflectance"],
            color=colors[condition],
            alpha=0.28,
            linewidth=1.1,
        )

    for condition, subset in summarize_reflectance(reflectance_df).groupby("condition"):
        ax.plot(
            subset["wavelength_nm"],
            subset["mean"],
            color=colors[condition],
            linewidth=3,
            label=f"{condition} mean",
        )

    ax.set_title("Demo Reflectance Replicates")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_xlim(400, 900)
    ax.set_ylim(0, 0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    reflectance_df = make_demo_reflectance_data()
    summary_df = summarize_reflectance(reflectance_df)
    indices_df = calculate_demo_indices(reflectance_df)

    reflectance_df.to_csv(OUTPUT_DIR / "demo_reflectance_spectra.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "demo_reflectance_summary.csv", index=False)
    indices_df.to_csv(OUTPUT_DIR / "demo_reflectance_indices.csv", index=False)

    plot_mean_reflectance(summary_df, OUTPUT_DIR / "healthy_vs_infected_reflectance_curve.png")
    plot_replicate_reflectance(reflectance_df, OUTPUT_DIR / "reflectance_replicate_curves.png")

    print("Reflectance curve demo complete.")
    print(f"Samples: {reflectance_df['sample_id'].nunique()}")
    print(f"Wavelength range: {reflectance_df['wavelength_nm'].min()}-{reflectance_df['wavelength_nm'].max()} nm")
    print("\nMean demo indices:")
    print(indices_df.groupby("condition")[["NDVI_demo", "green_red_ratio_demo"]].mean().round(3))
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print("  - demo_reflectance_spectra.csv")
    print("  - demo_reflectance_summary.csv")
    print("  - demo_reflectance_indices.csv")
    print("  - healthy_vs_infected_reflectance_curve.png")
    print("  - reflectance_replicate_curves.png")


if __name__ == "__main__":
    main()
