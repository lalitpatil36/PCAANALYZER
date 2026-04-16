"""Pooled one-way ANOVA demo using synthetic treatment data.

This script creates demo crop-yield data for four fertilizer treatments,
runs a one-way ANOVA using the pooled within-group error variance, and saves
CSV tables plus plots.

Run:
    python3 pooled_anova_demo.py
"""

from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RANDOM_SEED = 42
OUTPUT_DIR = Path("pooled_anova_demo_output")


def make_demo_data() -> pd.DataFrame:
    """Create balanced demo data with different treatment means."""
    rng = np.random.default_rng(RANDOM_SEED)
    treatment_specs = {
        "Control": 48.0,
        "Low": 52.0,
        "Medium": 58.0,
        "High": 61.0,
    }

    records = []
    for treatment, mean_yield in treatment_specs.items():
        values = rng.normal(loc=mean_yield, scale=4.2, size=10)
        for replicate, value in enumerate(values, start=1):
            records.append(
                {
                    "treatment": treatment,
                    "replicate": replicate,
                    "yield_kg_per_plot": round(float(value), 2),
                }
            )

    return pd.DataFrame(records)


def summarize_groups(data: pd.DataFrame) -> pd.DataFrame:
    """Return sample size, mean, standard deviation, and standard error."""
    summary = (
        data.groupby("treatment")["yield_kg_per_plot"]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    return summary


def run_pooled_anova(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run one-way ANOVA and expose the pooled within-group variance."""
    value_col = "yield_kg_per_plot"
    group_col = "treatment"

    group_summary = summarize_groups(data)
    grand_mean = data[value_col].mean()

    ss_between = float(
        (group_summary["n"] * (group_summary["mean"] - grand_mean) ** 2).sum()
    )
    ss_within = float(
        data.groupby(group_col)[value_col]
        .apply(lambda x: ((x - x.mean()) ** 2).sum())
        .sum()
    )
    ss_total = float(((data[value_col] - grand_mean) ** 2).sum())

    df_between = int(group_summary.shape[0] - 1)
    df_within = int(data.shape[0] - group_summary.shape[0])
    df_total = int(data.shape[0] - 1)

    ms_between = ss_between / df_between
    pooled_variance = ss_within / df_within
    pooled_sd = np.sqrt(pooled_variance)
    f_stat = ms_between / pooled_variance
    p_value = float(stats.f.sf(f_stat, df_between, df_within))

    eta_squared = ss_between / ss_total
    omega_squared = (ss_between - df_between * pooled_variance) / (
        ss_total + pooled_variance
    )

    anova_table = pd.DataFrame(
        [
            {
                "source": "Between treatments",
                "df": df_between,
                "sum_sq": ss_between,
                "mean_sq": ms_between,
                "f": f_stat,
                "p_value": p_value,
            },
            {
                "source": "Within treatments / pooled error",
                "df": df_within,
                "sum_sq": ss_within,
                "mean_sq": pooled_variance,
                "f": np.nan,
                "p_value": np.nan,
            },
            {
                "source": "Total",
                "df": df_total,
                "sum_sq": ss_total,
                "mean_sq": np.nan,
                "f": np.nan,
                "p_value": np.nan,
            },
        ]
    )

    diagnostics = {
        "grand_mean": float(grand_mean),
        "pooled_variance_mse": float(pooled_variance),
        "pooled_sd_rmse": float(pooled_sd),
        "eta_squared": float(eta_squared),
        "omega_squared": float(max(0.0, omega_squared)),
    }
    return anova_table, diagnostics


def run_pairwise_pooled_tests(data: pd.DataFrame, pooled_variance: float) -> pd.DataFrame:
    """Compare treatment means using the ANOVA pooled error term."""
    summary = summarize_groups(data).set_index("treatment")
    df_error = int(data.shape[0] - summary.shape[0])
    rows = []

    for first, second in combinations(summary.index, 2):
        first_mean = summary.loc[first, "mean"]
        second_mean = summary.loc[second, "mean"]
        first_n = summary.loc[first, "n"]
        second_n = summary.loc[second, "n"]

        mean_diff = first_mean - second_mean
        se_diff = np.sqrt(pooled_variance * (1 / first_n + 1 / second_n))
        t_stat = mean_diff / se_diff
        p_value = float(2 * stats.t.sf(abs(t_stat), df_error))

        rows.append(
            {
                "comparison": f"{first} - {second}",
                "mean_difference": mean_diff,
                "pooled_se": se_diff,
                "t": t_stat,
                "df_error": df_error,
                "p_value": p_value,
                "bonferroni_p_value": min(p_value * 6, 1.0),
            }
        )

    return pd.DataFrame(rows)


def plot_group_means(summary: pd.DataFrame, pooled_sd: float, output_path: Path) -> None:
    """Save a treatment mean plot with ordinary SE bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4c78a8", "#54a24b", "#f58518", "#e45756"]

    ax.bar(
        summary["treatment"],
        summary["mean"],
        yerr=summary["se"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
    )
    ax.axhline(summary["mean"].mean(), color="#666666", linestyle="--", linewidth=1)
    ax.set_title("Demo Treatment Means for Pooled ANOVA")
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Yield (kg per plot)")
    ax.text(
        0.01,
        0.98,
        f"Pooled SD = {pooled_sd:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.3"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_raw_data(data: pd.DataFrame, output_path: Path) -> None:
    """Save a jittered raw-data plot by treatment group."""
    rng = np.random.default_rng(RANDOM_SEED)
    treatments = list(data["treatment"].drop_duplicates())
    x_lookup = {treatment: idx for idx, treatment in enumerate(treatments)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for treatment in treatments:
        subset = data[data["treatment"] == treatment]
        x = x_lookup[treatment] + rng.uniform(-0.12, 0.12, size=len(subset))
        ax.scatter(
            x,
            subset["yield_kg_per_plot"],
            color="#4c78a8",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            s=55,
        )

    ax.set_xticks(range(len(treatments)), treatments)
    ax.set_title("Demo Yield Observations by Treatment")
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Yield (kg per plot)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = make_demo_data()
    summary = summarize_groups(data)
    anova_table, diagnostics = run_pooled_anova(data)
    pairwise_table = run_pairwise_pooled_tests(
        data, diagnostics["pooled_variance_mse"]
    )

    data.to_csv(OUTPUT_DIR / "demo_pooled_anova_data.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "group_summary.csv", index=False)
    anova_table.to_csv(OUTPUT_DIR / "pooled_anova_table.csv", index=False)
    pairwise_table.to_csv(OUTPUT_DIR / "pairwise_pooled_tests.csv", index=False)
    pd.Series(diagnostics, name="value").to_csv(OUTPUT_DIR / "anova_diagnostics.csv")

    plot_group_means(summary, diagnostics["pooled_sd_rmse"], OUTPUT_DIR / "group_means.png")
    plot_raw_data(data, OUTPUT_DIR / "raw_data_by_group.png")

    print("Pooled one-way ANOVA complete.")
    print(f"Observations: {len(data)}")
    print(f"Groups: {summary.shape[0]}")
    print(f"Grand mean: {diagnostics['grand_mean']:.2f}")
    print(f"Pooled variance / MSE: {diagnostics['pooled_variance_mse']:.3f}")
    print(f"Pooled SD / RMSE: {diagnostics['pooled_sd_rmse']:.3f}")

    test_row = anova_table.loc[anova_table["source"] == "Between treatments"].iloc[0]
    print(
        "\nANOVA result: "
        f"F({int(test_row['df'])}, {int(anova_table.loc[1, 'df'])}) = "
        f"{test_row['f']:.3f}, p = {test_row['p_value']:.6f}"
    )
    print(f"Effect size eta squared: {diagnostics['eta_squared']:.3f}")
    print(f"Effect size omega squared: {diagnostics['omega_squared']:.3f}")

    print("\nGroup summary:")
    print(summary.round(3).to_string(index=False))

    print("\nANOVA table:")
    print(anova_table.round(6).to_string(index=False))

    print("\nPairwise tests using pooled error variance:")
    print(pairwise_table.round(6).to_string(index=False))

    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print("  - demo_pooled_anova_data.csv")
    print("  - group_summary.csv")
    print("  - pooled_anova_table.csv")
    print("  - pairwise_pooled_tests.csv")
    print("  - anova_diagnostics.csv")
    print("  - group_means.png")
    print("  - raw_data_by_group.png")


if __name__ == "__main__":
    main()
