"""Advanced DNA sequence analysis demo.

This script uses one built-in demo DNA sequence and runs a compact analysis
pipeline: base composition, sliding-window GC/GC-skew/entropy, CpG islands,
motif scanning, six-frame ORF discovery, codon usage for the longest ORF, and
k-mer enrichment. CSV tables and summary plots are saved to an output folder.

Run:
    python3 advanced_sequence_analysis_demo.py
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("sequence_analysis_demo_output")
STOP_CODONS = {"TAA", "TAG", "TGA"}
START_CODON = "ATG"

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

MOTIF_PATTERNS = {
    "TATA_box": "TATA[AT]A[AT]",
    "Kozak_like_start": "GCC[AG]CCATGG",
    "EcoRI_site": "GAATTC",
    "poly_A_signal": "AATAAA",
    "CpG_dinucleotide": "CG",
}


def make_demo_sequence() -> str:
    """Return a deterministic demo sequence with known biological features."""
    promoter = (
        "GCGCGTTCGCGCGATTCGCGCGTATAAAAGGCTGACGCGGCCGCCGATCGCGCG"
        "TTGACATCGATCGCGGCGCGATTAATAAAGCTTGAATTCGCGCGCGTTCGCGCG"
    )
    spacer = "ATATATATATATGCGCGCGCGCGCGTTAACCGGTTAACCGGTTAA"
    coding_one = (
        "GCCACCATGGCTGCTGAAAAGGCTGCTGCTGTTGACGACGAACTGCTGGAAGGT"
        "TTCGACGCTAACCGTGCTGCTGCTAAAGGTGACCTGAAAGCTGCTGACGTTGCT"
        "GCTGCTGGTGACAAAGCTGCTGAACTGGCTGACGCTGCTGCTAAATAG"
    )
    repeat_rich = "GATCGATCGATCGATCGATCGATCGATCGATC"
    coding_two = (
        "CCGCGCATGCCGGAACCGGCTGCTGCTCGTCGTCGTGAAAACGCTGACGACGAC"
        "GCTGCTGGATGCTGCTGCTGAAGCTGCTGCTGCTTGA"
    )
    tail = (
        "TTTTATATATATATATATATATGCGTACGTAGCTAGCTAGGATCCGCGCG"
        "AATAAATTTGCGCGCGCGCGCGATATATATATATCGCGCG"
    )
    return (promoter + spacer + coding_one + repeat_rich + coding_two + tail).upper()


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def gc_content(sequence: str) -> float:
    return 100.0 * (sequence.count("G") + sequence.count("C")) / len(sequence)


def gc_skew(sequence: str) -> float:
    g_count = sequence.count("G")
    c_count = sequence.count("C")
    total = g_count + c_count
    return 0.0 if total == 0 else (g_count - c_count) / total


def shannon_entropy(sequence: str) -> float:
    counts = Counter(sequence)
    total = len(sequence)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def cpg_observed_expected(sequence: str) -> float:
    c_count = sequence.count("C")
    g_count = sequence.count("G")
    if c_count == 0 or g_count == 0:
        return 0.0
    observed = sequence.count("CG")
    return observed * len(sequence) / (c_count * g_count)


def sliding_window_metrics(sequence: str, window: int = 80, step: int = 20) -> pd.DataFrame:
    rows = []
    for start in range(0, len(sequence) - window + 1, step):
        chunk = sequence[start : start + window]
        rows.append(
            {
                "start": start + 1,
                "end": start + window,
                "midpoint": start + window / 2,
                "gc_percent": gc_content(chunk),
                "gc_skew": gc_skew(chunk),
                "cpg_observed_expected": cpg_observed_expected(chunk),
                "shannon_entropy": shannon_entropy(chunk),
            }
        )
    return pd.DataFrame(rows)


def identify_cpg_islands(window_df: pd.DataFrame) -> pd.DataFrame:
    island_windows = window_df[
        (window_df["gc_percent"] >= 50.0) & (window_df["cpg_observed_expected"] >= 0.6)
    ].copy()
    if island_windows.empty:
        return pd.DataFrame(columns=["start", "end", "window_count", "mean_gc_percent", "mean_cpg_oe"])

    groups = []
    current = [island_windows.iloc[0]]
    previous_end = int(island_windows.iloc[0]["end"])

    for _, row in island_windows.iloc[1:].iterrows():
        if int(row["start"]) <= previous_end:
            current.append(row)
            previous_end = max(previous_end, int(row["end"]))
        else:
            groups.append(current)
            current = [row]
            previous_end = int(row["end"])
    groups.append(current)

    rows = []
    for group in groups:
        group_df = pd.DataFrame(group)
        rows.append(
            {
                "start": int(group_df["start"].min()),
                "end": int(group_df["end"].max()),
                "window_count": int(group_df.shape[0]),
                "mean_gc_percent": float(group_df["gc_percent"].mean()),
                "mean_cpg_oe": float(group_df["cpg_observed_expected"].mean()),
            }
        )
    return pd.DataFrame(rows)


def scan_motifs(sequence: str) -> pd.DataFrame:
    rows = []
    reverse = reverse_complement(sequence)
    for motif_name, pattern in MOTIF_PATTERNS.items():
        for strand, target in [("+", sequence), ("-", reverse)]:
            for match in re.finditer(f"(?=({pattern}))", target):
                matched = match.group(1)
                start = match.start(1)
                end = start + len(matched)
                if strand == "+":
                    genomic_start = start + 1
                    genomic_end = end
                else:
                    genomic_start = len(sequence) - end + 1
                    genomic_end = len(sequence) - start
                rows.append(
                    {
                        "motif": motif_name,
                        "strand": strand,
                        "start": genomic_start,
                        "end": genomic_end,
                        "matched_sequence": matched,
                    }
                )
    return pd.DataFrame(rows).sort_values(["start", "motif", "strand"]).reset_index(drop=True)


def translate(sequence: str) -> str:
    amino_acids = []
    for i in range(0, len(sequence) - 2, 3):
        amino_acids.append(CODON_TABLE.get(sequence[i : i + 3], "X"))
    return "".join(amino_acids)


def find_orfs(sequence: str, min_nt: int = 60) -> pd.DataFrame:
    rows = []
    targets = [("+", sequence), ("-", reverse_complement(sequence))]
    for strand, target in targets:
        for frame in range(3):
            i = frame
            while i <= len(target) - 3:
                codon = target[i : i + 3]
                if codon != START_CODON:
                    i += 3
                    continue

                for j in range(i + 3, len(target) - 2, 3):
                    stop = target[j : j + 3]
                    if stop not in STOP_CODONS:
                        continue

                    orf_sequence = target[i : j + 3]
                    if len(orf_sequence) >= min_nt:
                        if strand == "+":
                            start = i + 1
                            end = j + 3
                        else:
                            start = len(sequence) - (j + 3) + 1
                            end = len(sequence) - i

                        protein = translate(orf_sequence)
                        rows.append(
                            {
                                "strand": strand,
                                "frame": frame + 1,
                                "start": start,
                                "end": end,
                                "length_nt": len(orf_sequence),
                                "stop_codon": stop,
                                "gc_percent": gc_content(orf_sequence),
                                "protein_length_aa": len(protein.rstrip("*")),
                                "protein_sequence": protein,
                            }
                        )
                    break
                i += 3

    if not rows:
        return pd.DataFrame(
            columns=[
                "strand", "frame", "start", "end", "length_nt", "stop_codon",
                "gc_percent", "protein_length_aa", "protein_sequence",
            ]
        )
    return pd.DataFrame(rows).sort_values("length_nt", ascending=False).reset_index(drop=True)


def codon_usage(orf_sequence: str) -> pd.DataFrame:
    coding = orf_sequence[:-3] if orf_sequence[-3:] in STOP_CODONS else orf_sequence
    codons = [coding[i : i + 3] for i in range(0, len(coding) - 2, 3)]
    counts = Counter(codons)
    total = sum(counts.values())
    rows = [
        {
            "codon": codon,
            "amino_acid": CODON_TABLE[codon],
            "count": count,
            "frequency_percent": 100.0 * count / total,
        }
        for codon, count in sorted(counts.items())
    ]
    return pd.DataFrame(rows).sort_values(["amino_acid", "codon"]).reset_index(drop=True)


def kmer_enrichment(sequence: str, k: int = 4) -> pd.DataFrame:
    counts = Counter(sequence[i : i + k] for i in range(0, len(sequence) - k + 1))
    base_freq = {base: sequence.count(base) / len(sequence) for base in "ACGT"}
    trials = len(sequence) - k + 1
    rows = []

    for kmer, observed in counts.items():
        probability = math.prod(base_freq[base] for base in kmer)
        expected = trials * probability
        variance = max(trials * probability * (1.0 - probability), 1e-9)
        rows.append(
            {
                "kmer": kmer,
                "observed": observed,
                "expected_independent_model": expected,
                "log2_observed_expected": math.log2((observed + 0.5) / (expected + 0.5)),
                "z_score": (observed - expected) / math.sqrt(variance),
            }
        )
    return pd.DataFrame(rows).sort_values("z_score", ascending=False).reset_index(drop=True)


def find_tandem_repeats(sequence: str, min_copies: int = 4) -> pd.DataFrame:
    rows = []
    for unit_size in range(2, 7):
        i = 0
        while i <= len(sequence) - unit_size * min_copies:
            unit = sequence[i : i + unit_size]
            copies = 1
            while sequence[i + copies * unit_size : i + (copies + 1) * unit_size] == unit:
                copies += 1
            if copies >= min_copies:
                rows.append(
                    {
                        "start": i + 1,
                        "end": i + copies * unit_size,
                        "repeat_unit": unit,
                        "unit_size": unit_size,
                        "copy_count": copies,
                    }
                )
                i += copies * unit_size
            else:
                i += 1
    return pd.DataFrame(rows).sort_values(["start", "unit_size"]).reset_index(drop=True)


def plot_window_metrics(window_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(window_df["midpoint"], window_df["gc_percent"], color="#24796c", linewidth=2)
    axes[0].axhline(50, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_ylabel("GC %")

    axes[1].plot(window_df["midpoint"], window_df["gc_skew"], color="#9b3a4a", linewidth=2)
    axes[1].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_ylabel("GC skew")

    axes[2].plot(window_df["midpoint"], window_df["shannon_entropy"], color="#4f6fb6", linewidth=2)
    axes[2].set_ylabel("Entropy")
    axes[2].set_xlabel("Sequence position")

    fig.suptitle("Sliding-Window Sequence Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_map(
    sequence_length: int,
    orf_df: pd.DataFrame,
    motif_df: pd.DataFrame,
    cpg_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.hlines(0, 1, sequence_length, color="#222222", linewidth=2)

    for _, row in cpg_df.iterrows():
        ax.add_patch(
            plt.Rectangle(
                (row["start"], 0.16),
                row["end"] - row["start"] + 1,
                0.12,
                color="#7a9f35",
                alpha=0.7,
            )
        )

    for _, row in orf_df.iterrows():
        y = 0.46 if row["strand"] == "+" else -0.46
        color = "#cf6f2e" if row["strand"] == "+" else "#5d7896"
        ax.add_patch(
            plt.Rectangle(
                (row["start"], y - 0.08),
                row["end"] - row["start"] + 1,
                0.16,
                color=color,
                alpha=0.9,
            )
        )
        ax.text((row["start"] + row["end"]) / 2, y + 0.14, f"ORF {int(row['length_nt'])} nt",
                ha="center", va="bottom", fontsize=8)

    displayed_motifs = motif_df[motif_df["motif"].isin(["TATA_box", "Kozak_like_start", "EcoRI_site", "poly_A_signal"])]
    motif_y = {
        "TATA_box": -0.16,
        "Kozak_like_start": -0.26,
        "EcoRI_site": -0.36,
        "poly_A_signal": -0.16,
    }
    for _, row in displayed_motifs.iterrows():
        x = (row["start"] + row["end"]) / 2
        ax.vlines(x, motif_y[row["motif"]], 0.04, color="#222222", linewidth=1)
        ax.text(x, motif_y[row["motif"]] - 0.05, row["motif"], rotation=45, ha="right", va="top", fontsize=7)

    ax.set_ylim(-0.8, 0.8)
    ax.set_xlim(1, sequence_length)
    ax.set_yticks([-0.46, 0, 0.22, 0.46])
    ax.set_yticklabels(["Reverse ORFs", "DNA", "CpG islands", "Forward ORFs"])
    ax.set_xlabel("Sequence position")
    ax.set_title("Feature Map of Demo DNA Sequence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_summary(sequence: str, orf_df: pd.DataFrame, motif_df: pd.DataFrame, cpg_df: pd.DataFrame) -> None:
    print("Advanced sequence analysis complete.")
    print(f"Sequence length: {len(sequence)} bp")
    print(f"GC content: {gc_content(sequence):.2f}%")
    print(f"GC skew: {gc_skew(sequence):+.3f}")
    print(f"CpG observed/expected: {cpg_observed_expected(sequence):.3f}")
    print(f"ORFs found (>=60 nt): {orf_df.shape[0]}")
    print(f"Motif hits: {motif_df.shape[0]}")
    print(f"CpG islands: {cpg_df.shape[0]}")

    if not orf_df.empty:
        longest = orf_df.iloc[0]
        print("\nLongest ORF:")
        print(
            f"  strand={longest['strand']} frame={int(longest['frame'])} "
            f"start={int(longest['start'])} end={int(longest['end'])} "
            f"length={int(longest['length_nt'])} nt"
        )
        print(f"  protein={longest['protein_sequence']}")

    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    for filename in [
        "demo_sequence.fasta",
        "sequence_summary.csv",
        "sliding_window_metrics.csv",
        "cpg_islands.csv",
        "motif_hits.csv",
        "orfs.csv",
        "longest_orf_codon_usage.csv",
        "kmer_enrichment.csv",
        "tandem_repeats.csv",
        "window_metrics.png",
        "feature_map.png",
    ]:
        print(f"  - {filename}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sequence = make_demo_sequence()

    window_df = sliding_window_metrics(sequence)
    cpg_df = identify_cpg_islands(window_df)
    motif_df = scan_motifs(sequence)
    orf_df = find_orfs(sequence)
    kmer_df = kmer_enrichment(sequence)
    repeats_df = find_tandem_repeats(sequence)

    summary_df = pd.DataFrame(
        [
            {
                "sequence_id": "demo_sequence",
                "length_bp": len(sequence),
                "gc_percent": gc_content(sequence),
                "gc_skew": gc_skew(sequence),
                "cpg_observed_expected": cpg_observed_expected(sequence),
                "orf_count_min_60_nt": orf_df.shape[0],
                "motif_hit_count": motif_df.shape[0],
                "cpg_island_count": cpg_df.shape[0],
            }
        ]
    )

    with (OUTPUT_DIR / "demo_sequence.fasta").open("w", encoding="utf-8") as handle:
        handle.write(">demo_sequence advanced_sequence_analysis_demo\n")
        for i in range(0, len(sequence), 70):
            handle.write(sequence[i : i + 70] + "\n")

    summary_df.to_csv(OUTPUT_DIR / "sequence_summary.csv", index=False)
    window_df.to_csv(OUTPUT_DIR / "sliding_window_metrics.csv", index=False)
    cpg_df.to_csv(OUTPUT_DIR / "cpg_islands.csv", index=False)
    motif_df.to_csv(OUTPUT_DIR / "motif_hits.csv", index=False)
    orf_df.to_csv(OUTPUT_DIR / "orfs.csv", index=False)
    kmer_df.to_csv(OUTPUT_DIR / "kmer_enrichment.csv", index=False)
    repeats_df.to_csv(OUTPUT_DIR / "tandem_repeats.csv", index=False)

    if not orf_df.empty:
        longest = orf_df.iloc[0]
        if longest["strand"] == "+":
            orf_sequence = sequence[int(longest["start"]) - 1 : int(longest["end"])]
        else:
            orf_sequence = reverse_complement(sequence[int(longest["start"]) - 1 : int(longest["end"])])
        codon_usage(orf_sequence).to_csv(OUTPUT_DIR / "longest_orf_codon_usage.csv", index=False)
    else:
        pd.DataFrame(columns=["codon", "amino_acid", "count", "frequency_percent"]).to_csv(
            OUTPUT_DIR / "longest_orf_codon_usage.csv", index=False
        )

    plot_window_metrics(window_df, OUTPUT_DIR / "window_metrics.png")
    plot_feature_map(len(sequence), orf_df, motif_df, cpg_df, OUTPUT_DIR / "feature_map.png")

    print_summary(sequence, orf_df, motif_df, cpg_df)


if __name__ == "__main__":
    main()
