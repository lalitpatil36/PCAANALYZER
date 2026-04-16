"""Research-grade DNA/RNA sequence analysis and visualization.

The script accepts a FASTA file, analyzes the sequence, and writes
publication-style PNG/PDF figures plus CSV result tables.

Example:
    python3 ctv_research_sequence_analysis.py U56902_ctv.fasta \
        --output ctv_research_output \
        --title "Citrus tristeza virus U56902.1 coding region"
"""

from __future__ import annotations

import argparse
import math
import os
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


STOP_CODONS = {"TAA", "TAG", "TGA"}
START_CODONS = {"ATG"}
VALID_BASES = set("ACGTN")

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

MOTIFS = {
    "poly_A_signal_AATAAA": "AATAAA",
    "EcoRI_GAATTC": "GAATTC",
    "HindIII_AAGCTT": "AAGCTT",
    "BamHI_GGATCC": "GGATCC",
    "NdeI_CATATG": "CATATG",
    "TATA_like": "TATA[AT]A[AT]",
    "CpG": "CG",
}

PANEL_COLORS = {
    "gc": "#24796c",
    "skew": "#b24a3b",
    "entropy": "#4869a8",
    "orf_plus": "#d08b2f",
    "orf_minus": "#5c7f9c",
    "motif": "#252525",
    "cpg": "#6f9e3a",
    "neutral": "#555555",
}


@dataclass(frozen=True)
class FastaRecord:
    identifier: str
    description: str
    sequence: str


def read_fasta(path: Path) -> FastaRecord:
    header = ""
    sequence_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    break
                header = line[1:].strip()
            else:
                sequence_parts.append(line)

    if not sequence_parts:
        raise ValueError(f"No sequence found in FASTA file: {path}")

    sequence = re.sub(r"[^A-Za-z]", "", "".join(sequence_parts)).upper().replace("U", "T")
    invalid = sorted(set(sequence) - VALID_BASES)
    if invalid:
        raise ValueError(f"Invalid base(s) in sequence: {', '.join(invalid)}")

    identifier = header.split()[0] if header else path.stem
    return FastaRecord(identifier=identifier, description=header, sequence=sequence)


def write_fasta(record: FastaRecord, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f">{record.description or record.identifier}\n")
        for i in range(0, len(record.sequence), 70):
            handle.write(record.sequence[i : i + 70] + "\n")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def base_counts(sequence: str) -> Counter[str]:
    return Counter(base for base in sequence if base in "ACGT")


def gc_content(sequence: str) -> float:
    counts = base_counts(sequence)
    total = sum(counts.values())
    return 0.0 if total == 0 else 100.0 * (counts["G"] + counts["C"]) / total


def gc_skew(sequence: str) -> float:
    counts = base_counts(sequence)
    total = counts["G"] + counts["C"]
    return 0.0 if total == 0 else (counts["G"] - counts["C"]) / total


def at_skew(sequence: str) -> float:
    counts = base_counts(sequence)
    total = counts["A"] + counts["T"]
    return 0.0 if total == 0 else (counts["A"] - counts["T"]) / total


def shannon_entropy(sequence: str) -> float:
    counts = base_counts(sequence)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count)


def cpg_observed_expected(sequence: str) -> float:
    counts = base_counts(sequence)
    c_count = counts["C"]
    g_count = counts["G"]
    if c_count == 0 or g_count == 0:
        return 0.0
    return sequence.count("CG") * len(sequence) / (c_count * g_count)


def sliding_windows(sequence: str, window: int, step: int) -> pd.DataFrame:
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
                "at_skew": at_skew(chunk),
                "cpg_observed_expected": cpg_observed_expected(chunk),
                "shannon_entropy": shannon_entropy(chunk),
            }
        )
    return pd.DataFrame(rows)


def cumulative_gc_skew(sequence: str) -> pd.DataFrame:
    cumulative = 0
    rows = []
    for index, base in enumerate(sequence, start=1):
        if base == "G":
            cumulative += 1
        elif base == "C":
            cumulative -= 1
        rows.append({"position": index, "cumulative_gc_skew": cumulative})
    return pd.DataFrame(rows)


def scan_motifs(sequence: str) -> pd.DataFrame:
    rows = []
    rc_sequence = reverse_complement(sequence)
    for motif_name, pattern in MOTIFS.items():
        for strand, target in [("+", sequence), ("-", rc_sequence)]:
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
    if not rows:
        return pd.DataFrame(columns=["motif", "strand", "start", "end", "matched_sequence"])
    return pd.DataFrame(rows).sort_values(["start", "motif", "strand"]).reset_index(drop=True)


def identify_cpg_rich_regions(window_df: pd.DataFrame, min_gc: float = 50.0, min_oe: float = 0.6) -> pd.DataFrame:
    selected = window_df[
        (window_df["gc_percent"] >= min_gc) & (window_df["cpg_observed_expected"] >= min_oe)
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=["start", "end", "window_count", "mean_gc_percent", "mean_cpg_oe"])

    groups = []
    current = [selected.iloc[0]]
    previous_end = int(selected.iloc[0]["end"])
    for _, row in selected.iloc[1:].iterrows():
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


def translate(sequence: str) -> str:
    amino_acids = []
    for i in range(0, len(sequence) - 2, 3):
        amino_acids.append(CODON_TABLE.get(sequence[i : i + 3], "X"))
    return "".join(amino_acids)


def find_orfs(sequence: str, min_nt: int) -> pd.DataFrame:
    rows = []
    for strand, target in [("+", sequence), ("-", reverse_complement(sequence))]:
        for frame in range(3):
            active_starts: list[int] = []
            for i in range(frame, len(target) - 2, 3):
                codon = target[i : i + 3]
                if codon in START_CODONS:
                    active_starts.append(i)
                if codon not in STOP_CODONS:
                    continue
                for start in active_starts:
                    orf_sequence = target[start : i + 3]
                    if len(orf_sequence) >= min_nt:
                        if strand == "+":
                            genomic_start = start + 1
                            genomic_end = i + 3
                        else:
                            genomic_start = len(sequence) - (i + 3) + 1
                            genomic_end = len(sequence) - start
                        protein = translate(orf_sequence)
                        rows.append(
                            {
                                "strand": strand,
                                "frame": frame + 1,
                                "start": genomic_start,
                                "end": genomic_end,
                                "length_nt": len(orf_sequence),
                                "protein_length_aa": len(protein.rstrip("*")),
                                "stop_codon": codon,
                                "gc_percent": gc_content(orf_sequence),
                                "protein_preview": protein[:80],
                            }
                        )
                active_starts = []
    if not rows:
        return pd.DataFrame(
            columns=[
                "strand", "frame", "start", "end", "length_nt", "protein_length_aa",
                "stop_codon", "gc_percent", "protein_preview",
            ]
        )
    return pd.DataFrame(rows).sort_values("length_nt", ascending=False).reset_index(drop=True)


def non_overlapping_orfs(orf_df: pd.DataFrame, max_orfs: int = 20) -> pd.DataFrame:
    selected = []
    for _, row in orf_df.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        overlaps = False
        for kept in selected:
            overlap = max(0, min(end, int(kept["end"])) - max(start, int(kept["start"])) + 1)
            shorter = min(end - start + 1, int(kept["end"]) - int(kept["start"]) + 1)
            if shorter and overlap / shorter > 0.5:
                overlaps = True
                break
        if not overlaps:
            selected.append(row)
        if len(selected) >= max_orfs:
            break
    return pd.DataFrame(selected).reset_index(drop=True) if selected else orf_df.head(0)


def codon_usage(sequence: str) -> pd.DataFrame:
    sequence = sequence[: len(sequence) - len(sequence) % 3]
    codons = [sequence[i : i + 3] for i in range(0, len(sequence) - 2, 3)]
    codons = [codon for codon in codons if codon in CODON_TABLE and codon not in STOP_CODONS]
    counts = Counter(codons)
    total = sum(counts.values())
    rows = []
    for codon, count in sorted(counts.items()):
        rows.append(
            {
                "codon": codon,
                "amino_acid": CODON_TABLE[codon],
                "count": count,
                "frequency_percent": 0.0 if total == 0 else 100.0 * count / total,
            }
        )
    return pd.DataFrame(rows).sort_values(["amino_acid", "codon"]).reset_index(drop=True)


def kmer_enrichment(sequence: str, k: int) -> pd.DataFrame:
    clean = "".join(base for base in sequence if base in "ACGT")
    counts = Counter(clean[i : i + k] for i in range(0, len(clean) - k + 1))
    base_freq = {base: clean.count(base) / len(clean) for base in "ACGT"}
    trials = len(clean) - k + 1
    rows = []
    for kmer, observed in counts.items():
        probability = math.prod(base_freq[base] for base in kmer)
        expected = trials * probability
        variance = max(trials * probability * (1.0 - probability), 1e-12)
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


def tandem_repeats(sequence: str, min_copies: int = 5, max_unit: int = 8) -> pd.DataFrame:
    rows = []
    for unit_size in range(2, max_unit + 1):
        i = 0
        while i <= len(sequence) - unit_size * min_copies:
            unit = sequence[i : i + unit_size]
            if "N" in unit:
                i += 1
                continue
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
    if not rows:
        return pd.DataFrame(columns=["start", "end", "repeat_unit", "unit_size", "copy_count"])
    return pd.DataFrame(rows).sort_values(["start", "unit_size"]).reset_index(drop=True)


def parse_genbank_cds_features(path: Path) -> pd.DataFrame:
    """Parse simple GenBank CDS features without requiring Biopython."""
    rows = []
    current: dict[str, object] | None = None
    current_qualifier = ""

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("ORIGIN"):
                break
            if line.startswith("     CDS             "):
                if current:
                    rows.append(current)
                location = line[21:].strip()
                current = parse_location(location)
                current["raw_location"] = location
                current_qualifier = ""
                continue
            if line.startswith("     ") and line[5:21].strip():
                if current:
                    rows.append(current)
                current = None
                current_qualifier = ""
                continue
            if current is None or not line.startswith("                     "):
                continue

            text = line[21:].strip()
            if text.startswith("/"):
                key, value = parse_qualifier(text)
                current[key] = value
                current_qualifier = key
            elif current_qualifier:
                current[current_qualifier] = f"{current[current_qualifier]} {text.strip().strip('\"')}"

    if current:
        rows.append(current)

    if not rows:
        return pd.DataFrame(
            columns=[
                "start", "end", "strand", "partial_start", "partial_end",
                "raw_location", "product", "gene", "function", "note", "protein_id",
            ]
        )

    features = pd.DataFrame(rows)
    for column in ["product", "gene", "function", "note", "protein_id"]:
        if column not in features.columns:
            features[column] = ""
    features["label"] = features.apply(feature_label, axis=1)
    features["length_nt"] = features["end"] - features["start"] + 1
    return features.sort_values(["start", "end"]).reset_index(drop=True)


def parse_location(location: str) -> dict[str, object]:
    strand = "-"
    if not location.startswith("complement"):
        strand = "+"
    numbers = [int(value) for value in re.findall(r"\d+", location)]
    if not numbers:
        raise ValueError(f"Could not parse GenBank location: {location}")
    return {
        "start": min(numbers),
        "end": max(numbers),
        "strand": strand,
        "partial_start": location.strip().startswith("<") or "(<" in location,
        "partial_end": ">" in location,
    }


def parse_qualifier(text: str) -> tuple[str, str]:
    key_value = text[1:].split("=", 1)
    key = key_value[0]
    if len(key_value) == 1:
        return key, "true"
    return key, key_value[1].strip().strip("\"")


def feature_label(row: pd.Series) -> str:
    for column in ["product", "gene", "note"]:
        raw_value = row.get(column, "")
        if pd.isna(raw_value):
            continue
        value = str(raw_value).strip()
        if value:
            if column == "note" and ";" in value:
                value = value.split(";", 1)[0]
            return value
    return f"CDS {int(row['start'])}-{int(row['end'])}"


def infer_window_size(length: int) -> tuple[int, int]:
    window = max(300, min(1000, round(length / 40)))
    step = max(50, round(window / 5))
    return window, step


def summarize(record: FastaRecord, orfs: pd.DataFrame, motifs: pd.DataFrame, cpg_regions: pd.DataFrame) -> pd.DataFrame:
    counts = base_counts(record.sequence)
    total = sum(counts.values())
    return pd.DataFrame(
        [
            {
                "sequence_id": record.identifier,
                "description": record.description,
                "length_bp": len(record.sequence),
                "A_count": counts["A"],
                "C_count": counts["C"],
                "G_count": counts["G"],
                "T_count": counts["T"],
                "N_count": record.sequence.count("N"),
                "A_percent": 100.0 * counts["A"] / total,
                "C_percent": 100.0 * counts["C"] / total,
                "G_percent": 100.0 * counts["G"] / total,
                "T_percent": 100.0 * counts["T"] / total,
                "gc_percent": gc_content(record.sequence),
                "gc_skew": gc_skew(record.sequence),
                "at_skew": at_skew(record.sequence),
                "cpg_observed_expected": cpg_observed_expected(record.sequence),
                "shannon_entropy": shannon_entropy(record.sequence),
                "orf_count": int(orfs.shape[0]),
                "motif_hit_count": int(motifs.shape[0]),
                "cpg_rich_region_count": int(cpg_regions.shape[0]),
            }
        ]
    )


def extract_orf_sequence(sequence: str, row: pd.Series) -> str:
    segment = sequence[int(row["start"]) - 1 : int(row["end"])]
    return segment if row["strand"] == "+" else reverse_complement(segment)


def save_bar_labels(ax, values: list[float], fmt: str = "{:.1f}") -> None:
    max_value = max(values) if values else 0
    for index, value in enumerate(values):
        ax.text(index, value + max_value * 0.025, fmt.format(value), ha="center", va="bottom", fontsize=8)


def plot_publication_summary(
    record: FastaRecord,
    summary_df: pd.DataFrame,
    window_df: pd.DataFrame,
    skew_df: pd.DataFrame,
    orfs: pd.DataFrame,
    cpg_regions: pd.DataFrame,
    kmer_df: pd.DataFrame,
    output_prefix: Path,
    title: str,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.1, 1.1])

    ax_base = fig.add_subplot(grid[0, 0])
    base_percents = [
        float(summary_df.loc[0, "A_percent"]),
        float(summary_df.loc[0, "C_percent"]),
        float(summary_df.loc[0, "G_percent"]),
        float(summary_df.loc[0, "T_percent"]),
    ]
    base_colors = ["#5aa469", "#3978a8", "#d08b2f", "#bd4f45"]
    ax_base.bar(["A", "C", "G", "T"], base_percents, color=base_colors, edgecolor="#222222", linewidth=0.6)
    save_bar_labels(ax_base, base_percents)
    ax_base.set_ylabel("Composition (%)")
    ax_base.set_title("A. Nucleotide composition")
    ax_base.set_ylim(0, max(base_percents) * 1.25)

    ax_window = fig.add_subplot(grid[0, 1])
    ax_window.plot(window_df["midpoint"], window_df["gc_percent"], color=PANEL_COLORS["gc"], linewidth=1.6, label="GC %")
    ax_window.axhline(float(summary_df.loc[0, "gc_percent"]), color="#333333", linestyle="--", linewidth=0.8, label="Genome mean")
    ax_window.set_ylabel("GC (%)")
    ax_window.set_xlabel("Genome position (nt)")
    ax_window.set_title("B. Sliding-window GC content")
    ax_window.legend(frameon=False, loc="best")

    ax_skew = fig.add_subplot(grid[1, 0])
    ax_skew.plot(window_df["midpoint"], window_df["gc_skew"], color=PANEL_COLORS["skew"], linewidth=1.4, label="Window GC skew")
    ax_skew.plot(skew_df["position"], skew_df["cumulative_gc_skew"], color="#333333", linewidth=0.9, alpha=0.55, label="Cumulative skew")
    ax_skew.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax_skew.set_xlabel("Genome position (nt)")
    ax_skew.set_ylabel("GC skew")
    ax_skew.set_title("C. GC skew profile")
    ax_skew.legend(frameon=False, loc="best")

    ax_entropy = fig.add_subplot(grid[1, 1])
    ax_entropy.plot(window_df["midpoint"], window_df["shannon_entropy"], color=PANEL_COLORS["entropy"], linewidth=1.6)
    ax_entropy.set_xlabel("Genome position (nt)")
    ax_entropy.set_ylabel("Shannon entropy")
    ax_entropy.set_title("D. Local sequence complexity")

    ax_orf = fig.add_subplot(grid[2, 0])
    plot_orf_axis(ax_orf, len(record.sequence), orfs, cpg_regions)
    ax_orf.set_title("E. Predicted ORFs and CpG-rich regions")

    ax_kmer = fig.add_subplot(grid[2, 1])
    top_kmers = kmer_df.head(12).iloc[::-1]
    ax_kmer.barh(top_kmers["kmer"], top_kmers["z_score"], color="#5c7f9c", edgecolor="#222222", linewidth=0.4)
    ax_kmer.axvline(0, color="#777777", linewidth=0.8)
    ax_kmer.set_xlabel("Enrichment z-score")
    ax_kmer.set_title("F. Most enriched 4-mers")

    wrapped_title = "\n".join(textwrap.wrap(title, width=110))
    fig.suptitle(wrapped_title)
    for suffix in ("png", "pdf"):
        fig.savefig(output_prefix.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_orf_axis(ax, length: int, orfs: pd.DataFrame, cpg_regions: pd.DataFrame) -> None:
    ax.hlines(0, 1, length, color="#222222", linewidth=1.2)
    for _, row in cpg_regions.iterrows():
        ax.add_patch(
            Rectangle(
                (float(row["start"]), 0.16),
                float(row["end"] - row["start"] + 1),
                0.10,
                facecolor=PANEL_COLORS["cpg"],
                edgecolor="none",
                alpha=0.8,
            )
        )

    display_orfs = non_overlapping_orfs(orfs, max_orfs=14)
    plus_level = 0
    minus_level = 0
    for index, row in display_orfs.iterrows():
        is_plus = row["strand"] == "+"
        if is_plus:
            y = 0.42 + 0.13 * (plus_level % 3)
            plus_level += 1
            color = PANEL_COLORS["orf_plus"]
        else:
            y = -0.42 - 0.13 * (minus_level % 3)
            minus_level += 1
            color = PANEL_COLORS["orf_minus"]
        ax.add_patch(
            Rectangle(
                (float(row["start"]), y - 0.045),
                float(row["end"] - row["start"] + 1),
                0.09,
                facecolor=color,
                edgecolor="#222222",
                linewidth=0.35,
                alpha=0.92,
            )
        )
        if index < 8:
            label = f"{int(row['length_nt'])} nt"
            ax.text((float(row["start"]) + float(row["end"])) / 2, y + 0.07, label, ha="center", va="bottom", fontsize=7)

    ax.set_xlim(1, length)
    ax.set_ylim(-0.9, 0.9)
    ax.set_yticks([-0.52, 0, 0.22, 0.52])
    ax.set_yticklabels(["- ORFs", "Genome", "CpG-rich", "+ ORFs"])
    ax.set_xlabel("Genome position (nt)")


def plot_orf_map(
    record: FastaRecord,
    orfs: pd.DataFrame,
    motifs: pd.DataFrame,
    cpg_regions: pd.DataFrame,
    output_prefix: Path,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    plot_orf_axis(ax, len(record.sequence), orfs, cpg_regions)

    selected_motifs = motifs[~motifs["motif"].eq("CpG")].copy()
    selected_motifs = selected_motifs.head(80)
    motif_levels = {
        "poly_A_signal_AATAAA": -0.14,
        "EcoRI_GAATTC": -0.23,
        "HindIII_AAGCTT": -0.32,
        "BamHI_GGATCC": -0.41,
        "NdeI_CATATG": -0.50,
        "TATA_like": -0.59,
    }
    for _, row in selected_motifs.iterrows():
        x = (float(row["start"]) + float(row["end"])) / 2
        y = motif_levels.get(row["motif"], -0.14)
        ax.vlines(x, y, 0.04, color=PANEL_COLORS["motif"], linewidth=0.6, alpha=0.8)
    ax.text(0.01, 0.02, "Vertical ticks mark selected motif/restriction-site hits", transform=ax.transAxes, fontsize=8)
    ax.set_title("\n".join(textwrap.wrap(title + " | ORF and motif map", width=130)))

    for suffix in ("png", "pdf"):
        fig.savefig(output_prefix.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_codon_usage(codon_df: pd.DataFrame, output_prefix: Path, title: str, dpi: int) -> None:
    if codon_df.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    codon_df = codon_df.sort_values(["amino_acid", "codon"])
    colors = ["#3978a8" if aa in "ACDEFGHIKLM" else "#d08b2f" for aa in codon_df["amino_acid"]]
    labels = [f"{row.codon}\n{row.amino_acid}" for row in codon_df.itertuples()]
    ax.bar(range(codon_df.shape[0]), codon_df["frequency_percent"], color=colors, edgecolor="#222222", linewidth=0.25)
    ax.set_xticks(range(codon_df.shape[0]))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Codon frequency in longest ORF (%)")
    ax.set_xlabel("Codon / amino acid")
    ax.set_title("\n".join(textwrap.wrap(title + " | Codon usage of the longest predicted ORF", width=120)))
    for suffix in ("png", "pdf"):
        fig.savefig(output_prefix.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_curated_cds_map(
    record: FastaRecord,
    features: pd.DataFrame,
    window_df: pd.DataFrame,
    output_prefix: Path,
    title: str,
    dpi: int,
) -> None:
    if features.empty:
        return

    fig = plt.figure(figsize=(13, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.2])
    ax_gc = fig.add_subplot(grid[0, 0])
    ax_map = fig.add_subplot(grid[1, 0], sharex=ax_gc)

    ax_gc.plot(window_df["midpoint"], window_df["gc_percent"], color=PANEL_COLORS["gc"], linewidth=1.5)
    ax_gc.axhline(gc_content(record.sequence), color="#333333", linestyle="--", linewidth=0.8)
    ax_gc.set_ylabel("GC (%)")
    ax_gc.set_title("Local GC content")

    lanes: list[int] = []
    palette = ["#d08b2f", "#5c7f9c", "#6f9e3a", "#bd4f45", "#3978a8", "#8a7f35"]
    for index, row in features.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        lane = assign_feature_lane(start, end, lanes)
        y = lane * 0.34
        color = palette[index % len(palette)]
        ax_map.add_patch(
            Rectangle(
                (start, y - 0.09),
                end - start + 1,
                0.18,
                facecolor=color,
                edgecolor="#222222",
                linewidth=0.45,
                alpha=0.92,
            )
        )
        label = str(row["label"])
        if len(label) > 32:
            label = label[:29] + "..."
        ax_map.text((start + end) / 2, y + 0.14, label, ha="center", va="bottom", fontsize=8)
        ax_map.text((start + end) / 2, y - 0.14, f"{start}-{end}", ha="center", va="top", fontsize=6.5)

    ax_map.hlines(-0.28, 1, len(record.sequence), color="#222222", linewidth=1.2)
    ax_map.set_xlim(1, len(record.sequence))
    ax_map.set_ylim(-0.48, max(0.6, (max(lanes) if lanes else 0) * 0.34 + 0.55))
    ax_map.set_yticks([])
    ax_map.set_xlabel("Genome position (nt)")
    ax_map.set_title("Curated GenBank CDS architecture")

    wrapped_title = "\n".join(textwrap.wrap(title + " | curated coding sequence map", width=130))
    fig.suptitle(wrapped_title)
    for suffix in ("png", "pdf"):
        fig.savefig(output_prefix.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def assign_feature_lane(start: int, end: int, lane_ends: list[int]) -> int:
    for lane, previous_end in enumerate(lane_ends):
        if start > previous_end + 160:
            lane_ends[lane] = end
            return lane
    lane_ends.append(end)
    return len(lane_ends) - 1


def run_analysis(args: argparse.Namespace) -> None:
    fasta_path = Path(args.fasta)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    record = read_fasta(fasta_path)
    title = args.title or record.description or record.identifier
    window, step = (args.window, args.step) if args.window and args.step else infer_window_size(len(record.sequence))

    windows = sliding_windows(record.sequence, window=window, step=step)
    skew = cumulative_gc_skew(record.sequence)
    motifs = scan_motifs(record.sequence)
    cpg_regions = identify_cpg_rich_regions(windows)
    orfs = find_orfs(record.sequence, min_nt=args.min_orf_nt)
    kmers = kmer_enrichment(record.sequence, k=4)
    repeats = tandem_repeats(record.sequence)
    summary = summarize(record, orfs, motifs, cpg_regions)
    curated_features = pd.DataFrame()
    if args.genbank:
        curated_features = parse_genbank_cds_features(Path(args.genbank))

    write_fasta(record, output_dir / "cleaned_input_sequence.fasta")
    summary.to_csv(output_dir / "sequence_summary.csv", index=False)
    windows.to_csv(output_dir / "sliding_window_metrics.csv", index=False)
    skew.to_csv(output_dir / "cumulative_gc_skew.csv", index=False)
    motifs.to_csv(output_dir / "motif_hits.csv", index=False)
    cpg_regions.to_csv(output_dir / "cpg_rich_regions.csv", index=False)
    orfs.to_csv(output_dir / "predicted_orfs.csv", index=False)
    kmers.to_csv(output_dir / "kmer_enrichment_4mer.csv", index=False)
    repeats.to_csv(output_dir / "tandem_repeats.csv", index=False)
    if not curated_features.empty:
        curated_features.to_csv(output_dir / "genbank_cds_features.csv", index=False)

    codon_df = pd.DataFrame(columns=["codon", "amino_acid", "count", "frequency_percent"])
    if not orfs.empty:
        longest_orf_sequence = extract_orf_sequence(record.sequence, orfs.iloc[0])
        codon_df = codon_usage(longest_orf_sequence)
    codon_df.to_csv(output_dir / "longest_orf_codon_usage.csv", index=False)

    plot_publication_summary(
        record,
        summary,
        windows,
        skew,
        orfs,
        cpg_regions,
        kmers,
        output_dir / "figure_1_publication_summary",
        title,
        args.dpi,
    )
    plot_orf_map(
        record,
        orfs,
        motifs,
        cpg_regions,
        output_dir / "figure_2_orf_motif_map",
        title,
        args.dpi,
    )
    plot_codon_usage(
        codon_df,
        output_dir / "figure_3_longest_orf_codon_usage",
        title,
        args.dpi,
    )
    if not curated_features.empty:
        plot_curated_cds_map(
            record,
            curated_features,
            windows,
            output_dir / "figure_4_curated_genbank_cds_map",
            title,
            args.dpi,
        )

    print("Research sequence analysis complete.")
    print(f"Input: {fasta_path.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Length: {len(record.sequence)} bp")
    print(f"GC content: {float(summary.loc[0, 'gc_percent']):.2f}%")
    print(f"GC skew: {float(summary.loc[0, 'gc_skew']):+.4f}")
    print(f"CpG observed/expected: {float(summary.loc[0, 'cpg_observed_expected']):.3f}")
    print(f"Sliding window: {window} bp, step {step} bp")
    print(f"Predicted ORFs >= {args.min_orf_nt} nt: {orfs.shape[0]}")
    print(f"CpG-rich regions: {cpg_regions.shape[0]}")
    print(f"Motif hits: {motifs.shape[0]}")
    if not orfs.empty:
        longest = orfs.iloc[0]
        print(
            "Longest ORF: "
            f"{longest['strand']} strand, frame {int(longest['frame'])}, "
            f"{int(longest['start'])}-{int(longest['end'])}, "
            f"{int(longest['length_nt'])} nt"
        )
    print("Figures:")
    print("  - figure_1_publication_summary.png/.pdf")
    print("  - figure_2_orf_motif_map.png/.pdf")
    print("  - figure_3_longest_orf_codon_usage.png/.pdf")
    if not curated_features.empty:
        print("  - figure_4_curated_genbank_cds_map.png/.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a FASTA sequence and create research-quality figures.")
    parser.add_argument("fasta", help="Input FASTA file containing one sequence.")
    parser.add_argument("--output", default="ctv_research_output", help="Output directory.")
    parser.add_argument("--title", default="", help="Figure title. Defaults to FASTA header.")
    parser.add_argument("--min-orf-nt", type=int, default=300, help="Minimum ORF length in nucleotides.")
    parser.add_argument("--window", type=int, default=0, help="Sliding window size. Auto-selected by default.")
    parser.add_argument("--step", type=int, default=0, help="Sliding window step. Auto-selected by default.")
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI for PNG export.")
    parser.add_argument("--genbank", default="", help="Optional GenBank file for curated CDS feature plotting.")
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
