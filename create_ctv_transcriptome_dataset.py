from __future__ import annotations

import csv
import math
import re
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd


OUT_DIR = Path("ctv_transcriptome_dataset_output")
CACHE_DIR = Path("/tmp/ctv_transcriptome_sources")

SINGLE_ZIP_URL = (
    "https://static-content.springer.com/esm/art%3A10.1186%2Fs12864-016-2663-9/"
    "MediaObjects/12864_2016_2663_MOESM1_ESM.zip"
)
COINFECTION_XML_URL = (
    "https://www.frontiersin.org/journals/plant-science/articles/"
    "10.3389/fpls.2017.01419/xml/nlm"
)


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size == 0:
        urllib.request.urlretrieve(url, target)
    return target


def clean_float(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip().replace("\u2212", "-")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_single_infection_table(zip_path: Path) -> pd.DataFrame:
    extract_dir = CACHE_DIR / "single"
    table_path = extract_dir / "Additional file 1" / "Table S9.xlsx"
    if not table_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)

    raw = pd.read_excel(table_path, sheet_name="S9", header=3)
    raw = raw.rename(
        columns={
            "Unnamed: 0": "gene_symbol",
            "CTV-B2 ": "mild_ctv_b2_log2fc",
            "CTV-B6 ": "severe_ctv_b6_log2fc",
            "CaLas-B232  ": "calas_b232_log2fc",
            "Unnamed: 4": "citrus_sinensis_id",
            "Unnamed: 5": "arabidopsis_id",
            "Unnamed: 6": "e_value",
            "Unnamed: 7": "gene_description",
            "Unnamed: 8": "mapman_bin",
        }
    )
    raw = raw[raw["citrus_sinensis_id"].notna()].copy()
    raw["source_table"] = "Fu et al. 2016 Additional file 1 Table S9"
    raw["source_doi"] = "10.1186/s12864-016-2663-9"
    return raw[
        [
            "gene_symbol",
            "citrus_sinensis_id",
            "arabidopsis_id",
            "gene_description",
            "mapman_bin",
            "mild_ctv_b2_log2fc",
            "severe_ctv_b6_log2fc",
            "source_table",
            "source_doi",
        ]
    ]


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def parse_coinfection_tables(xml_path: Path) -> pd.DataFrame:
    root = ET.parse(xml_path).getroot()
    records = []

    for table_wrap in root.iter():
        if local_name(table_wrap.tag) != "table-wrap":
            continue

        label = ""
        caption = ""
        for child in table_wrap:
            if local_name(child.tag) == "label":
                label = " ".join("".join(child.itertext()).split())
            elif local_name(child.tag) == "caption":
                caption = " ".join("".join(child.itertext()).split())

        if label not in {"Table 1", "Table 2", "Table 3", "Table 4"}:
            continue

        for row in table_wrap.iter():
            if local_name(row.tag) != "tr":
                continue
            cells = [
                " ".join("".join(cell.itertext()).split())
                for cell in row
                if local_name(cell.tag) in {"td", "th"}
            ]
            if len(cells) < 5 or cells[0] == "Gene symbol":
                continue
            if not re.match(r"^(Orange1\.1g|orange1\.1g)", cells[1]):
                continue

            if label == "Table 4":
                log2fc = clean_float(cells[4])
                mild = clean_float(cells[5]) if len(cells) > 5 else None
                severe = clean_float(cells[6]) if len(cells) > 6 else None
            else:
                log2fc = clean_float(cells[4])
                mild = None
                severe = None

            records.append(
                {
                    "gene_symbol": cells[0] or "not_reported",
                    "citrus_sinensis_id": cells[1],
                    "arabidopsis_id": cells[2],
                    "gene_description": cells[3],
                    "coinfection_ctv_b2_b6_log2fc": log2fc,
                    "single_mild_ctv_b2_log2fc_reported_in_table4": mild,
                    "single_severe_ctv_b6_log2fc_reported_in_table4": severe,
                    "source_table": f"Fu et al. 2017 {label}: {caption}",
                    "source_doi": "10.3389/fpls.2017.01419",
                }
            )

    return pd.DataFrame.from_records(records)


def direction(value):
    value = clean_float(value)
    if value is None:
        return "not_differential_or_not_reported"
    if value >= 1:
        return "up"
    if value <= -1:
        return "down"
    return "below_threshold"


def make_long_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    condition_columns = [
        ("healthy_mock", "healthy_mock_log2fc", "healthy/mock-inoculated control"),
        ("mild_ctv_b2", "mild_ctv_b2_log2fc", "mild Citrus tristeza virus strain B2"),
        ("severe_ctv_b6", "severe_ctv_b6_log2fc", "severe Citrus tristeza virus strain B6"),
        (
            "coinfection_ctv_b2_b6",
            "coinfection_ctv_b2_b6_log2fc",
            "simultaneous mild B2 and severe B6 infection",
        ),
    ]
    id_cols = [
        "gene_symbol",
        "citrus_sinensis_id",
        "arabidopsis_id",
        "gene_description",
        "mapman_bin",
    ]
    for _, row in wide.iterrows():
        for condition, column, label in condition_columns:
            log2fc = row.get(column)
            if condition != "healthy_mock" and clean_float(log2fc) is None:
                continue
            item = {col: row.get(col) for col in id_cols}
            item.update(
                {
                    "condition": condition,
                    "condition_label": label,
                    "log2_fold_change_vs_healthy": 0.0
                    if condition == "healthy_mock"
                    else clean_float(log2fc),
                    "direction": "baseline" if condition == "healthy_mock" else direction(log2fc),
                }
            )
            rows.append(item)
    return pd.DataFrame(rows)


def write_reference_files() -> None:
    sample_metadata = [
        {
            "condition": "healthy_mock",
            "host": "Citrus sinensis cv. Valencia sweet orange seedlings",
            "infection": "mock inoculated",
            "strain_status": "healthy control",
            "rna_seq_design": "used as baseline for log2 fold-change comparisons",
            "symptom_summary": "healthy control plants",
        },
        {
            "condition": "mild_ctv_b2",
            "host": "Citrus sinensis cv. Valencia sweet orange seedlings",
            "infection": "Citrus tristeza virus strain B2",
            "strain_status": "mild CTV strain; T30 genotype",
            "rna_seq_design": "single infection compared with healthy control",
            "symptom_summary": "reported as not causing obvious symptoms on indicators",
        },
        {
            "condition": "severe_ctv_b6",
            "host": "Citrus sinensis cv. Valencia sweet orange seedlings",
            "infection": "Citrus tristeza virus strain B6",
            "strain_status": "severe CTV strain; mixed genotype related to SY568",
            "rna_seq_design": "single infection compared with healthy control",
            "symptom_summary": "severe strain associated with stem pitting, cupping, yellowing, leaf stiffening, and vein corking",
        },
        {
            "condition": "coinfection_ctv_b2_b6",
            "host": "Citrus sinensis cv. Valencia sweet orange seedlings",
            "infection": "simultaneous CTV-B2 and CTV-B6 graft inoculation",
            "strain_status": "mild + severe CTV co-infection",
            "rna_seq_design": "three infected biological RNA samples and healthy controls; DESeq2 vs healthy",
            "symptom_summary": "symptoms became indistinguishable from CTV-B6 alone; B6 p23 reads about 300-fold more frequent than B2",
        },
    ]
    pd.DataFrame(sample_metadata).to_csv(OUT_DIR / "ctv_sample_metadata.csv", index=False)

    condition_summary = [
        {
            "condition": "mild_ctv_b2",
            "upregulated_transcripts": 242,
            "downregulated_transcripts": 42,
            "read_depth_or_mapping": "38-44 million clean reads; approximately 73% mapped for CTV-B2",
            "threshold": "Gfold 0.01 in Fu et al. 2016",
            "source": "Fu et al. 2016 BMC Genomics",
            "doi": "10.1186/s12864-016-2663-9",
        },
        {
            "condition": "severe_ctv_b6",
            "upregulated_transcripts": 328,
            "downregulated_transcripts": 76,
            "read_depth_or_mapping": "38-44 million clean reads; approximately 76% mapped for CTV-B6",
            "threshold": "Gfold 0.01 in Fu et al. 2016",
            "source": "Fu et al. 2016 BMC Genomics",
            "doi": "10.1186/s12864-016-2663-9",
        },
        {
            "condition": "coinfection_ctv_b2_b6",
            "upregulated_transcripts": 411,
            "downregulated_transcripts": 356,
            "read_depth_or_mapping": "38-47 million raw reads per tree; approximately 61% mapped on average",
            "threshold": "Padj <= 0.1 and abs(log2FC) >= 1 in Fu et al. 2017",
            "source": "Fu et al. 2017 Frontiers in Plant Science",
            "doi": "10.3389/fpls.2017.01419",
        },
        {
            "condition": "healthy_mock",
            "upregulated_transcripts": 0,
            "downregulated_transcripts": 0,
            "read_depth_or_mapping": "control baseline; no log2FC by definition",
            "threshold": "not applicable",
            "source": "Fu et al. 2016 and Fu et al. 2017 controls",
            "doi": "10.1186/s12864-016-2663-9; 10.3389/fpls.2017.01419",
        },
    ]
    pd.DataFrame(condition_summary).to_csv(OUT_DIR / "ctv_condition_summary.csv", index=False)

    sources = [
        {
            "source_id": "fu_2016_single_ctv",
            "citation": "Fu S, Shao J, Zhou C, Hartung JS. Transcriptome analysis of sweet orange trees infected with 'Candidatus Liberibacter asiaticus' and two strains of Citrus tristeza virus. BMC Genomics. 2016;17:349.",
            "doi": "10.1186/s12864-016-2663-9",
            "used_for": "mild CTV-B2 and severe CTV-B6 differential transcript rows; healthy baseline; read-depth and DET summary",
            "data_accession": "not stated on the article page checked here; article supplementary ZIP Additional file 1 was used",
        },
        {
            "source_id": "fu_2017_coinfection_ctv",
            "citation": "Fu S, Shao J, Zhou C, Hartung JS. Co-infection of Sweet Orange with Severe and Mild Strains of Citrus tristeza virus Is Overwhelmingly Dominated by the Severe Strain on Both the Transcriptional and Biological Levels. Front Plant Sci. 2017;8:1419.",
            "doi": "10.3389/fpls.2017.01419",
            "used_for": "CTV-B2/CTV-B6 co-infection design, DET summary, and article Tables 1-4 transcript rows",
            "data_accession": "article reports Illumina HiSeq 2500 RNA-seq; supplementary Table S2 referenced by the paper",
        },
    ]
    pd.DataFrame(sources).to_csv(OUT_DIR / "ctv_source_notes.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    single_zip = download(SINGLE_ZIP_URL, CACHE_DIR / "single_infection_supplement.zip")
    coinfection_xml = download(COINFECTION_XML_URL, CACHE_DIR / "coinfection_article.xml")

    single = parse_single_infection_table(single_zip)
    coinfection = parse_coinfection_tables(coinfection_xml)

    wide = single.merge(
        coinfection[
            [
                "citrus_sinensis_id",
                "coinfection_ctv_b2_b6_log2fc",
                "source_table",
                "source_doi",
            ]
        ].rename(
            columns={
                "source_table": "coinfection_source_table",
                "source_doi": "coinfection_source_doi",
            }
        ),
        on="citrus_sinensis_id",
        how="outer",
    )

    for column in ["mild_ctv_b2_log2fc", "severe_ctv_b6_log2fc", "coinfection_ctv_b2_b6_log2fc"]:
        wide[column] = wide[column].map(clean_float)
    wide["healthy_mock_log2fc"] = 0.0

    first_cols = [
        "gene_symbol",
        "citrus_sinensis_id",
        "arabidopsis_id",
        "gene_description",
        "mapman_bin",
        "healthy_mock_log2fc",
        "mild_ctv_b2_log2fc",
        "severe_ctv_b6_log2fc",
        "coinfection_ctv_b2_b6_log2fc",
    ]
    for col in first_cols:
        if col not in wide:
            wide[col] = None
    wide = wide[first_cols + [col for col in wide.columns if col not in first_cols]]
    wide.to_csv(OUT_DIR / "ctv_differential_transcripts_wide.csv", index=False)

    long = make_long_table(wide)
    long.to_csv(OUT_DIR / "ctv_differential_transcripts_long.csv", index=False)

    coinfection.to_csv(OUT_DIR / "ctv_coinfection_article_tables.csv", index=False)
    write_reference_files()

    print(f"Wrote {OUT_DIR / 'ctv_differential_transcripts_wide.csv'} ({len(wide)} rows)")
    print(f"Wrote {OUT_DIR / 'ctv_differential_transcripts_long.csv'} ({len(long)} rows)")
    print(f"Wrote {OUT_DIR / 'ctv_coinfection_article_tables.csv'} ({len(coinfection)} rows)")
    print(f"Wrote metadata, condition summary, and source notes in {OUT_DIR}")


if __name__ == "__main__":
    main()
