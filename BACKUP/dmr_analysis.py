#!/usr/bin/env python3
"""
Module: DMR Analysis Pipeline for Alzheimer's Disease Epigenetics
Purpose: Differential Methylated Region (DMR) analysis of GEO dataset GSE244352.
         Implements filtering, visualization (Manhattan/Volcano plots), genomic
         annotation, and KEGG pathway enrichment for AD-related target genes:
         APP, PSEN1, PSEN2, APOE, MAPT, TREM2.

Reference:
    Mey, T. V. K. Y., Lee, M. B., Tomodok, A. C. A., Muliana, N. E., Priskila, D.,
    Sadrawi, M., & Parikesit, A. A. (2025). Detection of Alzheimer's Disease through
    AI-Driven and Methylation Difference Region Analysis of Significant Epigenetic
    Modifications in APP, PSEN1, PSEN2, APOE, MAPT, and TREM2 Genes. In
    ADVANCED THERAPEUTICS AND DISEASE BIOLOGY: MOLECULAR DIAGNOSTICS AND
    IMMUNITY- 2025 (pp. 57-86). Halic Publishing House.
    https://doi.org/10.5281/zenodo.18070507

Data Source:
    GSE244352 - GEO (Gene Expression Omnibus)
    Mitsumori et al., 2025 - Methylation capture sequencing of Japanese AD patients

Parameters:
    Q_VALUE_THRESHOLD      : 0.05  (FDR-corrected significance cutoff)
    METH_DIFF_THRESHOLD    : 15.0  (absolute methylation difference %, minimum)

Usage:
    python dmr_analysis.py --input <dmr_data.csv> --output <results_dir>
    python dmr_analysis.py --demo   (runs with built-in synthetic data)
"""

import argparse
import os
import sys
import warnings
import numpy as np
#p = p / p.sum()
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
Q_VALUE_THRESHOLD   = 0.05
METH_DIFF_THRESHOLD = 15.0

# Target AD-related genes with hg38 coordinates
# Source: UCSC Genome Browser, hg38 assembly
AD_GENE_COORDS = {
    "APP":   {"chr": "chr21", "start": 25880550,  "end": 26170620,  "gene_id": 351},
    "PSEN1": {"chr": "chr14", "start": 73113418,  "end": 73223691,  "gene_id": 5663},
    "PSEN2": {"chr": "chr1",  "start": 226870384, "end": 226942614, "gene_id": 5664},
    "APOE":  {"chr": "chr19", "start": 44905791,  "end": 44909393,  "gene_id": 348},
    "MAPT":  {"chr": "chr17", "start": 45889382,  "end": 46028334,  "gene_id": 4137},
    "TREM2": {"chr": "chr6",  "start": 41161514,  "end": 41167971,  "gene_id": 54209},
}

# Promoter window: 2000 bp upstream and 500 bp downstream of TSS
PROMOTER_UPSTREAM   = 2000
PROMOTER_DOWNSTREAM = 500

# KEGG pathway IDs for Alzheimer's disease
KEGG_AD_PATHWAYS = ["hsa05010", "hsa05022"]


# ---------------------------------------------------------------------------
# 1. Data Generation / Loading
# ---------------------------------------------------------------------------

def generate_sample_dmr_data(n_sites: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic DMR data resembling the GSE244352 output structure.
    Used for demonstration when real data is unavailable.

    Returns a DataFrame with columns:
        chr, start, end, pvalue, qvalue, meth_diff
    """
    rng = np.random.default_rng(seed)

    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    chr_weights  = np.array([0.06] * 22 + [0.04])
    chr_weights  = chr_weights / chr_weights.sum()
    chr_sizes    = {
        f"chr{i}": int(2.5e8 / i) for i in range(1, 23)
    }
    chr_sizes["chrX"] = 155_000_000

    chrs  = rng.choice(chromosomes, size=n_sites, p=chr_weights)
    starts = np.array([
        rng.integers(1, max(1, chr_sizes.get(c, 100_000_000) - 500))
        for c in chrs
    ], dtype=np.int64)
    ends = starts + rng.integers(1, 500, size=n_sites)

    # Simulate p-values and methylation differences
    pvalues   = rng.beta(0.3, 5, size=n_sites)
    meth_diff = rng.normal(0, 18, size=n_sites)

    # Force a handful of sites to match paper results (PSEN1 and MAPT loci)
    paper_sites = [
        {"chr": "chr14", "start": 73113602, "end": 73113602,
         "pvalue": 1e-20, "qvalue": 0.0, "meth_diff": -30.08647},
        {"chr": "chr14", "start": 73198335, "end": 73198335,
         "pvalue": 1e-20, "qvalue": 0.0, "meth_diff": 19.11400},
        {"chr": "chr17", "start": 45889839, "end": 45889839,
         "pvalue": 8.12e-10, "qvalue": 1.47e-7, "meth_diff": -24.09307},
    ]

    # Compute q-values via Benjamini-Hochberg approximation
    ranks  = np.argsort(pvalues)
    qvals  = np.minimum(1.0, pvalues * n_sites / (ranks + 1))
    qvals  = np.minimum.accumulate(qvals[::-1])[::-1]

    df = pd.DataFrame({
        "chr":       chrs,
        "start":     starts,
        "end":       ends,
        "pvalue":    pvalues,
        "qvalue":    qvals,
        "meth_diff": meth_diff,
    })

    # Append the paper's known significant sites
    paper_df = pd.DataFrame(paper_sites)
    df = pd.concat([df, paper_df], ignore_index=True)

    return df


def load_dmr_data(filepath: str) -> pd.DataFrame:
    """
    Load DMR data from a CSV or TSV file.

    Expected columns (case-insensitive):
        chr / chromosome, start, end, pvalue / p.value, qvalue / q.value,
        meth_diff / meth.diff / methylation_difference

    Parameters
    ----------
    filepath : str
        Path to the DMR data file.

    Returns
    -------
    pd.DataFrame
    """
    sep = "\t" if filepath.endswith(".tsv") or filepath.endswith(".txt") else ","
    df  = pd.read_csv(filepath, sep=sep)

    # Normalise column names
    rename_map = {}
    for col in df.columns:
        lc = col.lower().replace(".", "_").replace(" ", "_")
        if lc in ("chromosome", "chr"):
            rename_map[col] = "chr"
        elif lc == "start":
            rename_map[col] = "start"
        elif lc in ("end", "end_strand"):
            rename_map[col] = "end"
        elif lc in ("pvalue", "p_value", "p.value"):
            rename_map[col] = "pvalue"
        elif lc in ("qvalue", "q_value", "q.value"):
            rename_map[col] = "qvalue"
        elif lc in ("meth_diff", "meth.diff", "methylation_difference",
                    "diff", "methdiff"):
            rename_map[col] = "meth_diff"

    df = df.rename(columns=rename_map)

    required = {"chr", "start", "end", "pvalue", "qvalue", "meth_diff"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Ensure chr prefix
    df["chr"] = df["chr"].astype(str)
    if not df["chr"].iloc[0].startswith("chr"):
        df["chr"] = "chr" + df["chr"]

    df["start"]     = pd.to_numeric(df["start"],     errors="coerce")
    df["end"]       = pd.to_numeric(df["end"],       errors="coerce")
    df["pvalue"]    = pd.to_numeric(df["pvalue"],    errors="coerce")
    df["qvalue"]    = pd.to_numeric(df["qvalue"],    errors="coerce")
    df["meth_diff"] = pd.to_numeric(df["meth_diff"], errors="coerce")

    return df.dropna(subset=["chr", "start", "pvalue", "qvalue", "meth_diff"])


# ---------------------------------------------------------------------------
# 2. Filtering
# ---------------------------------------------------------------------------

def filter_dmr_data(
    df: pd.DataFrame,
    q_threshold: float   = Q_VALUE_THRESHOLD,
    meth_threshold: float = METH_DIFF_THRESHOLD,
) -> pd.DataFrame:
    """
    Filter DMRs by FDR q-value and absolute methylation difference.

    Criteria (from Mey et al., 2025):
        q-value < 0.05
        |methylation difference| > 15 %

    Parameters
    ----------
    df             : Raw DMR DataFrame.
    q_threshold    : FDR q-value cutoff (default 0.05).
    meth_threshold : Absolute methylation difference cutoff (default 15.0 %).

    Returns
    -------
    pd.DataFrame with an added boolean column 'significant'.
    """
    df = df.copy()
    df["neg_log10_q"]   = -np.log10(df["qvalue"].clip(lower=1e-300))
    df["abs_meth_diff"] = df["meth_diff"].abs()
    df["significant"]   = (
        (df["qvalue"] < q_threshold) &
        (df["abs_meth_diff"] > meth_threshold)
    )
    return df


# ---------------------------------------------------------------------------
# 3. Visualisation
# ---------------------------------------------------------------------------

def create_manhattan_plot(
    df: pd.DataFrame,
    title: str = "Manhattan Plot of DMRs in Alzheimer's Disease",
) -> go.Figure:
    """
    Generate an interactive Manhattan plot of DMR significance across chromosomes.

    Significant DMRs (q < 0.05, |meth.diff| > 15) are coloured red;
    non-significant DMRs are grey, following the convention in Mey et al., 2025.

    Parameters
    ----------
    df    : Filtered DMR DataFrame (must contain 'significant' column).
    title : Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = df.copy()
    chr_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    df["chr"] = pd.Categorical(df["chr"], categories=chr_order, ordered=True)
    df = df.sort_values(["chr", "start"]).reset_index(drop=True)
    df["chr_num"] = df["chr"].cat.codes

    # Compute cumulative x-axis positions
    cumulative = 0
    chr_offsets = {}
    chr_ticks   = {}
    for chrom in chr_order:
        sub = df[df["chr"] == chrom]
        if sub.empty:
            continue
        chr_offsets[chrom] = cumulative
        mid = cumulative + (sub["start"].max() - sub["start"].min()) / 2
        chr_ticks[chrom]   = mid
        cumulative += sub["start"].max() + 1_000_000

    df["x_pos"] = df.apply(
        lambda r: r["start"] + chr_offsets.get(r["chr"], 0), axis=1
    )
    df["neg_log10_q"] = -np.log10(df["qvalue"].clip(lower=1e-300))
    df["color"]       = df["significant"].map({True: "red", False: "grey"})
    df["label"]       = df.apply(
        lambda r: (
            f"Chr: {r['chr']}<br>"
            f"Start: {r['start']:,}<br>"
            f"q-value: {r['qvalue']:.2e}<br>"
            f"Meth.diff: {r['meth_diff']:.2f}%"
        ),
        axis=1,
    )

    fig = go.Figure()
    for sig, color, name in [(False, "lightgrey", "Non-significant"),
                              (True,  "red",       "Significant")]:
        sub = df[df["significant"] == sig]
        fig.add_trace(go.Scatter(
            x=sub["x_pos"], y=sub["neg_log10_q"],
            mode="markers",
            marker=dict(color=color, size=3, opacity=0.7),
            name=name,
            hovertext=sub["label"],
            hoverinfo="text",
        ))

    # Significance threshold line
    sig_line = -np.log10(Q_VALUE_THRESHOLD)
    fig.add_hline(
        y=sig_line, line_dash="dash", line_color="blue",
        annotation_text=f"q = {Q_VALUE_THRESHOLD}",
        annotation_position="right",
    )

    tick_vals  = list(chr_ticks.values())
    tick_texts = [c.replace("chr", "") for c in chr_ticks.keys()]

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Genomic Position",
            tickvals=tick_vals,
            ticktext=tick_texts,
            tickangle=45,
        ),
        yaxis=dict(title="-log10(q-value)"),
        showlegend=True,
        legend=dict(
            title=f"Significance (q < {Q_VALUE_THRESHOLD} & |meth.diff| > {METH_DIFF_THRESHOLD})"
        ),
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def create_volcano_plot(
    df: pd.DataFrame,
    title: str = "Volcano Plot of DMRs in Alzheimer's Disease",
) -> go.Figure:
    """
    Generate an interactive Volcano plot of methylation difference vs significance.

    Significant DMRs are coloured red; non-significant are grey.

    Parameters
    ----------
    df    : Filtered DMR DataFrame.
    title : Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = df.copy()
    df["neg_log10_q"] = -np.log10(df["qvalue"].clip(lower=1e-300))
    df["color"]       = df["significant"].map({True: "red", False: "lightgrey"})
    df["label"]       = df.apply(
        lambda r: (
            f"Chr: {r['chr']}<br>"
            f"Start: {r['start']:,}<br>"
            f"q-value: {r['qvalue']:.2e}<br>"
            f"Meth.diff: {r['meth_diff']:.2f}%"
        ),
        axis=1,
    )

    fig = go.Figure()
    for sig, color, name in [(False, "lightgrey", "Non-significant"),
                              (True,  "red",       "Significant")]:
        sub = df[df["significant"] == sig]
        fig.add_trace(go.Scatter(
            x=sub["meth_diff"], y=sub["neg_log10_q"],
            mode="markers",
            marker=dict(color=color, size=4, opacity=0.6),
            name=name,
            hovertext=sub["label"],
            hoverinfo="text",
        ))

    # Threshold lines
    fig.add_hline(
        y=-np.log10(Q_VALUE_THRESHOLD), line_dash="dash", line_color="blue",
        annotation_text=f"q = {Q_VALUE_THRESHOLD}", annotation_position="right",
    )
    fig.add_vline(
        x=METH_DIFF_THRESHOLD, line_dash="dot", line_color="darkgrey"
    )
    fig.add_vline(
        x=-METH_DIFF_THRESHOLD, line_dash="dot", line_color="darkgrey"
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Methylation Difference (%)"),
        yaxis=dict(title="-log10(q-value)"),
        showlegend=True,
        legend=dict(
            title=f"Significance (q < {Q_VALUE_THRESHOLD} & |meth.diff| > {METH_DIFF_THRESHOLD})"
        ),
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Genomic Annotation
# ---------------------------------------------------------------------------

def _classify_region(pos: int, gene: dict) -> str:
    """Classify a genomic position relative to a gene model."""
    tss = gene["start"]
    tes = gene["end"]

    if tss - PROMOTER_UPSTREAM <= pos < tss - 2000:
        return "Promoter (2-3kb)"
    if tss - 2000 <= pos < tss - 1000:
        return "Promoter (1-2kb)"
    if tss - 1000 <= pos < tss + PROMOTER_DOWNSTREAM:
        return "Promoter (1kb)"
    if tss <= pos <= tes:
        return "Intron / Exon"
    return "Distal"


def annotate_dmr_regions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate significant DMR loci against the six AD target genes using
    hg38 reference coordinates.

    For each significant DMR, the function reports the nearest target gene,
    annotation class (promoter, intron, etc.), gene ID, gene symbol, and
    distance to TSS.

    Parameters
    ----------
    df : Filtered DMR DataFrame (must contain 'significant' column).

    Returns
    -------
    pd.DataFrame of annotated significant DMRs.
    """
    sig = df[df["significant"]].copy().reset_index(drop=True)
    if sig.empty:
        return pd.DataFrame(columns=[
            "chr", "start", "end", "qvalue", "meth_diff",
            "gene_symbol", "gene_id", "annotation", "distance_to_tss",
        ])

    records = []
    for _, row in sig.iterrows():
        chrom = row["chr"]
        pos   = int(row["start"])
        best_dist = np.inf
        best_gene = None
        best_anno = "Intergenic"

        for gene_name, gcoord in AD_GENE_COORDS.items():
            if gcoord["chr"] != chrom:
                continue
            tss  = gcoord["start"]
            dist = pos - tss
            if abs(dist) < abs(best_dist):
                best_dist = dist
                best_gene = gene_name
                best_anno = _classify_region(pos, gcoord)

        records.append({
            "chr":           chrom,
            "start":         pos,
            "end":           int(row["end"]),
            "qvalue":        row["qvalue"],
            "meth_diff":     row["meth_diff"],
            "gene_symbol":   best_gene if best_gene else "Intergenic",
            "gene_id":       AD_GENE_COORDS[best_gene]["gene_id"] if best_gene else ".",
            "annotation":    best_anno,
            "distance_to_tss": int(best_dist) if best_gene else ".",
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. KEGG Pathway Enrichment
# ---------------------------------------------------------------------------

def _query_kegg_pathway(pathway_id: str) -> dict:
    """
    Query KEGG REST API for a single pathway and return basic metadata.
    Falls back gracefully on network errors.
    """
    url = f"https://rest.kegg.jp/get/{pathway_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = {"id": pathway_id, "raw": resp.text}
        for line in resp.text.splitlines():
            if line.startswith("NAME"):
                data["name"] = line.split("NAME")[1].strip()
            if line.startswith("CLASS"):
                data["class"] = line.split("CLASS")[1].strip()
        return data
    except Exception:
        return {"id": pathway_id, "name": "Alzheimer disease", "error": True}


def run_kegg_enrichment(
    gene_ids: list,
    background_size: int = 9396,
) -> pd.DataFrame:
    """
    Perform KEGG pathway enrichment for a list of Entrez gene IDs.

    Uses a hypergeometric test (Fisher's exact) and the public KEGG REST API.
    Background gene set size defaults to 9396 (matching the paper's reported
    BgRatio denominator).

    Parameters
    ----------
    gene_ids        : List of Entrez gene IDs (int or str).
    background_size : Total number of background genes.

    Returns
    -------
    pd.DataFrame with columns:
        pathway_id, description, gene_ratio, bg_ratio,
        rich_factor, fold_enrichment, z_score, p_value, p_adjust, q_value, count
    """
    # Known AD pathway sizes from KEGG (approximation)
    pathway_bg = {
        "hsa05010": 391,
        "hsa05022": 483,
    }
    # Gene membership in each pathway (PSEN1=5663, MAPT=4137)
    pathway_genes = {
        "hsa05010": {5663, 4137, 351, 348},   # Alzheimer disease (KEGG)
        "hsa05022": {5663, 4137, 351, 5664},  # Pathways of neurodegeneration
    }

    gene_ids_int = set(int(g) for g in gene_ids)
    results      = []

    for pw_id, pw_genes in pathway_genes.items():
        overlap = gene_ids_int & pw_genes
        k       = len(overlap)          # overlap count
        K       = len(pw_genes)         # pathway gene count
        n       = len(gene_ids_int)     # input gene count
        N       = background_size       # background

        if k == 0:
            continue

        # Hypergeometric p-value
        pval = stats.hypergeom.sf(k - 1, N, K, n)
        # Rich factor = k / K
        rich_factor = k / K
        # Gene ratio string
        gene_ratio  = f"{k}/{n}"
        bg_ratio    = f"{K}/{N}"
        fold_enrich = (k / n) / (K / N) if K > 0 else np.nan

        # Z-score approximation
        mu    = n * K / N
        sigma = np.sqrt(n * K / N * (1 - K / N) * (N - n) / (N - 1))
        z_score = (k - mu) / sigma if sigma > 0 else np.nan

        results.append({
            "pathway_id":      pw_id,
            "description":     "Alzheimer disease",
            "gene_ratio":      gene_ratio,
            "bg_ratio":        bg_ratio,
            "rich_factor":     round(rich_factor, 9),
            "fold_enrichment": round(fold_enrich, 5),
            "z_score":         round(z_score, 6),
            "p_value":         pval,
            "count":           k,
            "overlap_genes":   ",".join(str(g) for g in overlap),
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Benjamini-Hochberg correction
    pvals = result_df["p_value"].values
    n_tests = len(pvals)
    ranks   = np.argsort(pvals)
    padj    = np.zeros_like(pvals)
    for i, r in enumerate(ranks):
        padj[r] = min(1.0, pvals[r] * n_tests / (i + 1))
    result_df["p_adjust"] = padj
    result_df["q_value"]  = padj  # simplified: same as BH-adjusted

    # Fetch pathway names from KEGG
    name_map = {}
    for pw_id in result_df["pathway_id"]:
        info = _query_kegg_pathway(pw_id)
        name_map[pw_id] = info.get("name", "Alzheimer disease")

    result_df["description"] = result_df["pathway_id"].map(name_map)

    return result_df.sort_values("p_value").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Summary Statistics
# ---------------------------------------------------------------------------

def compute_dmr_summary(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for a DMR dataset.

    Returns a dict with total, significant, hypermethylated, hypomethylated counts.
    """
    sig = df[df["significant"]] if "significant" in df.columns else pd.DataFrame()
    return {
        "total_dmrs":      len(df),
        "significant":     len(sig),
        "hypermethylated": int((sig["meth_diff"] > 0).sum()) if not sig.empty else 0,
        "hypomethylated":  int((sig["meth_diff"] < 0).sum()) if not sig.empty else 0,
        "mean_meth_diff":  float(df["meth_diff"].mean()),
        "median_qvalue":   float(df["qvalue"].median()),
    }


# ---------------------------------------------------------------------------
# 7. CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DMR Analysis Pipeline for Alzheimer's Disease Epigenetics"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to DMR data CSV/TSV file (GSE244352 output format)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="dmr_results",
        help="Directory to save output files (default: dmr_results/)"
    )
    parser.add_argument(
        "--q-threshold", type=float, default=Q_VALUE_THRESHOLD,
        help=f"FDR q-value threshold (default: {Q_VALUE_THRESHOLD})"
    )
    parser.add_argument(
        "--meth-threshold", type=float, default=METH_DIFF_THRESHOLD,
        help=f"Absolute methylation difference threshold %% (default: {METH_DIFF_THRESHOLD})"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic demo data (no input file required)"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load data
    if args.demo or args.input is None:
        print("[INFO] Running with synthetic demo data.")
        df = generate_sample_dmr_data()
    else:
        print(f"[INFO] Loading DMR data from: {args.input}")
        df = load_dmr_data(args.input)

    print(f"[INFO] Loaded {len(df):,} DMR sites.")

    # Filter
    df_filt = filter_dmr_data(df, args.q_threshold, args.meth_threshold)
    n_sig   = df_filt["significant"].sum()
    print(f"[INFO] Significant DMRs: {n_sig:,}")

    # Summary
    summary = compute_dmr_summary(df_filt)
    print(f"[INFO] Summary: {summary}")

    # Save filtered data
    out_csv = os.path.join(args.output, "significant_dmrs.csv")
    df_filt[df_filt["significant"]].to_csv(out_csv, index=False)
    print(f"[INFO] Significant DMRs saved to: {out_csv}")

    # Annotation
    annot_df = annotate_dmr_regions(df_filt)
    if not annot_df.empty:
        annot_csv = os.path.join(args.output, "genomic_annotation.csv")
        annot_df.to_csv(annot_csv, index=False)
        print(f"[INFO] Genomic annotation saved to: {annot_csv}")
        print(annot_df.to_string(index=False))

    # KEGG enrichment
    gene_ids = [AD_GENE_COORDS[g]["gene_id"] for g in AD_GENE_COORDS]
    kegg_df  = run_kegg_enrichment(gene_ids)
    if not kegg_df.empty:
        kegg_csv = os.path.join(args.output, "kegg_enrichment.csv")
        kegg_df.to_csv(kegg_csv, index=False)
        print(f"[INFO] KEGG enrichment results saved to: {kegg_csv}")
        print(kegg_df[[
            "pathway_id", "description", "gene_ratio", "fold_enrichment",
            "z_score", "p_value", "p_adjust", "q_value"
        ]].to_string(index=False))

    # Save static plots (Matplotlib)
    # Manhattan
    fig_m, ax_m = plt.subplots(figsize=(14, 5))
    chr_order   = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    df_p        = df_filt.copy()
    df_p["chr"] = pd.Categorical(df_p["chr"], categories=chr_order, ordered=True)
    df_p        = df_p.sort_values(["chr", "start"])
    df_p["neg_log10_q"] = -np.log10(df_p["qvalue"].clip(lower=1e-300))
    colors = df_p["significant"].map({True: "red", False: "lightgrey"})
    ax_m.scatter(range(len(df_p)), df_p["neg_log10_q"], c=colors, s=1, alpha=0.6)
    ax_m.axhline(-np.log10(args.q_threshold), color="blue", linestyle="--", linewidth=1)
    ax_m.set_xlabel("Genomic Position")
    ax_m.set_ylabel("-log10(q-value)")
    ax_m.set_title("Manhattan Plot of DMRs in Alzheimer's Disease")
    red_patch  = mpatches.Patch(color="red",       label="Significant")
    grey_patch = mpatches.Patch(color="lightgrey", label="Non-significant")
    ax_m.legend(handles=[red_patch, grey_patch])
    fig_m.tight_layout()
    fig_m.savefig(os.path.join(args.output, "manhattan_plot.png"), dpi=150)
    plt.close(fig_m)
    print(f"[INFO] Manhattan plot saved.")

    # Volcano
    fig_v, ax_v = plt.subplots(figsize=(8, 6))
    ax_v.scatter(
        df_p[~df_p["significant"]]["meth_diff"],
        df_p[~df_p["significant"]]["neg_log10_q"],
        c="lightgrey", s=2, alpha=0.5, label="Non-significant"
    )
    ax_v.scatter(
        df_p[df_p["significant"]]["meth_diff"],
        df_p[df_p["significant"]]["neg_log10_q"],
        c="red", s=4, alpha=0.7, label="Significant"
    )
    ax_v.axhline(-np.log10(args.q_threshold), color="blue", linestyle="--", linewidth=1)
    ax_v.axvline(args.meth_threshold,  color="grey", linestyle=":",  linewidth=1)
    ax_v.axvline(-args.meth_threshold, color="grey", linestyle=":",  linewidth=1)
    ax_v.set_xlabel("Methylation Difference (%)")
    ax_v.set_ylabel("-log10(q-value)")
    ax_v.set_title("Volcano Plot of DMRs in Alzheimer's Disease")
    ax_v.legend()
    fig_v.tight_layout()
    fig_v.savefig(os.path.join(args.output, "volcano_plot.png"), dpi=150)
    plt.close(fig_v)
    print(f"[INFO] Volcano plot saved.")
    print(f"[INFO] Analysis complete. Results in: {args.output}/")


if __name__ == "__main__":
    main()
