---
name: alzheimer-epigenetics-ai-pipeline
version: "1.0"
title: "Alzheimer's Disease Epigenetics & AI Detection Pipeline"
description: >
  Integrated bioinformatics pipeline combining Differential Methylated Region
  (DMR) analysis and Multilayer Perceptron (MLP) classification for Alzheimer's
  Disease detection. Targets APP, PSEN1, PSEN2, APOE, MAPT, and TREM2 genes.
authors:
  - name: Theshia Veronica Kusuma Yun Mey
    affiliation: i3L University, Jakarta
  - name: Michael Branson Lee
    affiliation: i3L University, Jakarta
  - name: Angelo Christiano Aouad Tomodok
    affiliation: i3L University, Jakarta
  - name: Nathaniel Emmanuel Muliana
    affiliation: i3L University, Jakarta
  - name: Dhea Priskila
    affiliation: i3L University, Jakarta
  - name: Muammar Sadrawi
    affiliation: i3L University, Jakarta
  - name: Arli Aditya Parikesit
    affiliation: i3L University, Jakarta
    orcid: "0000-0001-8716-3926"
citation:
  apa: >
    Mey, T. V. K. Y., Lee, M. B., Tomodok, A. C. A., Muliana, N. E.,
    Priskila, D., Sadrawi, M., & Parikesit, A. A. (2025). Detection of
    Alzheimer's Disease through AI-Driven and Methylation Difference Region
    Analysis of Significant Epigenetic Modifications in APP, PSEN1, PSEN2,
    APOE, MAPT, and TREM2 Genes. In A. ISYAKU (Ed.), Advanced Therapeutics
    and Disease Biology: Molecular Diagnostics and Immunity - 2025 (pp. 57-86).
    Halic Publishing House.
  doi: "10.5281/zenodo.18070507"
  url: "https://doi.org/10.5281/zenodo.18070507"
data_sources:
  dmr_analysis:
    geo_id: GSE244352
    type: methylation_capture_sequencing
    platform: Illumina HiSeq 2500
    samples: "12 AD patients + 12 healthy controls"
    reference: "Mitsumori et al., 2025 (Clinical Epigenetics, 17(1), 107)"
  ml_classification:
    geo_ids:
      - GSE48350
      - GSE11882
    type: brain_microarray
    platform: "Affymetrix HG-U133 Plus 2.0 (GPL570)"
    brain_regions: [HC, EC, SG, PCG]
    classification: "Braak 0-II = Normal; Braak III-VI = AD"
target_genes:
  - symbol: APP
    full_name: Amyloid Precursor Protein
    chr_hg38: chr21
    gene_id: 351
  - symbol: PSEN1
    full_name: Presenilin 1
    chr_hg38: chr14
    gene_id: 5663
  - symbol: PSEN2
    full_name: Presenilin 2
    chr_hg38: chr1
    gene_id: 5664
  - symbol: APOE
    full_name: Apolipoprotein E
    chr_hg38: chr19
    gene_id: 348
  - symbol: MAPT
    full_name: Microtubule-Associated Protein Tau
    chr_hg38: chr17
    gene_id: 4137
  - symbol: TREM2
    full_name: Triggering Receptor on Myeloid Cells 2
    chr_hg38: chr6
    gene_id: 54209
parameters:
  dmr:
    q_value_threshold: 0.05
    meth_diff_threshold: 15.0
    genome_assembly: hg38
    promoter_upstream_bp: 2000
    promoter_downstream_bp: 500
    kegg_pathways: [hsa05010, hsa05022]
  ml:
    architecture: "Dense(64,relu)->Dropout(0.3)->Dense(32,relu)->Dropout(0.2)->Dense(1,sigmoid)"
    optimiser: Adam
    learning_rate: 0.001
    loss: binary_crossentropy
    epochs: 200
    batch_size: 16
    train_test_split: 0.80
    random_seed: 42
    braak_normal_max: 2
    braak_ad_min: 3
entry_points:
  streamlit_app: "SCRIPT/app.py"
  dmr_cli: "SCRIPT/dmr_analysis.py"
  ml_cli: "SCRIPT/ml_classification.py"
tags:
  - alzheimer
  - epigenetics
  - dmr
  - methylation
  - mlp
  - transcriptomics
  - geo
  - kegg
  - bioinformatics
  - neurodegenerative
created: "2026-05-05"
license: MIT
---

# SKILL: Alzheimer's Disease Epigenetics & AI Detection Pipeline

## Overview

This skill implements the integrated bioinformatics pipeline described in
**Mey et al. (2025)** for detecting Alzheimer's Disease (AD) through:

1. **DMR Analysis** - Differential Methylated Region profiling of blood-derived methylation data (GSE244352)
2. **AI/MLP Classification** - Multilayer Perceptron binary classification from brain transcriptomic data (GSE48350 + GSE11882)

Both analyses converge on six established AD biomarker genes: **APP, PSEN1, PSEN2, APOE, MAPT, TREM2**.

---

## Citation

> Mey, T. V. K. Y., Lee, M. B., Tomodok, A. C. A., Muliana, N. E., Priskila, D.,
> Sadrawi, M., & Parikesit, A. A. (2025). Detection of Alzheimer's Disease through
> AI-Driven and Methylation Difference Region Analysis of Significant Epigenetic
> Modifications in APP, PSEN1, PSEN2, APOE, MAPT, and TREM2 Genes. In A. ISYAKU (Ed.),
> *ADVANCED THERAPEUTICS AND DISEASE BIOLOGY: MOLECULAR DIAGNOSTICS AND IMMUNITY- 2025*
> (pp. 57–86). Halic Publishing House.
> https://doi.org/10.5281/zenodo.18070507

---

## Pipeline 1: DMR Analysis

### Data

| Field       | Value                                                        |
|-------------|--------------------------------------------------------------|
| GEO ID      | GSE244352                                                    |
| Type        | DNA methylation capture sequencing (peripheral blood)        |
| Samples     | 12 clinically diagnosed AD patients + 12 healthy controls    |
| Platform    | Illumina HiSeq 2500                                          |
| Instrument  | Maxwell RSC (Promega) + Maxwell RSC Buffy Coat DNA Kit       |
| Source ref  | Mitsumori et al., 2025 (Clinical Epigenetics, 17(1), 107)   |

### Input Data Format

CSV or TSV file with the following columns:

```
chr        - Chromosome identifier (e.g. chr14)
start      - Genomic start position (hg38 coordinates)
end        - Genomic end position
pvalue     - Raw p-value
qvalue     - FDR-adjusted q-value (Benjamini-Hochberg)
meth_diff  - Methylation difference (%) between AD and control (positive = hypermethylated)
```

### Filtering Criteria

| Parameter                  | Threshold | Rationale                                    |
|----------------------------|-----------|----------------------------------------------|
| FDR q-value                | < 0.05    | Standard false discovery rate control        |
| Absolute methylation diff  | > 15%     | Biologically meaningful methylation change   |

### Analytical Steps

1. **Data Loading** (`load_dmr_data`) - Parses CSV/TSV with column name normalisation
2. **Filtering** (`filter_dmr_data`) - Applies q-value and methylation difference thresholds
3. **Manhattan Plot** (`create_manhattan_plot`) - Genome-wide significance landscape
4. **Volcano Plot** (`create_volcano_plot`) - Effect size vs. significance
5. **Genomic Annotation** (`annotate_dmr_regions`) - Maps DMRs to six target genes (hg38)
6. **KEGG Enrichment** (`run_kegg_enrichment`) - Hypergeometric test + KEGG REST API

### Target Gene Coordinates (hg38)

| Gene  | Chr   | Start      | End        | Gene ID |
|-------|-------|------------|------------|---------|
| APP   | chr21 | 25880550   | 26170620   | 351     |
| PSEN1 | chr14 | 73113418   | 73223691   | 5663    |
| PSEN2 | chr1  | 226870384  | 226942614  | 5664    |
| APOE  | chr19 | 44905791   | 44909393   | 348     |
| MAPT  | chr17 | 45889382   | 46028334   | 4137    |
| TREM2 | chr6  | 41161514   | 41167971   | 54209   |

### Key Findings (Paper Results)

| Chr   | Position  | Q-value  | Meth.Diff (%) | Gene  | Annotation     |
|-------|-----------|----------|----------------|-------|----------------|
| Chr14 | 73113602  | ~0       | -30.09         | PSEN1 | Promoter (2-3kb) |
| Chr14 | 73198335  | ~0       | +19.11         | PSEN1 | Promoter (1-2kb) |
| Chr17 | 45889839  | 1.47e-7  | -24.09         | MAPT  | Intron 1 of 6  |

### KEGG Enrichment Results

| Pathway ID | Description          | GeneRatio | FoldEnrichment | Z-score | p-value   | p.adjust |
|------------|----------------------|-----------|----------------|---------|-----------|----------|
| hsa05010   | Alzheimer disease    | 2/2       | 24.03          | 6.79    | 0.001727  | 0.010549 |
| hsa05022   | Neuro degeneration   | 2/2       | 19.45          | 6.08    | 0.002637  | 0.010549 |

---

## Pipeline 2: AI / MLP Classification

### Data

| Field       | Value                                                            |
|-------------|------------------------------------------------------------------|
| GEO IDs     | GSE48350, GSE11882                                               |
| Type        | Postmortem brain microarray (gene expression)                    |
| Platform    | Affymetrix HG-U133 Plus 2.0 (GPL570)                            |
| Brain regions | Hippocampus (HC), Entorhinal Cortex (EC), Superior Frontal Gyrus (SG), Postcentral Gyrus (PCG) |
| Classification | Braak stage 0-II = Normal (0); Braak stage III-VI = AD (1) |

### Affymetrix Probe IDs (GPL570)

| Gene  | Probe IDs                                      |
|-------|------------------------------------------------|
| APP   | 207317_s_at, 214953_s_at, 207318_s_at          |
| PSEN1 | 202627_s_at, 202628_s_at                       |
| PSEN2 | 204465_s_at, 204466_s_at                       |
| APOE  | 203382_s_at                                    |
| MAPT  | 203132_at, 209173_at                           |
| TREM2 | 220461_s_at                                    |

### Feature Matrix

| Feature Group     | Features                                            |
|-------------------|-----------------------------------------------------|
| Gene expression   | APP, PSEN1, PSEN2, APOE, MAPT, TREM2 (averaged across probes) |
| Demographics      | age, sex_F, sex_M                                   |
| Brain region (OHE)| region_HC, region_EC, region_SG, region_PCG         |

All features are normalised with `StandardScaler`.

### MLP Architecture

```
Input(n_features)
    -> Dense(64, activation='relu')
    -> Dropout(0.3)
    -> Dense(32, activation='relu')
    -> Dropout(0.2)
    -> Dense(1, activation='sigmoid')

Optimiser : Adam (lr=0.001)
Loss      : Binary cross-entropy
Epochs    : 200
Batch     : 16
Split     : 80% train / 20% test (stratified)
```

### Evaluation Metrics

| Metric                          | Value (Paper) |
|---------------------------------|---------------|
| Overall Accuracy                | ~88%          |
| True Negatives (Normal correct) | 71/73         |
| Specificity (Normal)            | ~97%          |
| True Positives (AD correct)     | 5/13          |
| Sensitivity (AD)                | ~38%          |
| Limitation                      | Class imbalance (Normal >> AD) |

---

## File Structure

```
SCRIPT/
    app.py               - Streamlit web application (interactive pipeline)
    dmr_analysis.py      - DMR analysis functions and CLI entry point
    ml_classification.py - MLP model functions and CLI entry point

SKILL/
    SKILL.md             - This document

requirements.txt         - Python package dependencies
README.md                - Repository overview and usage guide
.gitignore               - Git exclusion rules
```

---

## Usage

### Streamlit App

```bash
cd SCRIPT
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### DMR Analysis (CLI)

```bash
# Demo mode (synthetic data)
python SCRIPT/dmr_analysis.py --demo --output results/dmr/

# Real data
python SCRIPT/dmr_analysis.py --input your_dmr_data.csv --output results/dmr/

# Custom thresholds
python SCRIPT/dmr_analysis.py --input data.csv --q-threshold 0.01 --meth-threshold 20
```

### ML Classification (CLI)

```bash
# Demo mode
python SCRIPT/ml_classification.py --demo --output results/ml/

# Download from GEO
python SCRIPT/ml_classification.py --fetch-geo --output results/ml/

# Pre-processed file
python SCRIPT/ml_classification.py --input expression_data.csv --output results/ml/
```

---

## Dependencies

Core packages (see `requirements.txt`):
- `streamlit >= 1.32.0`
- `tensorflow >= 2.14.0`
- `scikit-learn >= 1.3.0`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `plotly >= 5.18.0`
- `matplotlib >= 3.8.0`
- `seaborn >= 0.13.0`
- `scipy >= 1.11.0`
- `requests >= 2.31.0`
- `GEOparse >= 2.0.4`

---

## Biological Interpretation Guide

### DMR Findings

- **Hypomethylation** of PSEN1 promoter (Chr14:73113602, -30%) may increase PSEN1 expression,
  elevating gamma-secretase activity and amyloid beta production.
- **Hypermethylation** of PSEN1 promoter (Chr14:73198335, +19%) may suppress transcription
  in a cell-type-specific or stage-dependent manner.
- **Hypomethylation** of MAPT intron 1 (Chr17:45889839, -24%) may alter tau isoform
  splicing, contributing to neurofibrillary tangle formation.

### KEGG Pathways

- **hsa05010** - Alzheimer disease: PSEN1 mutations affect gamma-secretase
  (amyloid beta production), calcium signalling (SERCA binding), and axonal transport (GSK-3).
- **hsa05022** - Pathways of neurodegeneration: Multi-disease pathway covering
  ALS, Parkinson, Huntington, SCA, and prion disease, all sharing neuroinflammation
  and protein aggregation mechanisms.

### MLP Classification Caveats

1. Class imbalance (more Normal than AD samples) biases predictions toward Normal.
2. Braak stages 0-2 may include early pathological changes not yet fulfilling clinical criteria.
3. Model uses transcriptomic data from four brain regions, not peripheral blood.
4. External validation on independent cohorts is required before clinical use.

---

## Version

- Skill version: 1.0
- Created: May 2026
- Based on: Mey et al. (2025), DOI: 10.5281/zenodo.18070507
- Maintainer: Dr. Arli Aditya Parikesit, i3L University Jakarta
