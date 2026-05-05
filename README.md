# Alzheimer's Disease Epigenetics & AI Detection Pipeline

An interactive bioinformatics pipeline for detecting Alzheimer's Disease (AD)
through combined epigenetic profiling and AI-driven transcriptomic classification,
implemented as a Streamlit web application.

---

## Citation

If you use this pipeline in your research, please cite:

> **Mey, T. V. K. Y., Lee, M. B., Tomodok, A. C. A., Muliana, N. E., Priskila, D.,
> Sadrawi, M., & Parikesit, A. A. (2025). Detection of Alzheimer's Disease through
> AI-Driven and Methylation Difference Region Analysis of Significant Epigenetic
> Modifications in APP, PSEN1, PSEN2, APOE, MAPT, and TREM2 Genes. In A. ISYAKU (Ed.),
> *ADVANCED THERAPEUTICS AND DISEASE BIOLOGY: MOLECULAR DIAGNOSTICS AND IMMUNITY- 2025*
> (pp. 57–86). Halic Publishing House.**
> https://doi.org/10.5281/zenodo.18070507

BibTeX:

```bibtex
@incollection{mey2025alzheimer,
  author    = {Mey, Theshia Veronica Kusuma Yun and Lee, Michael Branson and
               Tomodok, Angelo Christiano Aouad and Muliana, Nathaniel Emmanuel and
               Priskila, Dhea and Sadrawi, Muammar and Parikesit, Arli Aditya},
  title     = {Detection of {Alzheimer's} Disease through {AI}-Driven and
               Methylation Difference Region Analysis of Significant Epigenetic
               Modifications in {APP}, {PSEN1}, {PSEN2}, {APOE}, {MAPT}, and
               {TREM2} Genes},
  booktitle = {Advanced Therapeutics and Disease Biology: Molecular Diagnostics
               and Immunity -- 2025},
  editor    = {Isyaku, A.},
  pages     = {57--86},
  publisher = {Halic Publishing House},
  year      = {2025},
  doi       = {10.5281/zenodo.18070507},
  url       = {https://doi.org/10.5281/zenodo.18070507}
}
```

---

## Overview

This repository implements the dual-pipeline approach from the paper above.
Two complementary analytical strategies are combined:

### Pipeline 1: Differential Methylated Region (DMR) Analysis

- **Dataset**: GSE244352 (NCBI GEO) - methylation capture sequencing from peripheral
  blood of 12 clinically diagnosed AD patients and 12 controls
- **Filtering**: q-value < 0.05, |methylation difference| > 15%
- **Outputs**: Manhattan plot, Volcano plot, genomic annotation (hg38), KEGG enrichment
- **Key findings**: Significant DMRs in PSEN1 promoter (Chr14) and MAPT intron (Chr17)

### Pipeline 2: AI / MLP Classification

- **Datasets**: GSE48350 + GSE11882 (NCBI GEO) - postmortem brain microarray
- **Platform**: Affymetrix HG-U133 Plus 2.0 (GPL570)
- **Model**: Multilayer Perceptron (TensorFlow/Keras), 200 epochs
- **Target genes**: APP, PSEN1, PSEN2, APOE, MAPT, TREM2
- **Classification**: Braak stage 0-II = Normal; Braak III-VI = AD

---

## Repository Structure

```
.
├── README.md                   This file
├── requirements.txt            Python dependencies
├── .gitignore                  Git exclusion rules
├── CLAUDE.md                   Project AI context (Dr. Arli Aditya Parikesit)
├── SCRIPT/
│   ├── app.py                  Streamlit web application (entry point)
│   ├── dmr_analysis.py         DMR analysis functions and CLI
│   └── ml_classification.py   MLP classification functions and CLI
└── SKILL/
    └── SKILL.md                Detailed pipeline documentation
```

---

## Installation

### Requirements

- Python 3.9 or higher
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/alzheimer-epigenetics.git
cd alzheimer-epigenetics

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Streamlit Application (recommended)

```bash
cd SCRIPT
streamlit run app.py
```

Then open `http://localhost:8501` in your browser. The application provides:

- **Home** - Pipeline overview and gene reference table
- **DMR Analysis** - Upload data or use demo mode, interactive plots
- **AI / MLP Classification** - Train and evaluate the MLP model
- **Citation & About** - Full reference and BibTeX entry

### Command-Line Interface

**DMR Analysis:**

```bash
# Demo mode (synthetic data, no download required)
python SCRIPT/dmr_analysis.py --demo --output results/dmr/

# With real DMR data
python SCRIPT/dmr_analysis.py --input path/to/dmr_data.csv --output results/dmr/

# Custom significance thresholds
python SCRIPT/dmr_analysis.py --input data.csv --q-threshold 0.01 --meth-threshold 20
```

**ML Classification:**

```bash
# Demo mode
python SCRIPT/ml_classification.py --demo --output results/ml/

# Download from GEO (requires internet access and GEOparse)
python SCRIPT/ml_classification.py --fetch-geo --output results/ml/

# Pre-processed expression CSV
python SCRIPT/ml_classification.py --input expression_data.csv --output results/ml/
```

---

## Input Data Formats

### DMR Data (GSE244352 format)

| Column    | Type   | Description                                 |
|-----------|--------|---------------------------------------------|
| chr       | string | Chromosome (e.g. chr14)                     |
| start     | int    | Start genomic position (hg38)               |
| end       | int    | End genomic position                        |
| pvalue    | float  | Raw p-value                                 |
| qvalue    | float  | FDR-adjusted q-value                        |
| meth_diff | float  | Methylation difference % (AD minus control) |

Column names are case-insensitive and accept common variants
(e.g. `meth.diff`, `p.value`, `q.value`).

### Expression Data (pre-processed)

| Column     | Type   | Description                                       |
|------------|--------|---------------------------------------------------|
| APP        | float  | Averaged probe expression                         |
| PSEN1      | float  | Averaged probe expression                         |
| PSEN2      | float  | Averaged probe expression                         |
| APOE       | float  | Averaged probe expression                         |
| MAPT       | float  | Averaged probe expression                         |
| TREM2      | float  | Averaged probe expression                         |
| age        | float  | Age at death                                      |
| sex        | string | "F" or "M"                                        |
| region     | string | "HC", "EC", "SG", or "PCG"                        |
| label      | int    | 0 = Normal (Braak 0-II), 1 = AD (Braak III-VI)   |

---

## Key Results Summary

### DMR Analysis

Three genomic loci showed significant differential methylation between AD patients
and controls:

| Chr   | Position  | Meth.Diff | Gene  | Feature Type  | Significance       |
|-------|-----------|-----------|-------|---------------|--------------------|
| Chr14 | 73113602  | -30.09%   | PSEN1 | Promoter 2-3kb| p=0, q=0           |
| Chr14 | 73198335  | +19.11%   | PSEN1 | Promoter 1-2kb| p=0, q=0           |
| Chr17 | 45889839  | -24.09%   | MAPT  | Intron 1 of 6 | p=8.1e-10, q=1.5e-7 |

KEGG enrichment confirmed both PSEN1 and MAPT are enriched in the Alzheimer's
disease pathway (hsa05010, fold enrichment ~24) and neurodegeneration pathway
(hsa05022, fold enrichment ~19.5).

### MLP Classification

| Metric            | Value   |
|-------------------|---------|
| Overall Accuracy  | ~88%    |
| Specificity       | ~97%    |
| Sensitivity (AD)  | ~38%    |
| Limitation        | Class imbalance (Normal:AD ~ 73:13 in test set) |

The model shows strong specificity but limited sensitivity for AD detection,
attributed to class imbalance in the training data.

---

## Biological Context

**PSEN1** encodes Presenilin 1, the catalytic subunit of gamma-secretase, which
cleaves APP to produce amyloid beta peptides. The opposing methylation patterns
in its promoter region suggest complex, cell-type-specific regulatory mechanisms
in AD pathogenesis.

**MAPT** encodes the tau protein critical for microtubule stabilisation.
Intronic hypomethylation at Chr17:45889839 may alter alternative splicing of
tau isoforms, some of which (e.g. 3R-tau from V337M variant) promote
neurofibrillary tangle formation.

---

## Reproducibility

All analyses use publicly available data and open-source tools:

- GEO data is downloaded programmatically (no login required)
- KEGG enrichment uses the public REST API (no API key required)
- Random seeds are fixed for reproducible ML training
- All intermediate outputs are saved to disk

No proprietary data, passwords, or API keys are required to run this pipeline.

---

## Disclaimer

This tool is intended for research and educational purposes. It does not constitute
medical advice and must not be used for clinical diagnosis. All computational
predictions require independent experimental validation before biological or
clinical conclusions can be drawn.

---

## Authors

- Theshia Veronica Kusuma Yun Mey
- Michael Branson Lee
- Angelo Christiano Aouad Tomodok
- Nathaniel Emmanuel Muliana
- Dhea Priskila
- Muammar Sadrawi
- Prof. Dr. Arli Aditya Parikesit

**Indonesia International Institute for Life Sciences (i3L)**
School of Life Sciences, Jakarta, Indonesia

---

## License

MIT License. See [LICENSE](LICENSE) for details.
