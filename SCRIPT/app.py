#!/usr/bin/env python3
"""
Streamlit Application: Alzheimer's Disease Epigenetics & AI Detection Pipeline
===============================================================================
Interactive web application implementing the dual-pipeline approach from
Mey et al. (2025):
  1. DMR Analysis  - Differentially Methylated Region profiling of GSE244352
  2. AI/MLP        - Multilayer Perceptron classification of AD vs Normal
                     from transcriptomic data (GSE48350 + GSE11882)

Target Genes: APP, PSEN1, PSEN2, APOE, MAPT, TREM2

Reference:
    Mey, T. V. K. Y., Lee, M. B., Tomodok, A. C. A., Muliana, N. E.,
    Priskila, D., Sadrawi, M., & Parikesit, A. A. (2025). Detection of
    Alzheimer's Disease through AI-Driven and Methylation Difference Region
    Analysis of Significant Epigenetic Modifications in APP, PSEN1, PSEN2,
    APOE, MAPT, and TREM2 Genes. In ADVANCED THERAPEUTICS AND DISEASE BIOLOGY:
    MOLECULAR DIAGNOSTICS AND IMMUNITY- 2025 (pp. 57-86). Halic Publishing House.
    https://doi.org/10.5281/zenodo.18070507

Usage:
    streamlit run app.py

Requirements:
    See requirements.txt
"""

import os
import sys
import io
import warnings

import numpy as np
#p = p / p.sum()
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Add the SCRIPT directory to path so sibling modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dmr_analysis import (
    generate_sample_dmr_data,
    load_dmr_data,
    filter_dmr_data,
    create_manhattan_plot,
    create_volcano_plot,
    annotate_dmr_regions,
    run_kegg_enrichment,
    compute_dmr_summary,
    AD_GENE_COORDS,
    Q_VALUE_THRESHOLD,
    METH_DIFF_THRESHOLD,
)
from ml_classification import (
    generate_sample_expression_data,
    preprocess_features,
    build_mlp_model,
    build_sklearn_mlp,
    train_model,
    evaluate_model,
    create_confusion_matrix_plot,
    create_roc_plot,
    create_training_history_plot,
    TARGET_GENES,
    EPOCHS,
    GEO_ACCESSIONS,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AD Epigenetics & AI Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Alzheimer Epigenetics & AI Detection Pipeline**\n\n"
            "Based on: Mey et al. (2025). DOI: 10.5281/zenodo.18070507"
        )
    },
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 6px 0;
    }
    .section-header {
        color: #1f4e79;
        border-bottom: 2px solid #1f4e79;
        padding-bottom: 4px;
        margin-bottom: 12px;
    }
    .cite-box {
        background: #eef4fb;
        border-left: 4px solid #1f4e79;
        padding: 10px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/"
        "Neuron_-_annotated.svg/240px-Neuron_-_annotated.svg.png",
        use_container_width=True,
    )
    st.title("Navigation")
    page = st.radio(
        "Go to",
        options=[
            "Home",
            "DMR Analysis",
            "AI / MLP Classification",
            "Citation & About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "Mey et al., 2025\n"
        "DOI: [10.5281/zenodo.18070507](https://doi.org/10.5281/zenodo.18070507)"
    )


# ===========================================================================
# PAGE: Home
# ===========================================================================

def page_home():
    st.title("Alzheimer's Disease Epigenetics & AI Detection Pipeline")
    st.markdown("""
This application implements the integrated analytical framework from **Mey et al. (2025)**,
combining epigenetic profiling with AI-driven classification for Alzheimer's Disease (AD)
biomarker discovery.
    """)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### Pipeline 1: DMR Analysis")
        st.markdown("""
**Dataset**: GSE244352 (GEO) - Methylation capture sequencing
**Samples**: 12 AD patients, 12 healthy controls
**Platform**: Illumina HiSeq 2500

**Steps performed:**
- Data loading and cleaning
- Filtering: q-value < 0.05 and |meth.diff| > 15%
- Manhattan plot visualisation
- Volcano plot visualisation
- Genomic annotation (hg38) for APP, PSEN1, PSEN2, APOE, MAPT, TREM2
- KEGG pathway enrichment (clusterProfiler equivalent)
        """)

    with col2:
        st.markdown("### Pipeline 2: AI / MLP Classification")
        st.markdown("""
**Datasets**: GSE48350 + GSE11882 (GEO) - Brain microarray
**Platform**: Affymetrix HG-U133 Plus 2.0 (GPL570)
**Brain regions**: HC, EC, SG, PCG

**Steps performed:**
- GEO data extraction via GEOparse
- Feature construction for 6 target genes
- Demographic integration (age, sex, Braak stage, brain region)
- StandardScaler normalisation + 80/20 split
- MLP training (TensorFlow/Keras, 200 epochs)
- Evaluation: accuracy, precision, recall, F1, AUC
        """)

    st.markdown("---")
    st.markdown("### Target Genes")
    gene_data = {
        "Gene":        ["APP", "PSEN1", "PSEN2", "APOE", "MAPT", "TREM2"],
        "Full Name":   [
            "Amyloid Precursor Protein",
            "Presenilin 1",
            "Presenilin 2",
            "Apolipoprotein E",
            "Microtubule-Associated Protein Tau",
            "Triggering Receptor on Myeloid Cells 2",
        ],
        "Chr (hg38)":  ["chr21", "chr14", "chr1", "chr19", "chr17", "chr6"],
        "Role in AD":  [
            "Central to amyloid beta production",
            "Gamma-secretase catalytic subunit",
            "Gamma-secretase component",
            "Strongest LOAD genetic risk factor",
            "Tau protein, neurofibrillary tangles",
            "Microglial amyloid clearance",
        ],
    }
    st.dataframe(
        pd.DataFrame(gene_data),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown('<div class="cite-box"><b>Citation:</b> Mey, T. V. K. Y., Lee, M. B., Tomodok, A. C. A., Muliana, N. E., '
                'Priskila, D., Sadrawi, M., &amp; Parikesit, A. A. (2025). Detection of Alzheimer\'s Disease through '
                'AI-Driven and Methylation Difference Region Analysis of Significant Epigenetic Modifications in APP, '
                'PSEN1, PSEN2, APOE, MAPT, and TREM2 Genes. In <i>ADVANCED THERAPEUTICS AND DISEASE BIOLOGY: MOLECULAR '
                'DIAGNOSTICS AND IMMUNITY- 2025</i> (pp. 57–86). Halic Publishing House. '
                '<a href="https://doi.org/10.5281/zenodo.18070507">https://doi.org/10.5281/zenodo.18070507</a>'
                '</div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE: DMR Analysis
# ===========================================================================

def page_dmr():
    st.title("DMR Analysis Pipeline")
    st.markdown("Differential Methylated Region analysis of GSE244352 (Mey et al., 2025).")

    # --- Data source ---
    st.markdown("### Step 1: Load Data")
    data_mode = st.radio(
        "Data source",
        ["Use built-in demo data (synthetic GSE244352-like)", "Upload your own CSV/TSV"],
        horizontal=True,
    )

    df_raw = None
    if data_mode.startswith("Use built-in"):
        st.info(
            "Demo data uses 5,000 synthetic DMR sites calibrated to resemble "
            "GSE244352, including the three paper-reported significant loci "
            "(PSEN1 on Chr14 and MAPT on Chr17)."
        )
        if st.button("Generate Demo Data", type="primary"):
            with st.spinner("Generating synthetic DMR data..."):
                df_raw = generate_sample_dmr_data()
            st.session_state["dmr_raw"] = df_raw
        elif "dmr_raw" in st.session_state:
            df_raw = st.session_state["dmr_raw"]
    else:
        uploaded = st.file_uploader(
            "Upload DMR data (CSV or TSV)",
            type=["csv", "tsv", "txt"],
            help=(
                "Required columns: chr, start, end, pvalue (or p.value), "
                "qvalue (or q.value), meth_diff (or meth.diff)"
            ),
        )
        if uploaded:
            try:
                raw_bytes = uploaded.read()
                # Try tab then comma separator
                for sep in ["\t", ","]:
                    try:
                        df_raw = pd.read_csv(io.StringIO(raw_bytes.decode()), sep=sep)
                        if df_raw.shape[1] > 1:
                            break
                    except Exception:
                        pass
                df_raw = load_dmr_data(uploaded.name) if df_raw is None else df_raw
                st.session_state["dmr_raw"] = df_raw
                st.success(f"Loaded {len(df_raw):,} DMR sites.")
            except Exception as e:
                st.error(f"Failed to load file: {e}")

    if df_raw is None and "dmr_raw" in st.session_state:
        df_raw = st.session_state["dmr_raw"]

    if df_raw is None:
        st.warning("Please load or generate data to proceed.")
        return

    st.success(f"Dataset loaded: {len(df_raw):,} DMR sites.")
    with st.expander("Preview raw data"):
        st.dataframe(df_raw.head(20), use_container_width=True)

    # --- Filtering parameters ---
    st.markdown("### Step 2: Filter Significant DMRs")
    col1, col2 = st.columns(2)
    with col1:
        q_thresh = st.slider(
            "FDR q-value threshold", 0.001, 0.1, float(Q_VALUE_THRESHOLD), 0.001,
            format="%.3f"
        )
    with col2:
        meth_thresh = st.slider(
            "Minimum |methylation difference| (%)", 5.0, 30.0,
            float(METH_DIFF_THRESHOLD), 0.5
        )

    df_filt = filter_dmr_data(df_raw, q_thresh, meth_thresh)
    summary = compute_dmr_summary(df_filt)
    st.session_state["dmr_filt"] = df_filt

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total DMR sites",    f"{summary['total_dmrs']:,}")
    c2.metric("Significant DMRs",   f"{summary['significant']:,}")
    c3.metric("Hypermethylated",     f"{summary['hypermethylated']:,}")
    c4.metric("Hypomethylated",      f"{summary['hypomethylated']:,}")

    # --- Manhattan Plot ---
    st.markdown("### Step 3: Manhattan Plot")
    st.markdown(
        "Significant DMRs (red) exceed both significance and methylation "
        "difference thresholds. Tall peaks on chromosomes 14 and 17 correspond "
        "to PSEN1 and MAPT loci."
    )
    with st.spinner("Rendering Manhattan plot..."):
        fig_m = create_manhattan_plot(df_filt)
    st.plotly_chart(fig_m, use_container_width=True)

    # --- Volcano Plot ---
    st.markdown("### Step 4: Volcano Plot")
    st.markdown(
        "X-axis: methylation difference (%). Y-axis: -log10(q-value). "
        "Dashed lines indicate thresholds. Significant DMRs in red."
    )
    with st.spinner("Rendering Volcano plot..."):
        fig_v = create_volcano_plot(df_filt)
    st.plotly_chart(fig_v, use_container_width=True)

    # --- Genomic Annotation ---
    st.markdown("### Step 5: Genomic Annotation")
    st.markdown(
        "Significant DMRs annotated against the six AD target genes using "
        "hg38 reference coordinates."
    )
    with st.spinner("Annotating DMR regions..."):
        annot_df = annotate_dmr_regions(df_filt)

    if annot_df.empty:
        st.info("No significant DMRs overlapping target gene regions found.")
    else:
        st.dataframe(annot_df, use_container_width=True, hide_index=True)
        csv_bytes = annot_df.to_csv(index=False).encode()
        st.download_button(
            "Download annotation CSV",
            csv_bytes,
            "genomic_annotation.csv",
            "text/csv",
        )

    # Reference table from paper
    st.markdown("**Key findings from Mey et al. (2025):**")
    paper_table = pd.DataFrame({
        "Chr":     ["Chr14", "Chr14", "Chr17"],
        "Start":   [73113602, 73198335, 45889839],
        "End":     [73113602, 73198335, 45889839],
        "P-value": ["~0.0", "~0.0", "8.12e-10"],
        "Q-value": ["~0.0", "~0.0", "1.47e-7"],
        "Meth.diff": [-30.08647, 19.11400, -24.09307],
        "Gene":    ["PSEN1", "PSEN1", "MAPT"],
        "Annotation": ["Promoter (2-3kb)", "Promoter (1-2kb)", "Intron 1 of 6"],
    })
    st.dataframe(paper_table, use_container_width=True, hide_index=True)

    # --- KEGG Enrichment ---
    st.markdown("### Step 6: KEGG Pathway Enrichment")
    st.markdown(
        "KEGG pathway enrichment for the set of annotated target genes. "
        "Pathways hsa05010 and hsa05022 represent Alzheimer's disease pathways."
    )
    if st.button("Run KEGG Enrichment", type="primary"):
        with st.spinner("Querying KEGG REST API and computing enrichment..."):
            gene_ids = [AD_GENE_COORDS[g]["gene_id"] for g in AD_GENE_COORDS]
            kegg_df  = run_kegg_enrichment(gene_ids)
        st.session_state["kegg_df"] = kegg_df

    if "kegg_df" in st.session_state:
        kegg_df = st.session_state["kegg_df"]
        if kegg_df.empty:
            st.warning("No significant KEGG pathways found.")
        else:
            display_cols = [
                "pathway_id", "description", "gene_ratio", "bg_ratio",
                "rich_factor", "fold_enrichment", "z_score",
                "p_value", "p_adjust", "q_value", "count",
            ]
            display_cols = [c for c in display_cols if c in kegg_df.columns]
            st.dataframe(
                kegg_df[display_cols].style.format({
                    "rich_factor":     "{:.6f}",
                    "fold_enrichment": "{:.4f}",
                    "z_score":         "{:.4f}",
                    "p_value":         "{:.6f}",
                    "p_adjust":        "{:.6f}",
                    "q_value":         "{:.6f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("""
**Interpretation (from Mey et al., 2025):**
- hsa05010 (Alzheimer disease): GeneRatio 2/2, FoldEnrichment ~24.03, Z-score ~6.79, p = 0.0017
- hsa05022 (Neurodegeneration pathways): GeneRatio 2/2, FoldEnrichment ~19.45, Z-score ~6.08, p = 0.0026
Both pathways are enriched at FDR q < 0.05, confirming biological relevance of PSEN1 and MAPT.
            """)


# ===========================================================================
# PAGE: AI / MLP Classification
# ===========================================================================

def page_ml():
    st.title("AI / MLP Classification Pipeline")
    st.markdown(
        "Binary classification of Alzheimer's Disease (AD) vs. Normal using "
        "transcriptomic gene expression from GEO datasets GSE48350 + GSE11882."
    )

    # --- Data source ---
    st.markdown("### Step 1: Load Expression Data")
    data_mode = st.radio(
        "Data source",
        [
            "Use built-in demo data (synthetic expression)",
            "Upload pre-processed CSV",
            "Download from GEO (requires internet + GEOparse)",
        ],
        horizontal=False,
    )

    df_expr = None

    if data_mode.startswith("Use built-in"):
        st.info(
            "Demo data generates 100 synthetic samples mimicking the merged "
            "GSE48350 + GSE11882 dataset with 6 gene features, age, sex, "
            "and brain region encodings."
        )
        n_samples = st.slider("Number of synthetic samples", 50, 500, 100, 10)
        if st.button("Generate Demo Expression Data", type="primary"):
            with st.spinner("Generating synthetic expression data..."):
                df_expr = generate_sample_expression_data(n_samples=n_samples)
            st.session_state["expr_df"] = df_expr

    elif data_mode.startswith("Upload"):
        st.markdown(
            "Expected columns: `APP`, `PSEN1`, `PSEN2`, `APOE`, `MAPT`, `TREM2`, "
            "`age`, `sex` (F/M), `region` (HC/EC/SG/PCG), `label` (0=Normal, 1=AD)"
        )
        uploaded = st.file_uploader("Upload expression CSV", type=["csv"])
        if uploaded:
            df_expr = pd.read_csv(uploaded)
            st.session_state["expr_df"] = df_expr
            st.success(f"Loaded {len(df_expr)} samples.")

    else:  # GEO download
        st.warning(
            "This option downloads data directly from NCBI GEO. "
            "Requires `GEOparse` and internet access. Download may take several minutes."
        )
        if st.button("Download GSE48350 + GSE11882 from GEO"):
            try:
                from ml_classification import fetch_geo_data, extract_features_from_geo
                with st.spinner("Downloading from GEO... this may take a few minutes."):
                    datasets = fetch_geo_data(GEO_ACCESSIONS, data_dir="geo_data")
                if datasets:
                    with st.spinner("Extracting features..."):
                        df_expr = extract_features_from_geo(datasets)
                    if not df_expr.empty:
                        st.session_state["expr_df"] = df_expr
                        st.success(f"Extracted {len(df_expr)} samples from GEO.")
                    else:
                        st.error("Feature extraction failed. Check GEO data format.")
                else:
                    st.error("GEO download failed. Check internet connection.")
            except ImportError:
                st.error("GEOparse not installed. Run: pip install GEOparse")

    if df_expr is None and "expr_df" in st.session_state:
        df_expr = st.session_state["expr_df"]

    if df_expr is None:
        st.warning("Please load or generate expression data to proceed.")
        return

    st.success(f"Expression dataset: {len(df_expr)} samples, {df_expr.shape[1]} columns.")

    # Label distribution
    if "label" in df_expr.columns:
        label_counts = df_expr["label"].value_counts()
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Samples",   len(df_expr))
        col_b.metric("Normal (label=0)", int(label_counts.get(0, 0)))
        col_c.metric("AD (label=1)",     int(label_counts.get(1, 0)))

    with st.expander("Preview expression data"):
        st.dataframe(df_expr.head(10), use_container_width=True)

    # --- Model configuration ---
    st.markdown("### Step 2: Model Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_epochs = st.slider("Training epochs", 50, 500, EPOCHS, 50)
    with col2:
        test_frac = st.slider("Test fraction", 0.1, 0.4, 0.20, 0.05)
    with col3:
        backend = st.selectbox(
            "Backend", ["TensorFlow/Keras (default)", "scikit-learn (fallback)"]
        )
    use_keras = "TensorFlow" in backend

    # --- Train ---
    st.markdown("### Step 3: Train MLP Model")
    if st.button("Train Model", type="primary"):
        if "label" not in df_expr.columns:
            st.error("DataFrame must contain a 'label' column (0=Normal, 1=AD).")
            return

        with st.spinner("Preprocessing features..."):
            try:
                X_train, X_test, y_train, y_test, feat_names, scaler = (
                    preprocess_features(df_expr)
                )
                st.session_state["X_test"]     = X_test
                st.session_state["y_test"]     = y_test
                st.session_state["feat_names"] = feat_names
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                return

        st.info(
            f"Train set: {X_train.shape[0]} samples | "
            f"Test set: {X_test.shape[0]} samples | "
            f"Features: {X_train.shape[1]}"
        )

        with st.spinner(f"Training MLP for {n_epochs} epochs..."):
            try:
                if use_keras:
                    model = build_mlp_model(X_train.shape[1])
                    if model is None:
                        st.warning("TensorFlow unavailable. Using scikit-learn MLP.")
                        model = build_sklearn_mlp(X_train.shape[1])
                else:
                    model = build_sklearn_mlp(X_train.shape[1])

                history = train_model(model, X_train, y_train)
                st.session_state["model"]   = model
                st.session_state["history"] = history

            except Exception as e:
                st.error(f"Training failed: {e}")
                return

        st.success("Model training complete.")

    # --- Evaluate ---
    if "model" in st.session_state and "X_test" in st.session_state:
        st.markdown("### Step 4: Model Evaluation")
        with st.spinner("Evaluating on test set..."):
            metrics = evaluate_model(
                st.session_state["model"],
                st.session_state["X_test"],
                st.session_state["y_test"],
            )
        st.session_state["metrics"] = metrics

        # Key metrics display
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy",   f"{metrics['accuracy']:.3f}")
        m2.metric("Precision (AD)", f"{metrics['precision_ad']:.3f}")
        m3.metric("Recall (AD)",    f"{metrics['recall_ad']:.3f}")
        m4.metric("F1 (AD)",        f"{metrics['f1_ad']:.3f}")
        m5.metric("ROC AUC",        f"{metrics['roc_auc']:.3f}")

        # Figures side by side
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown("**Confusion Matrix**")
            fig_cm = create_confusion_matrix_plot(
                metrics["confusion_matrix"],
                title="Machine Learning Confusion Matrix",
            )
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)

        with col_roc:
            st.markdown("**ROC Curve**")
            fig_roc = create_roc_plot(
                metrics["fpr"], metrics["tpr"], metrics["roc_auc"]
            )
            st.pyplot(fig_roc, use_container_width=True)
            plt.close(fig_roc)

        # Training history
        if st.session_state.get("history"):
            st.markdown("**Training History**")
            fig_hist = create_training_history_plot(st.session_state["history"])
            st.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)

        # Classification report
        with st.expander("Full Classification Report"):
            cr   = metrics["classification_report"]
            cr_df = pd.DataFrame(cr).T
            st.dataframe(
                cr_df.style.format("{:.4f}", na_rep="-"),
                use_container_width=True,
            )

        # Feature importance (scikit-learn only)
        if hasattr(st.session_state["model"], "coefs_"):
            feat_weights = np.abs(st.session_state["model"].coefs_[0]).mean(axis=1)
            feat_df = pd.DataFrame({
                "Feature":    st.session_state["feat_names"],
                "Importance": feat_weights,
            }).sort_values("Importance", ascending=False)

            with st.expander("Feature Importance (first layer weights)"):
                st.bar_chart(feat_df.set_index("Feature")["Importance"])

        # Download results
        st.markdown("### Step 5: Download Results")
        cr_df_export = pd.DataFrame(metrics["classification_report"]).T
        st.download_button(
            "Download Classification Report (CSV)",
            cr_df_export.to_csv().encode(),
            "classification_report.csv",
            "text/csv",
        )

        # Prediction table
        pred_df = pd.DataFrame({
            "True Label":  metrics["y_test"],
            "Predicted":   metrics["y_pred"],
            "Probability": [round(p, 4) for p in metrics["y_proba"]],
        })
        pred_df["Correct"] = pred_df["True Label"] == pred_df["Predicted"]
        st.download_button(
            "Download Predictions (CSV)",
            pred_df.to_csv(index=False).encode(),
            "predictions.csv",
            "text/csv",
        )

    st.markdown("---")
    st.markdown("""
**Model details (Mey et al., 2025):**
- Architecture: Dense(64, ReLU) -> Dropout(0.3) -> Dense(32, ReLU) -> Dropout(0.2) -> Dense(1, Sigmoid)
- Optimiser: Adam (lr=0.001), Loss: Binary cross-entropy
- Training split: 80% train / 20% test, Braak 0-II = Normal, Braak III-VI = AD
- Paper results: Accuracy ~88%, specificity 97% (71/73 normal cases), sensitivity 38% (5/13 AD cases)
    """)


# ===========================================================================
# PAGE: Citation & About
# ===========================================================================

def page_about():
    st.title("Citation & About")

    st.markdown("### Primary Reference")
    st.markdown("""
<div class="cite-box">
Mey, T. V. K. Y., Lee, M. B., Tomodok, A. C. A., Muliana, N. E., Priskila, D.,
Sadrawi, M., &amp; Parikesit, A. A. (2025). Detection of Alzheimer's Disease through
AI-Driven and Methylation Difference Region Analysis of Significant Epigenetic
Modifications in APP, PSEN1, PSEN2, APOE, MAPT, and TREM2 Genes. In A. ISYAKU (Ed.),
<em>ADVANCED THERAPEUTICS AND DISEASE BIOLOGY: MOLECULAR DIAGNOSTICS AND IMMUNITY- 2025</em>
(pp. 57–86). Halic Publishing House.
<a href="https://doi.org/10.5281/zenodo.18070507">https://doi.org/10.5281/zenodo.18070507</a>
</div>
    """, unsafe_allow_html=True)

    st.markdown("### Data Sources")
    data_table = pd.DataFrame({
        "Accession":   ["GSE244352", "GSE48350", "GSE11882"],
        "Pipeline":    ["DMR Analysis", "AI/MLP", "AI/MLP"],
        "Type":        [
            "Methylation capture sequencing (peripheral blood)",
            "Brain microarray (postmortem)",
            "Brain microarray (postmortem)",
        ],
        "Samples":     ["12 AD + 12 controls", "Multiple", "Multiple"],
        "Platform":    ["Illumina HiSeq 2500", "GPL570 (HG-U133 Plus 2.0)", "GPL570"],
        "Reference":   [
            "Mitsumori et al., 2025",
            "GEO public dataset",
            "GEO public dataset",
        ],
    })
    st.dataframe(data_table, use_container_width=True, hide_index=True)

    st.markdown("### Application Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Authors:**
- Theshia Veronica Kusuma Yun Mey
- Michael Branson Lee
- Angelo Christiano Aouad Tomodok
- Nathaniel Emmanuel Muliana
- Dhea Priskila
- Muammar Sadrawi
- Prof. Dr. Arli Aditya Parikesit

**Institution:**
Indonesia International Institute for Life Sciences (i3L)
Department of Biomedicine / Biotechnology, Jakarta
        """)
    with col2:
        st.markdown("""
**Application built with:**
- Python 3.9+
- Streamlit
- TensorFlow / Keras
- scikit-learn
- plotly
- matplotlib / seaborn
- pandas / numpy
- GEOparse
- scipy
- requests (KEGG REST API)

**GitHub Repository:**
[github.com/Johnkecops](https://github.com/Johnkecops)

**Contact:**
arli.parikesit@i3l.ac.id
        """)

    st.markdown("### BibTeX Entry")
    bibtex = """@incollection{mey2025alzheimer,
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
}"""
    st.code(bibtex, language="bibtex")

    st.markdown("### License & Reproducibility")
    st.markdown("""
This software is released under the MIT License.
All data used in the original study is publicly available via NCBI GEO.
The application is designed for reproducible research: no credentials or
proprietary data are required to run the default demo pipeline.
    """)

    st.markdown("### Disclaimer")
    st.warning(
        "This application is intended for research and educational purposes only. "
        "It does not constitute medical advice and should not be used for clinical "
        "diagnosis. All computational predictions require experimental validation."
    )


# ===========================================================================
# Router
# ===========================================================================

if page == "Home":
    page_home()
elif page == "DMR Analysis":
    page_dmr()
elif page == "AI / MLP Classification":
    page_ml()
elif page == "Citation & About":
    page_about()
