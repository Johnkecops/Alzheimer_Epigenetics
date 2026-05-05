#!/usr/bin/env python3
"""
Module: AI/MLP Classification Pipeline for Alzheimer's Disease Detection
Purpose: Multilayer Perceptron (MLP) model for binary classification of
         Alzheimer's Disease (AD) vs. Normal using transcriptomic gene
         expression data from GEO datasets GSE48350 and GSE11882.

         Target genes: APP, PSEN1, PSEN2, APOE, MAPT, TREM2
         Brain regions: Hippocampus (HC), Entorhinal Cortex (EC),
                        Superior Frontal Gyrus (SG), Postcentral Gyrus (PCG)
         Classification: Braak stage 0-II = Normal (0), III-VI = AD (1)

Reference:
    Mey, T. V. K. Y., Lee, M. B., Tomodok, A. C. A., Muliana, N. E., Priskila, D.,
    Sadrawi, M., & Parikesit, A. A. (2025). Detection of Alzheimer's Disease through
    AI-Driven and Methylation Difference Region Analysis of Significant Epigenetic
    Modifications in APP, PSEN1, PSEN2, APOE, MAPT, and TREM2 Genes. In
    ADVANCED THERAPEUTICS AND DISEASE BIOLOGY: MOLECULAR DIAGNOSTICS AND
    IMMUNITY- 2025 (pp. 57-86). Halic Publishing House.
    https://doi.org/10.5281/zenodo.18070507

Data Sources:
    GSE48350 - GEO (GPL570, HG-U133 Plus 2.0, postmortem brain microarray)
    GSE11882  - GEO (GPL570, HG-U133 Plus 2.0, postmortem brain microarray)

Model Architecture:
    Input  -> Dense(64, ReLU) -> Dropout(0.3) -> Dense(32, ReLU)
           -> Dropout(0.2) -> Dense(1, Sigmoid)
    Loss   : Binary cross-entropy
    Epochs : 200
    Split  : 80% train / 20% test

Usage:
    python ml_classification.py --fetch-geo          (download from GEO)
    python ml_classification.py --demo               (synthetic data, no download)
    python ml_classification.py --input data.csv     (pre-processed CSV)
"""

import argparse
import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GEO accessions used in the study
GEO_ACCESSIONS = ["GSE48350", "GSE11882"]

# Brain region labels used in metadata
BRAIN_REGIONS = ["hippocampus", "entorhinal cortex",
                 "superior frontal gyrus", "postcentral gyrus"]
BRAIN_REGION_SHORT = {"hippocampus": "HC", "entorhinal cortex": "EC",
                      "superior frontal gyrus": "SG", "postcentral gyrus": "PCG"}

# Braak stage thresholds for binary classification
BRAAK_NORMAL_MAX = 2   # Braak 0-II = Normal
BRAAK_AD_MIN     = 3   # Braak III-VI = AD

# Target AD genes and their Affymetrix HG-U133 Plus 2.0 (GPL570) probe IDs
# Source: Affymetrix annotation, NetAffx
TARGET_GENES = {
    "APP":   ["207317_s_at", "214953_s_at", "207318_s_at"],
    "PSEN1": ["202627_s_at", "202628_s_at"],
    "PSEN2": ["204465_s_at", "204466_s_at"],
    "APOE":  ["203382_s_at"],
    "MAPT":  ["203132_at",  "209173_at"],
    "TREM2": ["220461_s_at"],
}

# MLP hyperparameters matching the paper
EPOCHS      = 200
BATCH_SIZE  = 16
TEST_SIZE   = 0.20
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# 1. Synthetic Demo Data
# ---------------------------------------------------------------------------

def generate_sample_expression_data(
    n_samples: int = 100,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic gene expression data resembling the combined
    GSE48350 + GSE11882 preprocessed output.

    Returns a DataFrame with columns:
        APP, PSEN1, PSEN2, APOE, MAPT, TREM2,
        age, sex_F, sex_M,
        region_HC, region_EC, region_SG, region_PCG,
        label (0=Normal, 1=AD)
    """
    rng = np.random.default_rng(seed)

    n_normal = int(n_samples * 0.73)
    n_ad     = n_samples - n_normal
    labels   = np.array([0] * n_normal + [1] * n_ad)

    # Gene expression: AD samples show altered expression for target genes
    shifts = {"APP": 0.4, "PSEN1": -0.6, "PSEN2": -0.3,
              "APOE": 0.5, "MAPT": 0.8, "TREM2": -0.4}
    gene_data = {}
    for gene, shift in shifts.items():
        base = rng.normal(7.5, 1.2, n_samples)
        base[n_normal:] += shift * rng.normal(1.0, 0.5, n_ad)
        gene_data[gene] = base

    # Demographics
    ages      = rng.integers(55, 95, size=n_samples)
    sex_codes = rng.choice([0, 1], size=n_samples)  # 0=M, 1=F
    sex_F     = (sex_codes == 1).astype(int)
    sex_M     = (sex_codes == 0).astype(int)

    # Brain regions (one-hot)
    region_choices = rng.choice(["HC", "EC", "SG", "PCG"], size=n_samples)
    region_HC  = (region_choices == "HC").astype(int)
    region_EC  = (region_choices == "EC").astype(int)
    region_SG  = (region_choices == "SG").astype(int)
    region_PCG = (region_choices == "PCG").astype(int)

    df = pd.DataFrame({
        **gene_data,
        "age":        ages,
        "sex_F":      sex_F,
        "sex_M":      sex_M,
        "region_HC":  region_HC,
        "region_EC":  region_EC,
        "region_SG":  region_SG,
        "region_PCG": region_PCG,
        "label":      labels,
    })
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. GEO Data Retrieval
# ---------------------------------------------------------------------------

def fetch_geo_data(accessions: list, data_dir: str = "geo_data") -> dict:
    """
    Download and parse GEO datasets using GEOparse.

    Parameters
    ----------
    accessions : List of GEO accession strings (e.g. ['GSE48350', 'GSE11882']).
    data_dir   : Local directory to cache downloaded data.

    Returns
    -------
    dict mapping accession -> GEOparse GSE object.
    """
    try:
        import GEOparse
    except ImportError:
        raise ImportError(
            "GEOparse is required for GEO data download. "
            "Install with: pip install GEOparse"
        )

    os.makedirs(data_dir, exist_ok=True)
    datasets = {}
    for acc in accessions:
        print(f"[INFO] Downloading {acc} from GEO...")
        try:
            gse = GEOparse.get_GEO(
                geo=acc,
                destdir=data_dir,
                silent=True,
                how="full",
            )
            datasets[acc] = gse
            print(f"[INFO] {acc}: {len(gse.gsms)} samples loaded.")
        except Exception as e:
            print(f"[WARN] Failed to download {acc}: {e}")
    return datasets


# ---------------------------------------------------------------------------
# 3. Feature Extraction
# ---------------------------------------------------------------------------

def _extract_braak_stage(title: str, characteristics: dict) -> int:
    """Parse Braak stage from GEO sample metadata."""
    for key in ["braak stage", "braak_stage", "braak nft stage"]:
        val = characteristics.get(key, "")
        if val:
            digits = "".join(c for c in str(val) if c.isdigit())
            if digits:
                return int(digits[0])
    # Try from title
    for s in range(7):
        if f"braak {s}" in title.lower() or f"braak{s}" in title.lower():
            return s
    return -1


def _extract_sex(characteristics: dict) -> str:
    """Parse sex from GEO sample metadata."""
    for key in ["sex", "gender"]:
        val = str(characteristics.get(key, "")).lower()
        if "female" in val or val == "f":
            return "F"
        if "male" in val or val == "m":
            return "M"
    return "Unknown"


def _extract_age(characteristics: dict) -> float:
    """Parse age from GEO sample metadata."""
    for key in ["age", "age at death", "age_at_death"]:
        val = str(characteristics.get(key, ""))
        digits = "".join(c for c in val if c.isdigit() or c == ".")
        if digits:
            try:
                return float(digits)
            except ValueError:
                pass
    return np.nan


def _extract_brain_region(title: str, characteristics: dict) -> str:
    """Parse brain region from GEO sample metadata."""
    text = (title + " " + str(characteristics)).lower()
    if "hippocampus" in text or " hc" in text:
        return "HC"
    if "entorhinal" in text or " ec" in text:
        return "EC"
    if "frontal" in text or " sg" in text or "superior frontal" in text:
        return "SG"
    if "postcentral" in text or " pcg" in text:
        return "PCG"
    return "Unknown"


def extract_features_from_geo(datasets: dict) -> pd.DataFrame:
    """
    Extract gene expression features and metadata from GEOparse objects.

    For each sample:
    - Average expression across Affymetrix probe IDs per target gene
    - Extract Braak stage -> binary label (0-II=Normal, III-VI=AD)
    - Extract age, sex, brain region

    Parameters
    ----------
    datasets : dict of accession -> GEOparse GSE object.

    Returns
    -------
    pd.DataFrame with gene + demographic features and binary label.
    """
    records = []

    for acc, gse in datasets.items():
        # Build probe -> expression lookup per sample
        for gsm_name, gsm in gse.gsms.items():
            try:
                char_flat = {}
                for k, v in gsm.metadata.get("characteristics_ch1", {}).items():
                    if isinstance(v, list):
                        v = v[0] if v else ""
                    parts = str(v).split(":")
                    if len(parts) >= 2:
                        char_flat[parts[0].strip().lower()] = parts[1].strip()
                    else:
                        char_flat[k] = str(v)

                title      = " ".join(gsm.metadata.get("title", [""])).lower()
                braak      = _extract_braak_stage(title, char_flat)
                sex        = _extract_sex(char_flat)
                age        = _extract_age(char_flat)
                region     = _extract_brain_region(title, char_flat)

                if braak < 0:
                    continue  # Skip samples without Braak staging

                label = 0 if braak <= BRAAK_NORMAL_MAX else 1

                # Expression data
                table = gsm.table
                if table is None or table.empty:
                    continue

                # Set probe IDs as index
                if "ID_REF" in table.columns:
                    table = table.set_index("ID_REF")
                elif table.index.name != "ID_REF":
                    table.index = table.index.astype(str)

                row = {
                    "sample_id": gsm_name,
                    "accession": acc,
                    "braak":     braak,
                    "label":     label,
                    "age":       age,
                    "sex":       sex,
                    "region":    region,
                }

                # Average expression per gene
                for gene, probes in TARGET_GENES.items():
                    available = [p for p in probes if p in table.index]
                    if available:
                        vals = []
                        for p in available:
                            v = pd.to_numeric(
                                table.loc[p, "VALUE"]
                                if "VALUE" in table.columns
                                else table.loc[p].iloc[0],
                                errors="coerce",
                            )
                            if not np.isnan(v):
                                vals.append(v)
                        row[gene] = float(np.mean(vals)) if vals else np.nan
                    else:
                        row[gene] = np.nan

                records.append(row)

            except Exception as e:
                print(f"[WARN] Skipping {gsm_name}: {e}")
                continue

    if not records:
        print("[WARN] No valid samples extracted. Check GEO data structure.")
        return pd.DataFrame()

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Preprocessing
# ---------------------------------------------------------------------------

def preprocess_features(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit_scaler: bool = True,
) -> tuple:
    """
    Prepare feature matrix X and label vector y for MLP training.

    Steps:
    1. One-hot encode brain region
    2. Binary encode sex (F=1, M=0)
    3. Impute missing values with column medians
    4. StandardScaler normalisation
    5. 80/20 train/test split

    Parameters
    ----------
    df         : DataFrame with gene + demographic columns and 'label'.
    scaler     : Existing StandardScaler (use during inference); None to fit new one.
    fit_scaler : Whether to fit a new scaler (True for training).

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, fitted_scaler
    """
    required_genes = list(TARGET_GENES.keys())
    available_genes = [g for g in required_genes if g in df.columns]

    # One-hot encode brain region
    if "region" in df.columns:
        region_dummies = pd.get_dummies(df["region"], prefix="region")
        for col in ["region_HC", "region_EC", "region_SG", "region_PCG"]:
            if col not in region_dummies.columns:
                region_dummies[col] = 0
    else:
        region_dummies = pd.DataFrame(
            0, index=df.index,
            columns=["region_HC", "region_EC", "region_SG", "region_PCG"]
        )

    # Binary encode sex
    sex_encoded = pd.DataFrame(index=df.index)
    if "sex" in df.columns:
        sex_encoded["sex_F"] = (df["sex"] == "F").astype(int)
        sex_encoded["sex_M"] = (df["sex"] == "M").astype(int)
    else:
        sex_encoded["sex_F"] = 0
        sex_encoded["sex_M"] = 0

    # Age
    age_col = df["age"].copy() if "age" in df.columns else pd.Series(
        np.nan, index=df.index
    )

    feature_parts = [df[available_genes], age_col.rename("age"),
                     sex_encoded, region_dummies]
    X = pd.concat(feature_parts, axis=1).copy()

    # Impute NaN with column median
    for col in X.columns:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    y = df["label"].values.astype(int)
    feature_names = list(X.columns)

    # Scale
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=y,
    )

    return X_train, X_test, y_train, y_test, feature_names, scaler


# ---------------------------------------------------------------------------
# 5. MLP Model
# ---------------------------------------------------------------------------

def build_mlp_model(input_dim: int):
    """
    Build the Multilayer Perceptron classifier using TensorFlow/Keras.

    Architecture (Mey et al., 2025):
        Input(input_dim) -> Dense(64, relu) -> Dropout(0.3)
                         -> Dense(32, relu)  -> Dropout(0.2)
                         -> Dense(1, sigmoid)

    Parameters
    ----------
    input_dim : Number of input features.

    Returns
    -------
    Compiled Keras Sequential model.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        tf.random.set_seed(RANDOM_SEED)

        model = Sequential([
            Dense(64, activation="relu",    input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1,  activation="sigmoid"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    except ImportError:
        print("[WARN] TensorFlow not available. Falling back to scikit-learn MLP.")
        return None


def build_sklearn_mlp(input_dim: int):
    """Fallback MLP using scikit-learn MLPClassifier."""
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=EPOCHS,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
    )


def train_model(model, X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Train the MLP model for EPOCHS iterations.

    Parameters
    ----------
    model   : Keras model or sklearn MLPClassifier.
    X_train : Training features.
    y_train : Training labels.

    Returns
    -------
    dict with training history (loss, accuracy per epoch).
    """
    history_data = {}
    try:
        # Keras model
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=0,
        )
        history_data = {
            "loss":          history.history.get("loss", []),
            "val_loss":      history.history.get("val_loss", []),
            "accuracy":      history.history.get("accuracy", []),
            "val_accuracy":  history.history.get("val_accuracy", []),
        }
    except AttributeError:
        # sklearn fallback
        model.fit(X_train, y_train)
        history_data = {"loss": [], "accuracy": []}

    return history_data


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the trained model on the held-out test set.

    Parameters
    ----------
    model  : Trained Keras or sklearn model.
    X_test : Test features (scaled).
    y_test : Ground truth labels.

    Returns
    -------
    dict with confusion_matrix, classification_report, accuracy,
         precision, recall, f1, roc_auc, y_pred, y_proba.
    """
    try:
        y_proba = model.predict(X_test).flatten()
    except Exception:
        y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_proba >= 0.5).astype(int)

    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred,
                                 target_names=["Normal", "AD"],
                                 output_dict=True)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)

    return {
        "confusion_matrix":      cm.tolist(),
        "classification_report": cr,
        "accuracy":              cr["accuracy"],
        "precision_normal":      cr["Normal"]["precision"],
        "recall_normal":         cr["Normal"]["recall"],
        "f1_normal":             cr["Normal"]["f1-score"],
        "precision_ad":          cr["AD"]["precision"],
        "recall_ad":             cr["AD"]["recall"],
        "f1_ad":                 cr["AD"]["f1-score"],
        "roc_auc":               roc_auc_val,
        "y_pred":                y_pred.tolist(),
        "y_proba":               y_proba.tolist(),
        "y_test":                y_test.tolist(),
        "fpr":                   fpr.tolist(),
        "tpr":                   tpr.tolist(),
    }


# ---------------------------------------------------------------------------
# 6. Visualisation
# ---------------------------------------------------------------------------

def create_confusion_matrix_plot(
    cm: list,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Generate a matplotlib confusion matrix heatmap.

    Parameters
    ----------
    cm    : 2x2 confusion matrix as list of lists.
    title : Plot title.

    Returns
    -------
    matplotlib Figure.
    """
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "AD"],
        yticklabels=["Normal", "AD"],
        ax=ax, cbar=False,
        linewidths=0.5, linecolor="grey",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(title,              fontsize=13)
    fig.tight_layout()
    return fig


def create_roc_plot(fpr: list, tpr: list, roc_auc: float) -> plt.Figure:
    """
    Generate a ROC curve plot.

    Parameters
    ----------
    fpr     : False positive rates.
    tpr     : True positive rates.
    roc_auc : AUC value.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve - AD Classification", fontsize=13)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def create_training_history_plot(history: dict) -> plt.Figure:
    """
    Plot training and validation loss / accuracy curves.

    Parameters
    ----------
    history : dict with loss, val_loss, accuracy, val_accuracy lists.

    Returns
    -------
    matplotlib Figure.
    """
    if not history.get("loss"):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No training history available",
                ha="center", va="center")
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    epochs = range(1, len(history["loss"]) + 1)
    ax1.plot(epochs, history["loss"],     label="Train loss")
    ax1.plot(epochs, history.get("val_loss", []), label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.plot(epochs, history["accuracy"],     label="Train acc")
    ax2.plot(epochs, history.get("val_accuracy", []), label="Val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Full Pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    data_source: str = "demo",
    input_file: str  = None,
    output_dir: str  = "ml_results",
    fetch_geo: bool  = False,
) -> dict:
    """
    Execute the complete ML classification pipeline end-to-end.

    Parameters
    ----------
    data_source : 'demo' | 'geo' | 'file'
    input_file  : Path to pre-processed CSV when data_source='file'.
    output_dir  : Directory to write output files.
    fetch_geo   : Whether to download GEO data online.

    Returns
    -------
    dict with evaluation metrics and paths to saved outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    if data_source == "demo":
        print("[INFO] Using synthetic demo data.")
        df = generate_sample_expression_data()
    elif data_source == "geo" and fetch_geo:
        datasets = fetch_geo_data(GEO_ACCESSIONS)
        if not datasets:
            print("[WARN] GEO download failed. Falling back to demo data.")
            df = generate_sample_expression_data()
        else:
            df = extract_features_from_geo(datasets)
            if df.empty:
                print("[WARN] Feature extraction failed. Falling back to demo data.")
                df = generate_sample_expression_data()
    elif data_source == "file" and input_file:
        df = pd.read_csv(input_file)
    else:
        print("[INFO] No data source specified. Using synthetic demo data.")
        df = generate_sample_expression_data()

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts().to_string()}")

    # 2. Preprocess
    X_train, X_test, y_train, y_test, feat_names, scaler = preprocess_features(df)
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    # Save feature names
    pd.Series(feat_names).to_csv(
        os.path.join(output_dir, "feature_names.csv"), index=False, header=False
    )

    # 3. Build model
    model = build_mlp_model(X_train.shape[1])
    use_keras = model is not None
    if not use_keras:
        model = build_sklearn_mlp(X_train.shape[1])

    # 4. Train
    print(f"[INFO] Training MLP for {EPOCHS} epochs...")
    history = train_model(model, X_train, y_train)

    # 5. Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    print(f"[INFO] Accuracy: {metrics['accuracy']:.4f}")
    print(f"[INFO] ROC AUC:  {metrics['roc_auc']:.4f}")

    report_df = pd.DataFrame(metrics["classification_report"]).T
    print(report_df.to_string())

    # 6. Save metrics
    metrics_clean = {
        k: v for k, v in metrics.items()
        if k not in ("y_pred", "y_proba", "y_test", "fpr", "tpr",
                     "classification_report", "confusion_matrix")
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_clean, f, indent=2)

    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))

    # 7. Figures
    cm_fig = create_confusion_matrix_plot(metrics["confusion_matrix"])
    cm_fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(cm_fig)

    roc_fig = create_roc_plot(metrics["fpr"], metrics["tpr"], metrics["roc_auc"])
    roc_fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close(roc_fig)

    if history:
        hist_fig = create_training_history_plot(history)
        hist_fig.savefig(os.path.join(output_dir, "training_history.png"), dpi=150)
        plt.close(hist_fig)

    # Save model (Keras only)
    if use_keras:
        try:
            model.save(os.path.join(output_dir, "mlp_model.keras"))
            print(f"[INFO] Model saved to {output_dir}/mlp_model.keras")
        except Exception:
            pass

    print(f"[INFO] All outputs saved to: {output_dir}/")
    return {
        "metrics": metrics_clean,
        "output_dir": output_dir,
        "n_samples": len(df),
        "n_features": len(feat_names),
    }


# ---------------------------------------------------------------------------
# 8. CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI/MLP Classification Pipeline for Alzheimer's Disease"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic demonstration data"
    )
    group.add_argument(
        "--fetch-geo", action="store_true",
        help="Download GSE48350 and GSE11882 from GEO (requires internet)"
    )
    group.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to pre-processed expression CSV file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="ml_results",
        help="Output directory for results (default: ml_results/)"
    )
    args = parser.parse_args()

    if args.fetch_geo:
        run_full_pipeline(data_source="geo", fetch_geo=True,
                          output_dir=args.output)
    elif args.input:
        run_full_pipeline(data_source="file", input_file=args.input,
                          output_dir=args.output)
    else:
        run_full_pipeline(data_source="demo", output_dir=args.output)


if __name__ == "__main__":
    main()
