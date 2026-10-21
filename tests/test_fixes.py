#!/usr/bin/env python3
"""
Module: test_fixes
Purpose: Assert-based regression checks for the 2026-10 code-review fixes.
Author: Dr. Arli Aditya Parikesit
Date: 2026
Usage:
    python tests/test_fixes.py   (offline; KEGG names fall back to built-ins)
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SCRIPT"))

import dmr_analysis as dmr
import ml_classification as ml

dmr._query_kegg_pathway = lambda pw: {"id": pw}  # stay offline


# 1. Demo q-values are true BH
df = dmr.generate_sample_dmr_data()
n = len(df) - 3
assert np.allclose(df["qvalue"][:n], stats.false_discovery_control(df["pvalue"][:n]))

# 2. KEGG reproduces the paper (PSEN1 + MAPT -> hsa05010 FE 24.03, p 0.0017)
k = dmr.run_kegg_enrichment([5663, 4137]).set_index("pathway_id")
assert abs(k.loc["hsa05010", "fold_enrichment"] - 24.03) < 0.01
assert abs(k.loc["hsa05010", "p_value"] - 0.0017) < 0.0001
assert k.loc["hsa05022", "description"].startswith("Pathways of neurodegeneration")

# 3. Strand-aware promoter bins; distant DMR is intergenic
psen1, app = dmr.AD_GENE_COORDS["PSEN1"], dmr.AD_GENE_COORDS["APP"]
assert dmr._classify_region(psen1["start"] - 2500, psen1) == "Promoter (2-3kb)"
assert dmr._classify_region(app["end"] + 1500, app) == "Promoter (1-2kb)"
far = dmr.filter_dmr_data(pd.DataFrame([
    {"chr": "chr14", "start": 20_000_000, "end": 20_000_001,
     "pvalue": 1e-9, "qvalue": 1e-9, "meth_diff": 40.0}]))
a = dmr.annotate_dmr_regions(far)
assert a["gene_symbol"][0] == "Intergenic" and dmr.significant_gene_ids(a) == []

# 4. Uploaded methylKit-style file loads with renamed columns
up = io.StringIO("chr\tstart\tend\tstrand\tp.value\tq.value\tmeth.diff\n14\t1\t2\t*\t0.01\t0.02\t-20\n")
assert {"chr", "pvalue", "qvalue", "meth_diff"} <= set(dmr.load_dmr_data(up).columns)

# 5. GEO characteristics (list of "key: value") and Roman Braak stages parse
class G:
    metadata = {"characteristics_ch1": ["braak stage: III", "gender: female", "age: 81"],
                "title": ["hippocampus_x"]}
    table = pd.DataFrame({"ID_REF": ["203382_s_at"], "VALUE": [8.0]})
geo = ml.extract_features_from_geo({"GSE": type("S", (), {"gsms": {"GSM1": G}})})
assert geo.loc[0, "label"] == 1 and geo.loc[0, "sex"] == "F" and geo.loc[0, "APOE"] == 8.0

# 6. sklearn backend trains, sex/region features are live, epochs/test_size honoured
ex = ml.generate_sample_expression_data()
Xtr, Xte, ytr, yte, names, _ = ml.preprocess_features(ex, test_size=0.4)
assert len(Xte) == 40 and Xtr[:, names.index("sex_F")].std() > 0
m = ml.build_sklearn_mlp(Xtr.shape[1])
ml.train_model(m, Xtr, ytr, epochs=30)
assert m.max_iter == 30
r = ml.evaluate_model(m, Xte, yte)
assert not set(np.round(r["y_proba"], 6)) <= {0.0, 1.0}  # real probabilities

print("all checks passed")
