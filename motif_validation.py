# Ran in VS Code.
# Contains the code to validate the motif candidate extracted by STREME.
# Workflow: Calculate the abundance of each motif and then statistically correlate that abundance with the measured physical properties of the corresponding silk fibers.

import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

###############################################
# Importing Dataset
############################################### 
seq_dict = {}
with open("Data/spider-silkome-database.v1.prot.fasta") as f:
    current_id = None
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            # header example: ">123|7047|Oecobiidae|…"
            current_id = int(line[1:].split("|", 1)[0])
            seq_dict[current_id] = ""
        else:
            seq_dict[current_id] += line

val_df = pd.read_csv("Data/validation_dataset.csv")

# 2) make sure idv_id is integer
val_df["idv_id"] = val_df["idv_id"].astype(int)

# 3) merge in the spidroin sequence
val_df["sequence"] = val_df["idv_id"].map(seq_dict)

# 4) check for missing sequences
# print(val_df["sequence"].isna().sum())
# print(val_df.shape)
print(val_df.head())

###############################################
# Motif Counts 
############################################### 
with open('output_may1_streme_cluster_0/motif_candidates_export.txt', 'r') as f:
    motif_candidates = f.read().splitlines()

# print(type(motif_candidates))
# print(len(motif_candidates))
# print(motif_candidates)

# count overlapping matches
for m in motif_candidates:
    val_df[f"count_{m}"] = val_df["sequence"].apply(lambda s: len(re.findall(f"(?={m})", s)))

# normalize for sequence length
val_df["seq_length"] = val_df["sequence"].str.len()
for m in motif_candidates:
    val_df[f"freq_{m}"] = val_df[f"count_{m}"] / val_df["seq_length"]

print(val_df.head())

###############################################
# Compute Correlation
############################################### 
"""
Magnitude interpretation (same for both Pearson_r and Spearman_rho):
|r| > 0.50: strong linear correlation
0.30 ≤ |r| ≤ 0.49: moderate linear correlation
|r| < 0.29: weak or negligible linear correlation

Validate monotonic vs. linear: If Spearman’s ρ is much higher than Pearson’s r, consider fitting a non-linear model or rank-based analysis
"""

properties = ["toughness", "toughness_sd", "young's_modulus", "young's_modulus_sd", "tensile_strength", "tensile_strength_sd", "strain_at_break", "strain_at_break_sd"]
results = []
for m in motif_candidates:
    for p in properties:
        x, y = val_df[f"count_{m}"], val_df[p]
        mask = x.notna() & y.notna()
        r, pval = pearsonr(x[mask], y[mask]) if mask.sum()>2 else (None,None)
        rho, sval = spearmanr(x[mask], y[mask]) if mask.sum()>2 else (None,None)
        results.append((m, p, r, pval, rho, sval))

corr_df = pd.DataFrame(results, columns=["motif","property","pearson_r","p","spearman_rho","p_s"])
# FDR correction on Pearson p-values
reject, p_adj, _, _ = multipletests(corr_df["p"], method="fdr_bh")
corr_df["p_adj"] = p_adj
corr_df["significant"] = reject
print(corr_df.sort_values("pearson_r", ascending=False))

###############################################
# Visualize Correlation
############################################### 