# Motif Discovery Pipeline for Spider Silk Proteins

A computational framework for identifying and validating sequence motifs in spider silk proteins using fine‑tuned ESM-2 embeddings and statistical correlation with mechanical properties.

---

## Overview

This repository implements a fully automated pipeline for:

1. Generating residue embeddings with a fine‑tuned ESM-2 model.
2. Clustering embeddings to group similar residue contexts. (v0: K-means, v1: HDBSCAN)
3. Extracting fixed-length (15‑mer) windows around each residue cluster.
4. Running the MEME Suite’s **STREME** algorithm on clusters to discover candidate motifs.
5. Validating discovered motifs by counting their abundance in a validation dataset and statistically correlating motif counts with measured mechanical properties of spider silk fibers.

The pipeline is designed to run on a DGX-1 (8× V100 GPUs) for embedding generation, with the remainder executed in a Python/Conda environment.

## Features

* **ESM-2 Embedding Integration**: Leverage state‑of‑the‑art protein language model embeddings.
* **Unsupervised Residue Clustering**: Group residue positions based on embedding similarity.
* **Window Preprocessing**: Extract local sequence contexts (15‑mers) for motif search.
* **STREME Motif Discovery**: Identify enriched sequence patterns within clusters.
* **Statistical Validation**: Correlate motif occurrences with fiber mechanics using Pearson/Spearman tests and FDR correction.
* **Modular & Configurable**: Easy to adjust clustering parameters, window size, STREME thresholds, and validation properties.
