# Repo for Motif Discovery
Embeddings from finetuned ESM2 are generated using DGX-1 (8x V100 GPUs) \n
This repo takes those embeddings, clusters residue embeddings, preprocess each residue into 15-mer windows, then uses STREME to locate potential motifs. \n Motif validation is done by calculating the abundance of each motif and then statistically correlate that abundance with the measured physical properties of the corresponding silk fibers.\n
