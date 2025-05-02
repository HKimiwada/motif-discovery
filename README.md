# Repo for Motif Discovery
Embeddings from finetuned ESM2 are generated using DGX-1 (8x V100 GPUs)
This repo takes those embeddings, clusters residue embeddings, preprocess each residue into 15-mer windows, then uses STREME to locate potential motifs.
Validation of motifs are also done on DGX-1.