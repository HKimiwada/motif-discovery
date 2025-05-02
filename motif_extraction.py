# Used STREME in terminal for motif extraction on clustered data (FASTA file)

###############################################
# Terminal commands for motif extraction using STREME
###############################################

"""
conda create -n memesuite python=3.10
conda activate memesuite
conda config --prepend channels conda-forge
conda config --append channels bioconda
conda install meme
streme --version (Verify STREME installation)

streme \
  --protein \
  --p Data/cluster_0.fasta \
  --minw 3 --maxw 15 \
  --thresh 0.05 \
  -oc output_may1_streme_cluster0_out

--protein
Treats your input as amino acids (20-letter alphabet).

--p cluster_0.fasta
Primary FASTA of your 15-mers.

--minw 3 --maxw 8
Search for motifs of width 3–8 residues (you can adjust these; default is 3–15).

--thresh 0.05
Report motifs with p-value < 0.05 (default).

-oc streme_cluster0_out
Write all results into directory streme_cluster0_out/.
"""

###############################################
# Code for motif EDA
###############################################

