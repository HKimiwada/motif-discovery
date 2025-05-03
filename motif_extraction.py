# Ran in VS Code.
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

open output_may1_streme_cluster_0/streme.html -> Run in terminal to show STREME analysis results in browser.

Reference:
.venv(base) hikimiwada@Hirokis-MacBook-Air motif-discovery % git ls-files (code to show all tracked files)
.DS_Store
Data/cluster_0.fasta
README.md
embedding_clustering.py
motif_extraction.py
motif_extraction_preprocessing.py
output_may1_streme_cluster_0/sequences.tsv
output_may1_streme_cluster_0/sites.tsv
output_may1_streme_cluster_0/streme.html
output_may1_streme_cluster_0/streme.txt
output_may1_streme_cluster_0/streme.xml

streme.html
An interactive HTML report summarizing discovered motifs, their logos, p-values, E-values, and occurrence counts 

streme.txt
The motifs in MEME motif format (position-weight matrices and consensus) for downstream tools like FIMO or Tomtom 

sites.tsv
A table of every match (site) in your input sequences that exceeds the motif’s match threshold: one line per site, with sequence name, position, score, and strand 

sequences.tsv
A summary of which sequences contain at least one match for each motif, distinguishing true positives (in your data) from false positives (in shuffled background) 

streme.xml
An XML version of all results for programmatic parsing
"""
###############################################
# Extracting motif candidates
###############################################
import re

infile  = 'output_may1_streme_cluster_0/streme.txt'
outfile = 'output_may1_streme_cluster_0/motif_candidates_export.txt'

consensi = []
with open(infile) as fh:
    for line in fh:
        if line.startswith('MOTIF '):
            # line looks like: "MOTIF 1-VSAL STREME-1"
            rank_dash_consensus = line.split()[1]        # e.g. "1-VSAL"
            consensus = rank_dash_consensus.split('-',1)[1]
            consensi.append(consensus)

# keep only unique, in discovery order
seen = set()
unique = [c for c in consensi if not (c in seen or seen.add(c))]

with open(outfile, 'w') as out:
    for c in unique:
        out.write(c + '\n')

print(f"Wrote {len(unique)} motif candidates to {outfile}")



