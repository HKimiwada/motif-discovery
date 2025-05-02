# This code was written for a Jupyter Notebook and is intended to be run in a Google Colab environment.

###############################################
# Importing Data
###############################################
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os

def read_fasta(filename):
    """Reads a FASTA file and returns a dictionary of sequences."""
    sequences = {}
    with open(filename, 'r') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences[record.id] = str(record.seq)
    return sequences

import os
import pickle
import numpy as np

import polars as pl
from Bio import SeqIO

# Load & filter sequences (as before)
seq_file = "/content/drive/MyDrive/研究関連/大学/02_Data/spider-silkome-database.v1.prot.fasta"
records = list(SeqIO.parse(seq_file, "fasta"))           # parse FASTA :contentReference[oaicite:3]{index=3}
parsed = []
for rec in records:
    parts = rec.description.split("|")
    parsed.append({
        "protein": parts[6] if len(parts) > 6 else None,
        "sequence": str(rec.seq)
    })
proteome_df = pl.DataFrame(parsed)

target = ["MaSp3B","MaSp1","MaSp2","MaSp3","MaSp2B","Ampullate spidroin","MiSp","MaSp"]
filtered_df = (
    proteome_df
    .filter(pl.col("protein").is_in(target))
    .select(["protein","sequence"])
    .with_row_index("id")
)

# Map sequence labels to raw strings
sequences = {
    f"{row['protein']}_{row['id']}": row['sequence']
    for row in filtered_df.to_dicts()
}

file_path = "/content/drive/MyDrive/研究関連/大学/02_Data/clustered_embeddings.csv"
embeddings_df = pl.read_csv_batched(file_path)

all_batches = []
for _ in range(100):  
    batch = embeddings_df.next_batches(1)
    if batch:  # Check if a batch was returned
        all_batches.append(batch[0]) 
    else:
        break  # Stop if there are no more batches

sample_embeddings_df = pl.concat(all_batches)

###############################################
# Extracting Seq Windows across each Residue 
###############################################
sample_embeddings_df = sample_embeddings_df.with_columns(
    pl.col("label")
    .str.split_exact("_", 1)  # Split at the first underscore, results in 2 fields
    .struct.rename_fields(["protein", "id"])
    .alias("fields")
).unnest("fields")

# Optionally, convert 'id' to integer
sample_embeddings_df = sample_embeddings_df.with_columns(
    pl.col("id").cast(pl.Int64)
)

embeddings_analysis_df = sample_embeddings_df.join(
    filtered_df,
    on=["protein","id"],
    how="left"
)

window_size = 15
half_w = window_size // 2

embeddings_analysis_df = embeddings_analysis_df.with_columns([
    pl.when(
        (pl.col("pos") >= half_w) &
        (pl.col("pos") + half_w + 1 <= pl.col("sequence").str.len_chars())
    )
    .then(
        pl.col("sequence").str.slice(pl.col("pos") - half_w, window_size)
    )
    .otherwise(None)
    .alias("window")
]).filter(pl.col("window").is_not_null())

###############################################
# Seq windows to per-cluster FASTA files
###############################################
# 1.1 Select minimal columns and convert to pandas
to_export = embeddings_analysis_df.select([
    "cluster", "label", "pos", "window"
]).to_pandas()

# 1.2 Create output directory
out_dir = "cluster_windows_15mers"
os.makedirs(out_dir, exist_ok=True)

# 1.3 Iterate per cluster and write FASTA
for cl_id, grp in to_export.groupby("cluster"):
    fasta_path = os.path.join(out_dir, f"cluster_{cl_id}.fasta")
    records = []
    for _, row in grp.iterrows():
        # Construct a SeqRecord: header includes label & position
        header = f"{row['label']}_pos{row['pos']}_cl{cl_id}"
        rec = SeqRecord(Seq(row["window"]),
                        id=header,
                        description="")
        records.append(rec)
    # Write FASTA
    SeqIO.write(records, fasta_path, "fasta")
    print(f"Wrote {len(records)} windows to {fasta_path}")