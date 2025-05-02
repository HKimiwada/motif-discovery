# This code was written for a Jupyter Notebook and is intended to be run in a Google Colab environment.
# It uses the RAPIDS cuML library for GPU-accelerated machine learning and clustering.
# The code is designed to load protein sequence embeddings, perform clustering using KMeans,
# and visualize the results using UMAP. It also saves the clustered embeddings to a CSV file.
# The code assumes that the necessary libraries are installed and that the input data is available in the specified paths.

###############################################
# Importing Embeddings 
###############################################
import os
import pickle

import cudf                                               # cuDF: GPU DataFrame API :contentReference[oaicite:1]{index=1}
import numpy as np

# cuML algorithms, API mirrors scikit-learn
from cuml.cluster import KMeans as cuKMeans              # GPU-accelerated KMeans :contentReference[oaicite:2]{index=2}
from cuml.decomposition import PCA as cuPCA               # GPU PCA
from cuml.manifold import UMAP as cuUMAP                  # GPU UMAP

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

# Load precomputed embeddings dict
emb_path = "/content/drive/MyDrive/研究関連/大学/02_Data/01_embeddings.pkl"
with open(emb_path, 'rb') as f:
    loaded_embeddings = pickle.load(f)                   # load pickled dict :contentReference[oaicite:4]{index=4}

# Map sequence labels to raw strings
sequences = {
    f"{row['protein']}_{row['id']}": row['sequence']
    for row in filtered_df.to_dicts()
}

###############################################
# Clustering Embeddings
###############################################
# Prepare GPU DataFrame columns
labels, poses = [], []
# We'll accumulate embedding vectors into a NumPy array first
emb_list = []

for label, emb in loaded_embeddings.items():
    seq = sequences[label]
    L = len(seq)
    # emb is a torch.Tensor of shape (max_len+2, D)
    true_emb = emb[1:L+1].cpu().numpy()                  # slice off BOS/EOS :contentReference[oaicite:5]{index=5}
    for pos in range(L):
        labels.append(label)
        poses.append(pos)
        emb_list.append(true_emb[pos])                   # each is a (640,) array

emb_arr = np.vstack(emb_list)                           # shape: (n_tokens, 640) :contentReference[oaicite:6]{index=6}

# Create cuDF DataFrame with one column per embedding dimension
gdf = cudf.DataFrame({"label": labels, "pos": poses})
for i in range(emb_arr.shape[1]):
    gdf[f"f{i}"] = emb_arr[:, i]                         # add feature columns :contentReference[oaicite:7]{index=7}

feat_cols = [f"f{i}" for i in range(emb_arr.shape[1])]
gdf_feats = gdf[feat_cols]

###############################################
# Visualize Clusters
###############################################
# Fit KMeans on GPU
cu_kmeans = cuKMeans(n_clusters=20, random_state=0)
gdf["cluster"] = cu_kmeans.fit_predict(gdf_feats)       # GPU clustering :contentReference[oaicite:8]{index=8}

cu_pca = cuPCA(n_components=50, random_state=0)
gdf_pca = cu_pca.fit_transform(gdf_feats)                # GPU PCA

# 1) Fit UMAP on 50‑dim GPU PCA result
from cuml.manifold import UMAP as cuUMAP
import cudf

cu_umap = cuUMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)
gdf_umap = cu_umap.fit_transform(gdf_pca)  # cudf.DataFrame with columns [0, 1] :contentReference[oaicite:3]{index=3}

# 2) Assign coordinates:
gdf["x"] = gdf_umap.iloc[:, 0]
gdf["y"] = gdf_umap.iloc[:, 1]

from cuml.manifold import UMAP as cuUMAP
import cudf
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1) Fit GPU‑UMAP with tuned parameters ---
# Increase n_neighbors for more global context,
# raise min_dist & spread to force points apart,
# use cosine metric for high‑dim embedding spaces,
# spectral init for better cluster layouts.
cu_umap = cuUMAP(
    n_neighbors=30,   # e.g. 30 or 50 for broader neighborhood
    min_dist=0.3,          # larger than default to avoid clumping
    spread=2.0,            # scale factor to push clusters outward
    metric="cosine",       # often better for protein embeddings
    init="spectral",       # smarter initialization than random
    random_state=0
)


gdf_umap = cu_umap.fit_transform(gdf_pca)  # shape: (n_tokens, 2)

# --- 2) Extract the two UMAP dimensions robustly ---
# cuDF will name the columns 0 and 1 (integers), so use .iloc
gdf["x"] = gdf_umap.iloc[:, 0]
gdf["y"] = gdf_umap.iloc[:, 1]

# --- 3) Plot a representative subsample ---
frac = min(200_000 / len(gdf), 1.0)
pdf = gdf.sample(frac=frac).to_pandas()   # bring a manageable subset to CPU

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pdf,
    x="x", y="y",
    hue="cluster",
    palette="tab20",
    s=5, linewidth=0, alpha=0.7
)
plt.title(
    "GPU‑UMAP (n_neighbors=30, min_dist=0.3, spread=2.0, metric=cosine)"
)
plt.legend(bbox_to_anchor=(1,1), title="Cluster", markerscale=2)
plt.tight_layout()
plt.show()

# Convert gdf to pandas DataFrame
pdf = gdf.to_pandas()

###############################################
# Saving Clusters
###############################################
file_path = '/content/drive/MyDrive/研究関連/大学/02_Data/clustered_embeddings.csv'  # Adjust the path as needed

df_pl = pl.from_pandas(pdf)
df_pl.write_csv(
    file_path,
    separator=',',
    batch_size=1_000_000,   # larger chunks → fewer thread launches
    include_header=True,
    quote_char='"',         # default
    line_terminator='\n'    # default
)