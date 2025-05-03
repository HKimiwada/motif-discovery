# Ran in VS Code.
# Contains the code to validate the motif candidate extracted by STREME.
# Workflow: Calculate the abundance of each motif and then statistically correlate that abundance with the measured physical properties of the corresponding silk fibers.

###############################################
# Importing Dataset
############################################### 
import pandas as pd

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
print(val_df["sequence"].isna().sum())
print(val_df.head())
print(val_df.shape)
