# Ran in VS Code.
# Contains the code to validate the motif candidate extracted by STREME.
# Workflow: Calculate the abundance of each motif and then statistically correlate that abundance with the measured physical properties of the corresponding silk fibers.

###############################################
# Importing Dataset
############################################### 
import pandas as pd
test = pd.read_csv("Data/validation_dataset.csv")
test