import deepchem as dc
import pandas as pd
import numpy as np

INPUT_CSV = "BBBP.csv"

df = pd.read_csv(INPUT_CSV)

# BBBP uses:
# smiles = molecule
# p_np   = binary target
smiles = df["smiles"].astype(str).to_numpy()
labels = df["p_np"].to_numpy().reshape(-1, 1)

# DeepChem needs the SMILES strings in dataset.ids
dataset = dc.data.NumpyDataset(
    X=np.zeros((len(df), 1)),
    y=labels,
    ids=smiles,
)

splitter = dc.splits.ScaffoldSplitter()

train_idx, val_idx, test_idx = splitter.split(
    dataset,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
)

print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))

df.iloc[train_idx].to_csv(
    "bbbp_deepchem_refine80.csv",
    index=False,
)

df.iloc[val_idx].to_csv(
    "bbbp_deepchem_val10.csv",
    index=False,
)

df.iloc[test_idx].to_csv(
    "bbbp_deepchem_bench10.csv",
    index=False,
)