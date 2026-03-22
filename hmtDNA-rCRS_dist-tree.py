#%%
import argparse
import os
import numpy as np
import pandas as pd
from sklearn import tree
import torch
from scipy.spatial.distance import squareform
from skbio.stats.distance import DistanceMatrix
from skbio.tree import TreeNode, nj
from skbio.io import write
from Bio import Phylo
from io import StringIO
import copy

from _models import RGD
import _utils

N_TREE_LATENTS = 5000
SPLIT = "test"

# Read in rcrs data
thr = 0.9
df = pd.read_pickle("data/hmtDNA/rcrs.pkl", compression='gzip')
df = df[df['Quality'] >= thr]

# Load model and get latent representations for test set
checkpoint_path = "models/rgd_std0.5_02-12-17:43/model.pt"
model, ckpt = _utils.load_rgd_checkpoint(checkpoint_path, model_cls=RGD, device="cpu")
z_map = {"train": getattr(model, "z", None),
             "val": getattr(model, "z_val", None),
             "test": getattr(model, "z_test", None)}
z = z_map.get(SPLIT)
assert z is not None

# Get test indices for mapping back to sample IDs
split_indices = ckpt.get(f"{SPLIT}_indices")
assert split_indices is not None
if isinstance(split_indices, torch.Tensor):
    split_indices = split_indices.detach().cpu().numpy()
else:
    split_indices = np.asarray(split_indices)
split_indices = split_indices.astype(np.int64)

# Sample subset of latents and compute pairwise manifold distance matrix
z_sub, idx = _utils.sample_subset(z, n_sample=N_TREE_LATENTS)
dists = _utils.pairwise_manifold_distance_matrix(model.manifold, z_sub).detach().cpu().numpy()
assert dists.shape == (len(z_sub), len(z_sub)) and idx.shape[0] == len(z_sub)
dists = _utils.check_distance_matrix_properties(dists)

# NJ tree construction
dist_mat = DistanceMatrix(dists, ids=df['SampleID'].iloc[split_indices[idx]]) # Use major haplogroup (first letter) as label for tree construction
nj_tree = nj(dist_mat) # skbio TreeNode
newick_str = str(nj_tree) # Newick text
tree_obj = Phylo.read(StringIO(newick_str), "newick")  # Bio.Phylo tree

tip_dist = nj_tree.tip_tip_distances()
print(((tip_dist.data - dist_mat.data)**2).mean())

import sys
sys.setrecursionlimit(10000) 
tree_obj_rescale = copy.deepcopy(tree_obj)
for clade in tree_obj_rescale.find_clades():
    if clade.branch_length is not None:
        clade.branch_length /= (N_TREE_LATENTS - 2) # rescale by N-2


# %%
# Get tip names and map to major haplogroup (first letter) for coloring in tree visualization
tip_names = [tip.name for tip in tree_obj.get_terminals() if tip.name is not None]

hap_by_id = (
    df.dropna(subset=["SampleID"])
      .drop_duplicates(subset=["SampleID"])
      .assign(
          major_hap=lambda x: x["Haplogroup"].astype("string").str.slice(0, 1).fillna("Unknown")
      )
      .set_index("SampleID")["major_hap"]
)

tip_major = hap_by_id.reindex(tip_names).fillna("Unknown").to_dict()

#%%
_utils.plot_circular_tree(tree=tree_obj, tip_major=tip_major, out_path="hmtDNA_rCRS_tree.png")
_utils.plot_circular_tree(tree=tree_obj_rescale, tip_major=tip_major, out_path="hmtDNA_rCRS_tree_rescale.png")
# %%
# save distances and tree
np.save("hmtDNA_rCRS_distances.npy", dists) # .npy with no labels
dist_mat.write("hmtDNA_rCRS_distances.phy", format="phylip_dm") # .phy with labels
np.save("hmtDNA_rCRS_idx.npy", df.index[split_indices[idx]])
nj_tree.write("hmtDNA_rCRS_tree.nwk")
print("Done!")

# %%
# from skbio import read
# from skbio.tree import TreeNode
# # Read both trees
# tree_skbio = read("hmtDNA_rCRS_tree.nwk", format="newick", into=TreeNode)
# tree_dt = read("hmtDNA_rCRS_decenttree.nwk", format="newick", into=TreeNode)

# # Compare topology
# topo_same = tree_skbio.compare_rfd(tree_dt)
# print("Trees have the same topology?", topo_same)
# %%
