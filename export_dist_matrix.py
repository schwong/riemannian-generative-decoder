import argparse
import os
import numpy as np
import pandas as pd
import torch

from _models import RGD
import _utils


def parse_args():
    p = argparse.ArgumentParser(description="Export manifold distance matrix and PHYLIP file.")
    p.add_argument("--save-dir", required=True, help="Run directory containing model.pt")
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint path (default: <save-dir>/model.pt)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which z_* to use for distance matrix")
    p.add_argument("--n-sample", type=int, default=5000, help="Number of samples to use (default: 5000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--device", default="cpu", help="Device, e.g. cpu or cuda:0")
    p.add_argument("--out-prefix", default=None, help="Output prefix (default: <save-dir>/dist_matrix_<split>_<n>)")
    p.add_argument("--data-path", default=None, help="Optional path to rcrs.pkl for sample IDs")
    p.add_argument("--label-column", default="SampleID", help="Column name to use for labels when data-path is set")
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = args.save_dir
    checkpoint_path = args.checkpoint or os.path.join(save_dir, "model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, ckpt = _utils.load_rgd_checkpoint(checkpoint_path, model_cls=RGD, device=args.device)
    z_map = {"train": getattr(model, "z", None),
             "val": getattr(model, "z_val", None),
             "test": getattr(model, "z_test", None)}
    z = z_map.get(args.split)
    if z is None:
        raise ValueError(f"Model is missing z for split '{args.split}'.")

    z_sub, idx = _utils.sample_subset(z, n_sample=args.n_sample, seed=args.seed)
    dist = _utils.compute_dist_matrix(z_sub, model.manifold, device=args.device)

    if args.out_prefix is None:
        out_prefix = os.path.join(save_dir, f"dist_matrix_{args.split}_{len(z_sub)}")
    else:
        out_prefix = args.out_prefix

    np.save(out_prefix + ".npy", dist)
    labels = None
    if args.data_path is not None:
        split_key = f"{args.split}_indices"
        split_indices = ckpt.get(split_key)
        if split_indices is None:
            raise ValueError(f"Checkpoint missing '{split_key}' for sample ID mapping.")
        df = pd.read_pickle(args.data_path, compression="gzip")
        if args.label_column not in df.columns:
            raise ValueError(f"Label column '{args.label_column}' not found in dataframe.")
        labels = [str(df.iloc[int(split_indices[i])][args.label_column]) for i in idx]
    _utils.write_phylip_dist_matrix(dist, out_prefix + ".phy", labels=labels)
    print("Saved:", out_prefix + ".npy")
    print("Saved:", out_prefix + ".phy")


if __name__ == "__main__":
    main()
