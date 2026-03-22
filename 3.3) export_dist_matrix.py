import argparse
import os
import numpy as np
import pandas as pd

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
    p.add_argument("--export-group-dist", action="store_true", help="Also export major-haplogroup median-representative group distance matrix")
    p.add_argument("--haplogroup-column", default="Haplogroup", help="Column name used to derive major haplogroup labels")
    p.add_argument("--no-project-group-medians", action="store_true", help="Do not project coordinate-wise medians back to manifold before distance computation")
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
    df = None
    split_indices = None
    if args.data_path is not None or args.export_group_dist:
        if args.data_path is None:
            raise ValueError("--data-path is required when --export-group-dist is used.")
        split_key = f"{args.split}_indices"
        split_indices = ckpt.get(split_key)
        if split_indices is None:
            raise ValueError(f"Checkpoint missing '{split_key}' for sample ID mapping.")
        df = pd.read_pickle(args.data_path, compression="gzip")
        df = df[df['Quality'] >= 0.9] # Same filtering as in training, to ensure indices match

    if args.data_path is not None:
        if args.label_column not in df.columns:
            raise ValueError(f"Label column '{args.label_column}' not found in dataframe.")
        labels = [str(df.iloc[int(split_indices[i])][args.label_column]) for i in idx]

    _utils.write_phylip_dist_matrix(dist, out_prefix + ".phy", labels=labels)
    print("Saved:", out_prefix + ".npy")
    print("Saved:", out_prefix + ".phy")

    if args.export_group_dist:
        if args.haplogroup_column not in df.columns:
            raise ValueError(f"Haplogroup column '{args.haplogroup_column}' not found in dataframe.")

        selected_row_idx = [int(split_indices[i]) for i in idx]
        haplogroups = df.iloc[selected_row_idx][args.haplogroup_column].astype(str).str.strip().str.upper()
        major_groups = haplogroups.str[0].replace("", np.nan).fillna("Unknown").to_numpy()

        group_dist, group_labels, group_counts = _utils.compute_group_median_dist_matrix(
            z_sub,
            major_groups,
            model.manifold,
            device=args.device,
            project_medians=not args.no_project_group_medians,
        )

        group_prefix = out_prefix + "_major_haplogroup"
        np.save(group_prefix + ".npy", group_dist)
        _utils.write_phylip_dist_matrix(group_dist, group_prefix + ".phy", labels=group_labels)
        pd.DataFrame({
            "major_haplogroup": group_labels,
            "n_samples": group_counts,
        }).to_csv(group_prefix + "_counts.csv", index=False)

        print("Saved:", group_prefix + ".npy")
        print("Saved:", group_prefix + ".phy")
        print("Saved:", group_prefix + "_counts.csv")


if __name__ == "__main__":
    main()
