from matplotlib import colors
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from tqdm import trange, tqdm
import geoopt
import matplotlib as mpl
import torch.nn.functional as F
from scipy.stats import spearmanr
from matplotlib.lines import Line2D
from pathlib import Path
import pandas as pd
from matplotlib.colors import ListedColormap


def set_plt_layout():
    mpl.style.use('seaborn-v0_8-bright')
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['text.usetex'] = False
    plt.rcParams.update({
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'axes.titlesize': 20,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.bottom': False,
        'axes.spines.left': False,
        'axes.grid': False,
        'xtick.bottom': False,
        'xtick.labelbottom': False,
        'ytick.left': False,
        'ytick.labelleft': False,
        'figure.figsize': (8, 8)
    })
    mpl.rcParams['image.cmap'] = 'magma'


def reset_plt_layout():
    plt.clf()
    plt.close('all')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.style.use('default')


def set_all_seeds(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def calculate_reconstruction_metrics(model, z, dataset):
    model.eval()
    with torch.no_grad():
        reconstructions = model(z).detach().cpu()
        target_data = dataset.data
        if isinstance(target_data, np.ndarray):
             target_data = torch.from_numpy(target_data)
        target_data = target_data.cpu() 
        mae = F.l1_loss(reconstructions, target_data)
        mse = F.mse_loss(reconstructions, target_data)
    return mae.item(), mse.item()


def calculate_reconstruction_metrics_hmtDNA(model, z, dataset, threshold=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model(z)
        probs = torch.sigmoid(outputs).cpu()
        target = dataset.data
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        target = target.cpu()
        bce = F.binary_cross_entropy(probs, target, reduction='mean').item()
        preds = (probs > threshold)*1.
        tp = (preds * target).sum(dim=1)
        fp = (preds * (1 - target)).sum(dim=1)
        fn = ((1 - preds) * target).sum(dim=1)
        eps = 1e-8
        f1_per = 2 * tp / (2 * tp + fp + fn + eps)
        mean_f1 = f1_per.mean().item()
    return bce, mean_f1


def calculate_correlation_metrics_cellcycle(z, manifold, dataset, only_cc=False):
    # Filter for proliferating cells if only_cc is True
    if only_cc:
        mask = ~dataset.obs["color"].isna().values
        z_filtered = z[mask]
        theta_values = dataset.obs.cell_cycle_theta.values[mask]
        num_train_samples = len(z_filtered)
        print(f"Using {num_train_samples} proliferating cells for correlation metrics")
    else:
        z_filtered = z
        theta_values = dataset.obs.cell_cycle_theta.values
        num_train_samples = len(z)

    # Compute distance matrix between all pairs of points using the manifold's distance function
    dist_matrix = np.zeros((num_train_samples, num_train_samples))
    for i in range(num_train_samples):
        if (i + 1) % 100 == 0 or i == num_train_samples - 1: # Print progress less frequently
             print(f'Manifold distance matrix progress: {i+1}/{num_train_samples}', end='\r')
        with torch.no_grad():
            distance = manifold.dist(z_filtered[i].unsqueeze(0), z_filtered).detach().cpu().numpy().squeeze()
        dist_matrix[i] = distance

    triu_indices = np.triu_indices(num_train_samples, k=1) # upper triangular distances without redundancy
    rep_distances = dist_matrix[triu_indices]

    # Calculate cell cycle theta distances
    theta_diff = theta_values[:, None] - theta_values[None, :]
    theta_dist_matrix_abs = np.abs(theta_diff)

    # Account for circularity (theta is normalized to range [0, 1])
    theta_dist_matrix = np.minimum(theta_dist_matrix_abs, 1.0 - theta_dist_matrix_abs)
    cycle_distances = theta_dist_matrix[triu_indices]

    # Calculate correlations
    pearson_correlation = np.corrcoef(rep_distances, cycle_distances)[0, 1]
    spearman_correlation, _ = spearmanr(rep_distances, cycle_distances)

    return pearson_correlation, spearman_correlation


def calculate_correlation_metrics_hmtDNA(z, manifold, dataset, random_seed=42):
    d = dataset.data
    if len(z) > 5000:
        random.seed(random_seed) 
        indices = random.sample(range(len(z)), 5000)
        z = z[indices]
        d = dataset.data[indices]
    N = len(z)
    
    triu_idx = np.triu_indices(N, k=1)
    dist_matrix = np.zeros((N, N), dtype=float)
    for i in range(N):
        if (i + 1) % 100 == 0 or i == N - 1:
            print(f'Manifold distance progress: {i+1}/{N}', end='\r')
        with torch.no_grad():
            d_i = manifold.dist(z[i].unsqueeze(0), z).detach().cpu().numpy().squeeze()
        dist_matrix[i] = d_i
    rep_distances = dist_matrix[triu_idx]
    with torch.no_grad():
        genetic_matrix = torch.cdist(d, d, p=1).cpu().numpy()
    genetic_distances = genetic_matrix[triu_idx]
    pearson_corr = np.corrcoef(rep_distances, genetic_distances)[0, 1]
    spearman_corr, _ = spearmanr(rep_distances, genetic_distances)
    return pearson_corr, spearman_corr


def get_representations(model, loader, loss_fn, n_start_points_per_sample=100, n_epochs=1, lr=1e-3, betas=(0.5, 0.7), wd=0, device="cpu"):
    n_samples = len(loader.dataset)

    # take "n_start_points_per_sample" random samples from model.z tensor
    idx = torch.randint(0, len(model.z), (n_start_points_per_sample,))
    centers = model.z[idx]

    # repeat centers for each sample
    centers = centers.repeat(n_samples, 1, 1)
    z_init = centers.view(-1, centers.size(-1))

    # pass through model and keep the sample with minimum loss
    out = model(z_init)
    data = torch.cat([data for _, data, _ in loader], dim=0).to(device)
    loss = loss_fn(out, data.repeat_interleave(n_start_points_per_sample, dim=0)).sum(dim=-1)

    loss = loss.view(n_samples, n_start_points_per_sample, )

    # get the sample with minimum loss
    _, min_idx = torch.min(loss, dim=1)
    z_init_reshaped = z_init.view(n_samples, n_start_points_per_sample, -1)
    z_selected = z_init_reshaped[torch.arange(n_samples), min_idx]

    # Make z_selected a leaf variable with requires_grad=True
    z = z_selected.clone().detach().to(device)
    z = model.manifold.projx(z)
    z = geoopt.ManifoldParameter(z, manifold=model.manifold, requires_grad=True)
    optimizer = geoopt.optim.RiemannianAdam([z], lr=lr, weight_decay=wd, betas=betas, stabilize=10)
    model.to(device)

    for epoch in trange(n_epochs, desc="Optimizing representations"):
        optimizer.zero_grad()
        for batch in loader:
            i, data, _ = batch
            i, data = i.to(device), data.to(device)
            z_subset = z[i]
            y = model(z_subset)
            loss = loss_fn(y, data).sum()
            loss.backward()
        optimizer.step()
    return z


def load_rgd_checkpoint(checkpoint_path, model_cls, manifold=None, device="cpu", load_reps=True):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dim_list = ckpt.get("dim_list")
    if dim_list is None:
        raise ValueError("Checkpoint missing 'dim_list'.")
    if manifold is None:
        c = ckpt.get("c")
        if c is None:
            raise ValueError("Checkpoint missing curvature 'c'. Provide manifold explicitly.")
        manifold = geoopt.manifolds.Lorentz(k=c)
    try:
        model = model_cls(dim_list, manifold, device=device)
    except TypeError:
        model = model_cls(dim_list, manifold)
    state = ckpt.get("state_dict")
    if state is None:
        raise ValueError("Checkpoint missing 'state_dict'.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if k not in {"z", "z_val", "z_test"}]
    if unexpected:
        raise RuntimeError(f"Unexpected keys in state_dict: {unexpected}")
    if load_reps:
        for key in ("z", "z_val", "z_test"):
            if key in state:
                setattr(model, key, state[key].to(device))
    model.to(device)
    model.eval()
    return model, ckpt


def sample_subset(z, n_sample=5000, seed=42):
    if n_sample is None or n_sample >= len(z):
        idx = torch.arange(len(z))
        return z, idx.cpu().numpy()
    g = torch.Generator(device=z.device if torch.is_tensor(z) else "cpu")
    g.manual_seed(seed)
    idx = torch.randperm(len(z), generator=g)[:n_sample]
    return z[idx], idx.cpu().numpy()

def pairwise_manifold_distance_matrix(manifold_obj, z_points, chunk_size=256):
    n_points = z_points.shape[0]
    dist_mat = torch.empty((n_points, n_points), dtype=z_points.dtype, device=z_points.device)
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        dist_mat[start:end] = manifold_obj.dist(
            z_points[start:end, None, :],
            z_points[None, :, :]
        )
    return dist_mat

def check_distance_matrix_properties(dist_matrix):
    if not np.allclose(dist_matrix, dist_matrix.T, rtol=1e-4, atol=1e-6):
        max_err = np.max(np.abs(dist_matrix - dist_matrix.T))
        print(f"Warning: dist_matrix is not symmetric (max |A-A^T| = {max_err:.6e}), turned into symmetric matrix.")
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
    if np.any(dist_matrix < 0):
        negative_count = np.sum(dist_matrix < 0)
        min_neg = np.min(dist_matrix)
        print(f"Warning: dist_matrix contains {negative_count} negative values with minimum {min_neg:.6e}, set to 0.")
        dist_matrix[dist_matrix < 0] = 0
    if not np.allclose(np.diag(dist_matrix), 0, rtol=1e-4, atol=1e-6):
        diag_max = np.max(np.abs(np.diag(dist_matrix)))
        print(f"Warning: dist_matrix diagonal is not zero (max |diag| = {diag_max:.6e}), set diagonal to 0.")
        dist_matrix[np.diag_indices_from(dist_matrix)] = 0
    return dist_matrix


def plot_circular_tree(tree, tip_major: dict[str, str], out_path=None, figsize=(22, 22), dpi=300, save_dpi=500):
    depths = tree.depths()
    if not depths:
        raise ValueError("Tree depth calculation returned no values.")

    max_depth = max(depths.values())
    if max_depth == 0:
        depths = tree.depths(unit_branch_lengths=True)
        max_depth = max(depths.values())

    tip_order = tree.get_terminals()
    y_pos = {tip: i for i, tip in enumerate(tip_order)}

    def set_internal_y(clade):
        if clade in y_pos:
            return y_pos[clade]
        y_pos[clade] = float(np.mean([set_internal_y(child) for child in clade.clades]))
        return y_pos[clade]

    set_internal_y(tree.root)
    n_tips = max(len(tip_order), 1)

    def theta(clade):
        return 2 * np.pi * y_pos[clade] / n_tips

    def radius(clade):
        return depths.get(clade, 0.0) / max_depth

    def xy(r, th):
        return r * np.cos(th), r * np.sin(th)

    def draw_edge(parent, child, ax, line_width=0.2):
        r0, r1 = radius(parent), radius(child)
        th0, th1 = theta(parent), theta(child)

        dth = (th1 - th0 + np.pi) % (2 * np.pi) - np.pi
        arc = np.linspace(th0, th0 + dth, 24)
        ax.plot(r0 * np.cos(arc), r0 * np.sin(arc), color="black", lw=line_width, zorder=1)

        x0, y0 = xy(r0, th1)
        x1, y1 = xy(r1, th1)
        ax.plot([x0, x1], [y0, y1], color="black", lw=line_width, zorder=1)

        for grandchild in child.clades:
            draw_edge(child, grandchild, ax, line_width)

    _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for child in tree.root.clades:
        draw_edge(tree.root, child, ax)

    group_series = pd.Series([tip_major.get(tip.name, "Unknown") for tip in tip_order], dtype="string")
    cats = group_series.astype("category")
    codes = cats.cat.codes.to_numpy()
    groups = list(cats.cat.categories)

    #cmap = plt.get_cmap("magma_r", max(len(groups), 1))
    colors = (list(plt.get_cmap("tab20").colors) +
          list(plt.get_cmap("tab20b").colors) +
          list(plt.get_cmap("tab20c").colors))
    cmap = ListedColormap(colors[:len(groups)])

    tip_xy = [xy(radius(tip), theta(tip)) for tip in tip_order]
    tip_x = [pos[0] for pos in tip_xy]
    tip_y = [pos[1] for pos in tip_xy]
    ax.scatter(tip_x, tip_y, c=codes, s=5, cmap=cmap, edgecolors="white", linewidths=0.05, zorder=3)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5.5,
               markerfacecolor=cmap(i), markeredgecolor="none", label=group)
        for i, group in enumerate(groups)
    ]
    ax.legend(legend_handles, groups, title="Major haplogroup", loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=save_dpi)

    counts = group_series.value_counts().sort_index()
    return {
        "n_leaves": len(tip_order),
        "counts": counts,
        "saved_path": out_path,
    }
































def compute_dist_matrix(z, manifold, device="cpu"):
    z = z.to(device)
    n = len(z)
    dist_matrix = np.empty((n, n), dtype=np.float32)
    for i in tqdm(range(n), desc="Manifold distance", leave=True):
        with torch.no_grad():
            d_i = manifold.dist(z[i].unsqueeze(0), z).detach().cpu().numpy().squeeze()
        dist_matrix[i] = d_i
    print(f"Manifold distance: {n}/{n}")
    # Check for symmetry
    if not np.allclose(dist_matrix, dist_matrix.T, rtol=1e-4, atol=1e-6):
        max_err = np.max(np.abs(dist_matrix - dist_matrix.T))
        print(f"Warning: dist_matrix is not symmetric (max |A-A^T| = {max_err:.6e})")
    # Check for non-negativity
    if np.any(dist_matrix < 0):
        negative_count = np.sum(dist_matrix < 0)
        min_neg = np.min(dist_matrix)
        dist_matrix[dist_matrix < 0] = 0
        print(f"Warning: dist_matrix contained {negative_count} negative values with minimum {min_neg:.6e}, set to 0.")
    return dist_matrix


def compute_dist_matrix_poincare(z, manifold, device="cpu"):
    z = z.to(device)
    n = len(z)
    dist_matrix = np.empty((n, n), dtype=np.float32)

    # If Lorentz, map to Poincare ball and compute distances there
    if isinstance(manifold, geoopt.manifolds.Lorentz):
        k = manifold.k
        def lorentz_to_poincare(x):
            # x: (N, D+1) with time-like first coord
            return x[:, 1:] / x[:, :1]

        z_p = lorentz_to_poincare(z)
        poincare = geoopt.manifolds.PoincareBallExact(c=1.0/k) # or 1.0/k curvature?

        with torch.no_grad():
            for i in tqdm(range(n), desc="Poincare distance", leave=True):
                d_i = poincare.dist(z_p[i:i + 1], z_p).detach().cpu().numpy().squeeze()
                dist_matrix[i] = d_i
    else:
        with torch.no_grad():
            for i in tqdm(range(n), desc="Manifold distance", leave=True):
                d_i = manifold.dist(z[i:i + 1], z).detach().cpu().numpy().squeeze()
                dist_matrix[i] = d_i

    print(f"Manifold distance: {n}/{n}")
    if not np.allclose(dist_matrix, dist_matrix.T, rtol=1e-4, atol=1e-6):
        max_err = np.max(np.abs(dist_matrix - dist_matrix.T))
        print(f"Warning: dist_matrix is not symmetric (max |A-A^T| = {max_err:.6e})")
    return dist_matrix


def compute_group_median_dist_matrix(z, group_labels, manifold, device="cpu", project_medians=True):
    if len(z) != len(group_labels):
        raise ValueError("z and group_labels must have the same length.")

    z = z.to(device)
    labels = np.asarray(group_labels, dtype=object)
    unique_labels = np.unique(labels).tolist()

    group_representatives = []
    group_counts = []
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            continue
        z_group = z[idx]
        median = torch.median(z_group, dim=0).values
        if project_medians:
            median = manifold.projx(median.unsqueeze(0)).squeeze(0)
        group_representatives.append(median)
        group_counts.append(len(idx))

    if not group_representatives:
        raise ValueError("No groups with samples found.")

    reps = torch.stack(group_representatives, dim=0)
    n = len(reps)
    dist_matrix = np.empty((n, n), dtype=np.float32)
    print("Computing group median distance matrix...")
    for i in tqdm(range(n), desc="Group manifold distance", leave=True):
        with torch.no_grad():
            d_i = manifold.dist(reps[i].unsqueeze(0), reps).detach().cpu().numpy().squeeze()
        dist_matrix[i] = d_i
    print(f"Group manifold distance: {n}/{n}")
    if not np.allclose(dist_matrix, dist_matrix.T, rtol=1e-4, atol=1e-6):
        max_err = np.max(np.abs(dist_matrix - dist_matrix.T))
        print(f"Warning: group dist_matrix is not symmetric (max |A-A^T| = {max_err:.6e})")
    if np.any(dist_matrix < 0):
        negative_count = np.sum(dist_matrix < 0)
        min_neg = np.min(dist_matrix)
        dist_matrix[dist_matrix < 0] = 0
        print(f"Warning: group dist_matrix contained {negative_count} negative values with minimum {min_neg:.6e}, set to 0.")

    return dist_matrix, unique_labels, np.asarray(group_counts, dtype=int)


def write_phylip_dist_matrix(dist_matrix, out_path, labels=None, precision=6):
    n = dist_matrix.shape[0]
    if labels is None:
        labels = [f"S{i+1:05d}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length must match dist_matrix size.")
    with open(out_path, "w") as f:
        f.write(f"{n}\n")
        for label, row in zip(labels, dist_matrix):
            name = str(label)[:10].ljust(10)
            row_str = " ".join(f"{x:.{precision}f}" for x in row)
            f.write(f"{name} {row_str}\n")
