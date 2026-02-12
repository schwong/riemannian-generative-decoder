import torch
import torch.nn as nn
import geoopt


class RGD(nn.Module):
    def __init__(self, dim_list, manifold=geoopt.manifolds.Euclidean(), output_activation=None, device="cpu"):
        super(RGD, self).__init__()
        self.dim_list = dim_list
        self.manifold = manifold
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.origin = self.manifold.origin(dim_list[0], seed=None, device=self.device)

        # fc decoder stack with swish activations
        layers = []
        for i in range(len(dim_list) - 1):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            if i != len(dim_list) - 2:
                layers.append(nn.SiLU())
        if output_activation is not None:
            layers.append(output_activation)
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)

    def init_samples(self, n, device=None):
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = self.device
        z = 1e-3 * torch.randn(n, self.dim_list[0], device=device) - self.origin.to(device)
        z = self.manifold.projx(z)
        return geoopt.ManifoldParameter(z, manifold=self.manifold, requires_grad=True)
