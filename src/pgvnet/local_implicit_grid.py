# Adapted from: (https://github.com/maxjiang93/space_time_pde)

"""Local implicit grid query function."""

import torch

from . import regular_nd_grid_interpolation as rgi
from .dataset import fourier_feature

def query_local_implicit_grid(model, latent_grid, query_coords_norm, xmin, xmax, coords_absolute,
                               use_fourier, D, gamma):
    """
    Query a local implicit neural representation from a latent grid.

    Args:
        model: nn.Module to decode combined latent features and coordinates.
        latent_grid: Tensor [B, ...grid_shape..., C] of latent features.
        query_coords_norm: Tensor [B, N, D] — normalized query coordinates (0–1).
        xmin, xmax: Spatial bounds of the latent grid.
        coords_absolute: Tensor [B, N, D_abs] — absolute query coordinates.
        use_fourier: Bool — whether to apply Fourier features.
        D: Int — number of Fourier frequencies.
        gamma: Float — scaling for Fourier embedding.

    Returns:
        output: Tensor [B, N, O] — decoded predictions.
    """
    # Step 1: Interpolation weights and relative coordinates
    corner_vals, weights, coords_relative = rgi.regular_nd_grid_interpolation_coefficients(
        latent_grid, query_coords_norm, xmin, xmax)  # shapes: [B, N, 2**D, C], [B, N, 2**D], [B, N, 2**D, D]

    combined_latent = torch.cat([coords_relative, corner_vals], dim=-1)  # [B, N, 2**D, D+C]
    B, N, K, D_lat = combined_latent.shape

    # Step 2: Expand absolute coordinates across corners
    coords_absolute = coords_absolute.unsqueeze(2).repeat(1, 1, K, 1)  # [B, N, 2**D, D_abs]

    # Step 3: Optionally apply Fourier features to relative coords
    if use_fourier:
        fourier_coords = fourier_feature(coords_relative, D, gamma)  # [B, N, 2**D, D_f]
        mlp_input = torch.cat([
            combined_latent.reshape(B * N * K, D_lat),
            fourier_coords.reshape(B * N * K, -1),
            coords_absolute.reshape(B * N * K, -1)
        ], dim=-1)
    else:
        mlp_input = torch.cat([
            combined_latent.reshape(B * N * K, D_lat),
            coords_absolute.reshape(B * N * K, -1)
        ], dim=-1)

    # Step 4: MLP forward pass
    preds = model(mlp_input)  # [B*N*K, O]
    preds = preds.view(B, N, K, -1)  # [B, N, 2**D, O]

    # Step 5: Weighted interpolation
    output = torch.sum(preds * weights.unsqueeze(-1), dim=-2)  # [B, N, O]

    return output



