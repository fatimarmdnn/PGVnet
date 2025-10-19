"""
train utils for the EncoderMLP 
"""

import torch
import torch.nn.functional as F
from .local_implicit_grid import query_local_implicit_grid

def train_step(unet, imnet, data_loader, criterion, optimizer, device, config):
    
    unet.train()
    imnet.train()

    total_loss, total_point_loss, total_upsample_loss = 0.0, 0.0, 0.0
    λ = getattr(config, 'upsample_weight', 0.01)

    xmin = torch.zeros(2, dtype=torch.float32, device=device)
    xmax = torch.ones(2, dtype=torch.float32, device=device)

    for batch in data_loader:
        hr_pgv, lr_pgv, point_coord, point_value, point_coord_abs = batch

        lr_pgv = lr_pgv.to(device)
        hr_pgv = hr_pgv.to(device).unsqueeze(0) if hr_pgv.dim() == 3 else hr_pgv.to(device)
        point_coord = point_coord.to(device)
        point_value = point_value.to(device)
        point_coord_abs = point_coord_abs.to(device)

        optimizer.zero_grad()

        latent_grid = unet(lr_pgv).permute(0, 2, 3, 1)  # (B, H, W, C)
        pred_value = query_local_implicit_grid(
            imnet, latent_grid, point_coord,
            xmin, xmax, point_coord_abs,
            config.fourier_features, config.D, config.gamma)

        L_point = criterion(pred_value, point_value)

        up_h = lr_pgv.shape[2] * config.downsample_factor
        up_w = lr_pgv.shape[3] * config.downsample_factor
        upsampled_lr = F.interpolate(lr_pgv[:, :2, :, :], size=(up_h, up_w), mode='bilinear', align_corners=True)

        L_upsample = F.mse_loss(upsampled_lr, hr_pgv)

        total = L_point + λ * L_upsample
        total.backward()
        optimizer.step()

        total_loss += total.item()
        total_point_loss += L_point.item()
        total_upsample_loss += L_upsample.item()

    n = len(data_loader)
    return {
        'total_loss': total_loss / n,
        'point_loss': total_point_loss / n,
        'upsample_loss': total_upsample_loss / n
    }


def eval_step(unet, imnet, data_loader, criterion, device, config):
    unet.eval()
    imnet.eval()

    total_loss = 0.0
    xmin = torch.zeros(2, dtype=torch.float32, device=device)
    xmax = torch.ones(2, dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in data_loader:
            hr_pgv, lr_pgv, point_coord, point_value, point_coord_abs = batch

            lr_pgv = lr_pgv.to(device)
            point_coord = point_coord.to(device)
            point_value = point_value.to(device)
            point_coord_abs = point_coord_abs.to(device)

            latent_grid = unet(lr_pgv).permute(0, 2, 3, 1)
            pred_value = query_local_implicit_grid(
                imnet, latent_grid, point_coord,
                xmin, xmax, point_coord_abs,
                config.fourier_features, config.D, config.gamma
            )

            loss = criterion(pred_value, point_value)
            total_loss += loss.item()

    return total_loss / len(data_loader)



