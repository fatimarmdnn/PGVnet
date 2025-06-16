import os
import numpy as np
import torch
from tqdm import tqdm
from local_implicit_grid import query_local_implicit_grid
import dataset as ds

def generate_patch_indices(nx_lres, ny_lres, nx_patch, ny_patch, nb_sims, stride_x, stride_y):
    nx_start_range = np.arange(0, nx_lres - nx_patch + 1, stride_x)
    ny_start_range = np.arange(0, ny_lres - ny_patch + 1, stride_y)
    patch_grid = np.stack(np.meshgrid(ny_start_range, nx_start_range, indexing='ij'), axis=-1).reshape(-1, 2)
    sim_ids = np.repeat(np.arange(nb_sims), len(patch_grid))
    all_coords = np.tile(patch_grid, (nb_sims, 1))
    return np.column_stack((sim_ids, all_coords))

def postprocess_output(reconstructed, reference, config, stats_path):
    if config.normalize_output:
        stats = np.load(stats_path)
        mean  = stats["pgv_hres_mean"]
        std   = stats["pgv_hres_std"]
        reconstructed = reconstructed * std + mean
        reference = reference * std + mean

    if config.transform_output:
        reconstructed = np.exp(reconstructed)
        reference = np.exp(reference)

    return reconstructed, reference


def test_step(inp_data, out_data, imnet, unet, config, device):
    pad = config.pad_size
    d = config.downsample_factor
    nx_patch = config.nx_patch
    ny_patch = config.ny_patch
    sigma = config.sigma

    inp_data = np.pad(inp_data, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')
    out_data = np.pad(out_data, ((0, 0), (pad * d, pad * d), (pad * d, pad * d), (0, 0)), mode='reflect')

    nb_sims, nx_lres, ny_lres, _ = inp_data.shape
    _, nx_hres, ny_hres, _ = out_data.shape
    patch_indices = generate_patch_indices(nx_lres, ny_lres, nx_patch, ny_patch, nb_sims, nx_patch // 2, ny_patch // 2)

    pred_field = np.zeros((nb_sims, ny_hres, nx_hres, out_data.shape[-1]))
    weight_sum = np.zeros((nb_sims, ny_hres, nx_hres))
    patch_weights = get_gaussian_weights((ny_patch * d, nx_patch * d), sigma=sigma)

    imnet.eval()
    unet.eval()

    xmin = torch.zeros(2, dtype=torch.float32, device=device)
    xmax = torch.ones(2, dtype=torch.float32, device=device)

    for sim_id, y_id, x_id in tqdm(patch_indices, desc="Reconstructing patches For Test Maps"):
        patch = inp_data[sim_id, y_id:y_id+ny_patch, x_id:x_id+nx_patch, :][None, ...]
        patch = torch.from_numpy(patch).float().permute(0, 3, 1, 2).to(device)

        y_hr, x_hr = y_id * d, x_id * d

        with torch.no_grad():
            latent = unet(patch).permute(0, 2, 3, 1)
            x_seq = torch.linspace(0, 1, nx_patch * d, device=device)
            y_seq = torch.linspace(0, 1, ny_patch * d, device=device)
            coords = torch.cartesian_prod(x_seq, y_seq).reshape(1, -1, 2)
            abs_coords = coords + torch.tensor([y_hr, x_hr], device=device)
            abs_coords = abs_coords / torch.tensor(out_data.shape[1:3], device=device)

            if config.fourier_features:
                abs_coords = ds.fourier_feature(abs_coords.squeeze(0).cpu(), config.D, config.gamma)
                abs_coords = torch.as_tensor(abs_coords).float().unsqueeze(0).to(device)


            preds = query_local_implicit_grid(imnet, latent, coords, xmin, xmax, abs_coords, config.fourier_features, config.D, config.gamma)
            patch_hres = preds.reshape(ny_patch * d, nx_patch * d, -1).cpu().numpy()

        slice_y_end = min(y_hr + patch_hres.shape[0], ny_hres)
        slice_x_end = min(x_hr + patch_hres.shape[1], nx_hres)
        valid_h = slice_y_end - y_hr
        valid_w = slice_x_end - x_hr

        pred_field[sim_id, y_hr:slice_y_end, x_hr:slice_x_end, :] += patch_hres[:valid_h, :valid_w] * patch_weights[:valid_h, :valid_w][..., None]
        weight_sum[sim_id, y_hr:slice_y_end, x_hr:slice_x_end] += patch_weights[:valid_h, :valid_w]

    weight_sum[weight_sum == 0] = 1
    pred_field /= weight_sum[..., None]

    pad_hr = pad * d
    pred_field = pred_field[:, pad_hr:-pad_hr, pad_hr:-pad_hr, :]
    hres_ref = out_data[:, pad_hr:-pad_hr, pad_hr:-pad_hr, :]
    lres_ref = inp_data[:, pad:-pad, pad:-pad, :]

    stats_path = os.path.join(config.data_dir, f"data_stats_{config.data_tag}.npz")
    pred_field, hres_ref = postprocess_output(pred_field, hres_ref, config, stats_path)

    return pred_field, hres_ref, lres_ref

def get_gaussian_weights(patch_size, sigma=1.0):
    ny, nx = patch_size
    y, x = np.ogrid[:ny, :nx]
    y0, x0 = (ny - 1) / 2., (nx - 1) / 2.
    d2 = (x - x0)**2 + (y - y0)**2
    weights = np.exp(-d2 / (2 * sigma**2))
    return weights / weights.max()
