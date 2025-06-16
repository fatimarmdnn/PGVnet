import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class PGVDatasetPatchesPoints(Dataset):
    
    """
    PyTorch Dataset for extracting low-resolution and high-resolution PGV patches,
    and generating random point samples for super-resolution tasks.
    """
    
    def __init__(self, inp_data, out_data, config):

        self.inp_grid                   = inp_data            # pgv_lres
        self.out_grid                   = out_data            # pgv_hres
        
        self.global_grid_size           = out_data.shape[1:3] # 61 x 61
 
        self.nx_patch  = config.nx_patch  # patch size along x
        self.ny_patch  = config.ny_patch  # patch size along y    
        self.downsample_factor = config.downsample_factor  # factor between lres and hres  
        
        self.n_samp_pts_per_patch = config.n_samp_pts_per_patch  
        
        self.config = config

        nb_sims, nx_lres, ny_lres, _ = self.inp_grid.shape  # 90, 16, 16

        nx_start_range = np.arange(0, nx_lres - config.nx_patch, 2)
        ny_start_range = np.arange(0, ny_lres - config.ny_patch, 2)
        rand_grid      = np.stack(np.meshgrid(ny_start_range,nx_start_range, indexing='ij'), axis=-1)

        rand_start_id = rand_grid.reshape([-1, 2])

        xy_ids   = np.tile(rand_start_id, (nb_sims, 1))
        sim_ids  = np.repeat(np.arange(nb_sims), len(rand_start_id)).reshape(-1, 1)  
        rand_ids = np.hstack((sim_ids, xy_ids))
        self.rand_ids = rand_ids 
        print(f'Number of patches: {len(self.rand_ids)}')
        
        self.inp_grid       = torch.tensor(self.inp_grid.copy(),  dtype=torch.float32) 
        self.out_grid       = torch.tensor(self.out_grid.copy(), dtype=torch.float32)
    
    def __len__(self):
        return len(self.rand_ids)   
    
    def __getitem__(self, idx): 
        
        sim_id, y_id, x_id      = self.rand_ids[idx]
        patch_xy_lres           = self.inp_grid[sim_id, y_id:y_id+self.ny_patch,x_id:x_id+self.nx_patch,:]  
        
        # transform to hres coordinates
        x_id_hres, y_id_hres    = x_id *  self.downsample_factor, y_id * self.downsample_factor
        patch_xy_hres           = self.out_grid[sim_id, 
                                                y_id_hres :y_id_hres+self.ny_patch*self.downsample_factor, 
                                                x_id_hres: x_id_hres+ self.nx_patch*self.downsample_factor, :]
        
        lr_pgv = patch_xy_lres.permute(2, 0, 1)  # (channel, x, y)    
        hr_pgv = patch_xy_hres.permute(2, 0, 1)

        # sample random points from the hres patch 
        nx_hres = hr_pgv.shape[2]
        ny_hres = hr_pgv.shape[1]
        scale_hres = np.array([nx_hres, ny_hres])
 
        interp = RegularGridInterpolator((np.arange(ny_hres), np.arange(nx_hres)),values=patch_xy_hres.numpy(), method='linear')

        # create random point samples within space time crop
        point_coord = np.random.rand(self.n_samp_pts_per_patch, 2) * (scale_hres - 1)
        point_value = interp(point_coord)
        
        point_coord_abs = point_coord + np.array([y_id_hres, x_id_hres])
        
        # normalize coords
        point_coord     = point_coord / (scale_hres - 1)
        point_coord_abs = point_coord_abs / np.array(self.global_grid_size)
        point_coord_abs = torch.tensor(point_coord_abs, dtype=torch.float32)
    
        #transform to fourier features  
        if self.config.fourier_features:
            point_coord_abs = fourier_feature(point_coord_abs, self.config.D, self.config.gamma)   
             
        point_coord = torch.tensor(point_coord,  dtype=torch.float32)
        point_value = torch.tensor(point_value,  dtype=torch.float32)
        

        return hr_pgv, lr_pgv, point_coord, point_value, point_coord_abs 


def fourier_feature(x: torch.Tensor, D: int, gamma: float):
    """
    Applies random Fourier feature mapping to input coordinates.

    Args:
        x (Tensor): Input of shape [N, d]
        D (int): Number of frequencies
        gamma (float): Scaling factor

    Returns:
        Tensor: Fourier features of shape [N, 2D]
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    proj = torch.randn(x.shape[-1], D, device=x.device) * np.sqrt(2 * gamma)
    phase = torch.rand(D, device=x.device) * 2 * np.pi
    x_proj = x @ proj + phase
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


