import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image 
import numpy as np
import torch    
import matplotlib as mpl
from skimage.metrics import structural_similarity as ssim

def generate_test_images(hres_data_test, re_field_hres, sim_nb, comp): 
    
    if isinstance(hres_data_test, torch.Tensor):
        hres_data_test = hres_data_test.detach().cpu().numpy()
    if isinstance(re_field_hres, torch.Tensor):
        re_field_hres = re_field_hres.detach().cpu().numpy()
        
    #change from mm/s to cm/s   
    hres_data_test = hres_data_test / 10
    re_field_hres  = re_field_hres / 10
        
    error = np.abs(re_field_hres[sim_nb, :, :, comp] - hres_data_test[sim_nb, :, :, comp]) 
    
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))

    pgv_norm   = mpl.colors.Normalize(vmin=0, vmax=np.max(hres_data_test[sim_nb, :, :, comp]))
    error_norm = mpl.colors.Normalize(vmin=0, vmax=np.max(error))     

    contour1 = axs[0].contourf(hres_data_test[sim_nb, :, :, comp], cmap='plasma', norm=pgv_norm, levels=10)
    contour2 = axs[1].contourf(re_field_hres[sim_nb, :, :, comp], cmap='plasma', norm=pgv_norm, levels=10)

    # Add a shared colorbar for the first and second plots
    fig.colorbar(contour1, ax=[axs[0], axs[1]], norm=pgv_norm, orientation='vertical', label='PGV (cm/s)').set_label("PGV (cm/s)", fontsize=14) 

    # Use error_norm explicitly
    contour3 = axs[2].contourf(error, cmap='Grays', levels=100, norm=error_norm)

    # Add colorbar for the third plot
    fig.colorbar(contour3, ax=axs[2], orientation='vertical', label='Error (cm/s)', norm = error_norm).set_label("Error (cm/s)", fontsize=14) 

    # Titles and labels
    axs[0].set_xlabel('Distance Along X (Km)', fontsize=12)
    axs[0].set_ylabel('Distance Along Y (Km)', fontsize=12)
    axs[0].set_title('Ground Truth PGV (Simulated)', fontsize=12) 
    axs[1].set_title('Predicted PGV', fontsize=12)
    axs[2].set_title('Error', fontsize=12)

    buf = BytesIO() 
    plt.savefig(buf, format='png')
    plt.close(fig)  
    buf.seek(0)

    # Convert the buffer to a PIL image
    pil_image = Image.open(buf)
    
    return pil_image



def compute_ssim_field(ground_truth, prediction):
    """
    Computes the average SSIM over all test samples and components.
    """
    total_ssim = 0
    count = 0
    for i in range(ground_truth.shape[0]):  # Iterate over samples
        for comp in range(ground_truth.shape[-1]):  # Iterate over components
            gt_field = ground_truth[i, ..., comp]
            pred_field = prediction[i, ..., comp]
            
            # Compute SSIM for the individual component
            ssim_value = ssim(gt_field, pred_field, data_range=pred_field.max() - pred_field.min())
            total_ssim += ssim_value
            count += 1
    return total_ssim / count

# additional plotting utilities for the Demo

import os
import cartopy.crs as ccrs
from obspy.imaging.beachball import beach

def plot_pgv_example(
    results_dir: str,
    idx: int,
    *,
    comp: int = 1,
    utm_zone: int = 10,
    center_lon: float = -122.1,
    center_lat: float = 37.7,
    buffer_m: float = 5e4,
    cmap: str = "plasma",
    source_params_path: str = "data/forward_db/source_params_100_50.npz"
):
    """Return a Figure plotting one example (map + GT/Pred)."""

    sp = np.load(source_params_path)['source_params']
    test_ids = np.load(os.path.join(results_dir, 'test_ids.npy'))
    preds = np.load(os.path.join(results_dir, 'test_preds.npy'))
    gts   = np.load(os.path.join(results_dir, 'test_gts.npy'))

    if not (0 <= idx < preds.shape[0]):
        raise ValueError(f"idx out of range: 0..{preds.shape[0]-1}")

    utm = ccrs.UTM(zone=utm_zone, southern_hemisphere=False)
    geo = ccrs.Geodetic()
    center_utm = np.array(utm.transform_point(center_lon, center_lat, src_crs=geo))

    src_xy = sp[test_ids, 0:2] + center_utm
    mt = sdr_to_moment_tensor(sp[test_ids, 3], sp[test_ids, 4], sp[test_ids, 5])

    gt   = gts[idx,   :, :, comp] / 10.0
    pred = preds[idx, :, :, comp] / 10.0
    norm = plt.Normalize(vmin=0, vmax=np.max(gt))

    # Use constrained_layout instead of tight_layout (Cartopy-friendly)
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1.25, 1.0, 1.0])

    ax0 = fig.add_subplot(gs[0, 0], projection=utm)
    ax0.set_extent([center_utm[0] - buffer_m, center_utm[0] + buffer_m,
                    center_utm[1] - buffer_m, center_utm[1] + buffer_m], crs=utm)
    ax0.coastlines(resolution="10m", alpha=0.5)

    x, y = map(float, src_xy[idx])
    #ax0.scatter([x], [y], marker='^', color='red', s=40, label='Source', transform=utm)
    bb = beach(mt[idx], xy=(x, y), width=50, linewidth=1,
               facecolor='k', edgecolor='k', alpha=0.8, axes=ax0)
    ax0.add_collection(bb)

    gl = ax0.gridlines(draw_labels=True, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    ax0.set_aspect('equal', 'box')
    ax0.set_title('Source Location and Mechanism', fontsize=14) 

    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    cf1 = ax1.contourf(gt, cmap=cmap, norm=norm)
    cf2 = ax2.contourf(pred, cmap=cmap, norm=norm)

    ax1.set_title('Simulations')
    ax2.set_title('PGVnet')
    ax1.set_xlabel('East Distance (km)'); ax1.set_ylabel('North Distance (km)')
    ax2.set_xlabel('East Distance (km)'); ax2.set_ylabel('North Distance (km)')

    ax1.text(0.03, 0.97, f'Max: {np.max(gt):.2f} cm/s', transform=ax1.transAxes,
             fontsize=11, va='top', ha='left',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.25'))
    ax2.text(0.03, 0.97, f'Max: {np.max(pred):.2f} cm/s', transform=ax2.transAxes,
             fontsize=11, va='top', ha='left',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.25'))

    # With constrained_layout, omit fraction/pad unless needed
    cbar = fig.colorbar(cf2, ax=[ax1, ax2], location='right')
    cbar.set_label("PGV (cm/s)", fontsize=12)

    return fig



def sdr_to_moment_tensor(strike, dip, rake, M0=1.0):
    """
    Convert strike, dip, rake (degrees) to a normalized moment tensor.

    Args:
        strike, dip, rake: Scalars or arrays in degrees
        M0: Scalar moment (can be scalar or array); default = 1.0

    Returns:
        A numpy array of shape (..., 6): [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
    """
    strike = np.radians(strike)
    dip = np.radians(dip)
    rake = np.radians(rake)

    # Sines and cosines
    cos_lambda = np.cos(rake)
    sin_lambda = np.sin(rake)
    cos_phi = np.cos(strike)
    sin_phi = np.sin(strike)
    cos_delta = np.cos(dip)
    sin_delta = np.sin(dip)

    # Compute moment tensor components
    Mxx = -M0 * (sin_delta * cos_lambda * np.sin(2 * strike) + np.sin(2 * dip) * sin_lambda * (np.sin(strike))**2)
    Mxy = M0 * (sin_delta * cos_lambda * np.cos(2 * strike) + 0.5 * np.sin(2 * dip) * sin_lambda * np.sin(2 * strike))
    Mxz = -M0 * (cos_delta * cos_lambda * cos_phi + np.cos(2 * dip) * sin_lambda * np.sin(strike))
    Myy = M0 * (sin_delta * cos_lambda * np.sin(2 * strike) - np.sin(2 * dip) * sin_lambda * np.cos(strike)**2)
    Myz = -M0 * (cos_delta * cos_lambda * np.sin(strike) - np.cos(2 * dip) * sin_lambda * np.cos(strike))
    Mzz = M0 * np.sin(2 * dip) * sin_lambda
    
    Mrr = Mzz
    Mtt = Mxx   
    Mpp = Myy   

    Mrt = Mxz
    Mrp = -Myz   
    Mtp = -Mxy  

    M = np.stack([Mrr, Mtt, Mpp, Mrt, Mrp, Mtp], axis = -1)    

    return M 