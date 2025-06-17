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