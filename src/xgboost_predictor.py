import numpy as np
import pickle
import os
from tqdm import tqdm
from xgboost_utils import prep_input_map
import xgboost as xgb



def generate_sparse_pgv(  
    station_coords_path: str,   
    models_dir : str,
    output_path: str,
    data_tag: str = None,
    spacing_km: int = 4
):
    """
    Generates sparse PGV maps using the trained XGBoost models and saves them as a .npz file.

    Args:
        station_coords_path: Path to .npz file with 'station_coords' key
        models_dir         : Path to directory with trained XGBoost models 
        output_path        : Directory where the sparse maps output .npz file will be saved
        data_tag           : e.g 50_50 corresponds to 50 sources location and 50 mechanism per source location
        spacing_km         : Grid spacing in km (4, 6, or 8)
    """

    # Load inputs
    source_params_path =  f'./data/forward_db/source_params_{data_tag}.npz'
    source_params = np.load(source_params_path)['source_params']
    station_coords = np.load(station_coords_path)['station_coords']
    
    n_sources = source_params.shape[0]
    n_receivers = station_coords.shape[0]

    X = prep_input_map(source_params, station_coords)
    pred_pgv = np.zeros((n_sources, n_receivers, 2))
    
    model_dir_0 = os.path.join(models_dir, 'xgb_models_0')
    model_dir_1 = os.path.join(models_dir, 'xgb_models_1')

    # Predict PGV for each receiver
    for receiver_idx in tqdm(range(n_receivers), desc=f"Predicting Sparse PGV Maps for {n_sources} events"):
        model_path0 = os.path.join(model_dir_0, f"xgb_receiver_{receiver_idx:03d}.json")
        model_path1 = os.path.join(model_dir_1, f"xgb_receiver_{receiver_idx:03d}.json")

        model0 = xgb.Booster()
        model0.load_model(model_path0)

        model1 = xgb.Booster()
        model1.load_model(model_path1)

        # Convert input to DMatrix for prediction
        dmat = xgb.DMatrix(X[:, receiver_idx, :])  # shape: (n_samples, n_features)

        pred0 = model0.predict(dmat)
        pred1 = model1.predict(dmat)

        # Stack predictions from both models
        pred_pgv[:, receiver_idx, :] = np.stack((pred0, pred1), axis=-1)

    # Format: reshape to grid and flip axes to match spatial convention in AxiSEM3D
    pred_pgv_r   = np.exp(pred_pgv.reshape(n_sources, 16, 16, 2, order='F'))
    pred_pgv_map = np.flip(pred_pgv_r, axis=2)

    # Downsample if needed
    spacing_to_size = {4: 16, 6: 11, 8: 8}
    target_size = spacing_to_size.get(spacing_km)

    if target_size is None:
        raise ValueError("spacing_km must be one of: 4, 6, or 8")

    if target_size != 16:
        pred_pgv_map_down = np.zeros((n_sources, target_size, target_size, 2))
        for i in range(n_sources):
            for comp in range(2):
                pred_pgv_map_down[i, :, :, comp] = resize_with_aligned_corners(pred_pgv_map[i, :, :, comp], target_size)
        pred_pgv_map = pred_pgv_map_down

    # Save output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    out_file = os.path.join(output_path, f"step1_preds_{data_tag}_x{spacing_km}.npz")
    np.savez(out_file, pred_pgv=pred_pgv_map, spacing_km=spacing_km)
    print(f"Saved sparse PGV map to {out_file}")
    return pred_pgv_map
    
    
def resize_with_aligned_corners(data, target_size):
    """
    Resize a 2D array (lat x lon) to target_size x target_size
    while aligning the corners using linear interpolation.
    """
    from scipy.interpolate import RegularGridInterpolator

    input_size = data.shape[0]
    grid_x = np.linspace(0, 1, input_size)
    grid_y = np.linspace(0, 1, input_size)
    interp = RegularGridInterpolator((grid_x, grid_y), data, method='linear', bounds_error=False, fill_value=None)

    new_x = np.linspace(0, 1, target_size)
    new_y = np.linspace(0, 1, target_size)
    new_grid = np.array(np.meshgrid(new_x, new_y, indexing='ij')).reshape(2, -1).T
    resized = interp(new_grid).reshape((target_size, target_size))
    return resized



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--station_coords", type=str, default = "data/station_coords_sparse.npz",  )
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="data/step1_preds")
    parser.add_argument("--data_tag", type=str, required=True, help="Tag for the output data file")
    parser.add_argument("--spacing_km", type=int, required = True, choices=[4, 6, 8], default=4, help="Grid spacing in km (supports 4, 6, or 8)")

    args = parser.parse_args()

    generate_sparse_pgv(
        station_coords_path=args.station_coords,
        models_dir         =args.models_dir,
        output_path        =args.output_path,
        data_tag           =args.data_tag,
        spacing_km         =args.spacing_km
    )
