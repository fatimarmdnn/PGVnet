import os
import numpy as np
from typing import Dict, Tuple


def load_pgv_data(config) -> Dict[str, np.ndarray]:
    """
    Load PGV data and metadata from .npz files.
    """
    suffix     = config.data_tag
    step1_path = os.path.join(config.data_dir, f'step1_preds/step1_preds_{suffix}.npz')
    db_path    = os.path.join(config.data_dir, f'forward_db/pgv_database_{suffix}.npz')

    try:
        pgv_lres = np.load(step1_path)['pred_pgv']
        db_file = np.load(db_path)
        pgv_hres = db_file['pgv_hres']
        distance_map = db_file['distances']
        azimuth_map = db_file['azimuths']
        depth_map = db_file['depths']
    except (FileNotFoundError, KeyError) as e:
        raise RuntimeError(f"Data loading failed: {e}")

    return {
        'pgv_lres': pgv_lres,
        'pgv_hres': pgv_hres,
        'distance_map': distance_map,
        'azimuth_map': azimuth_map,
        'depth_map': depth_map
    }


def apply_transforms(data: Dict[str, np.ndarray], config) -> Dict[str, np.ndarray]:
    """
    Apply log transforms and compute gradients.
    """
    if config.transform_input:
        data['pgv_lres'] = np.log(data['pgv_lres'])
    if config.transform_output:
        data['pgv_hres'] = np.log(data['pgv_hres'])

    grad_x, grad_y = np.gradient(data['pgv_lres'], axis=(1, 2))
    data['pgv_grad_x'] = grad_x
    data['pgv_grad_y'] = grad_y
    return data


def compute_statistics(data: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute mean and std for each field.
    """
    stats = {}
    for key, value in data.items():
        mean = np.mean(value, axis=(0, 1, 2))
        std = np.std(value, axis=(0, 1, 2))
        stats[key] = (mean, std)
    return stats


def save_stats(stats: Dict[str, Tuple[np.ndarray, np.ndarray]], path: str):
    """
    Save statistics to a .npz file.
    """
    flat = {f"{k}_mean": v[0] for k, v in stats.items()}
    flat.update({f"{k}_std": v[1] for k, v in stats.items()})
    np.savez(path, **flat)


def load_stats(path: str, keys) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load statistics from a .npz file.
    """
    stats_file = np.load(path)
    return {
        k: (stats_file[f"{k}_mean"], stats_file[f"{k}_std"]) for k in keys
    }


def normalize_data(data: Dict[str, np.ndarray], stats: Dict[str, Tuple[np.ndarray, np.ndarray]], config) -> Dict[str, np.ndarray]:
    """
    Normalize all data fields.
    """
    def norm(x, mean, std):
        return (x - mean) / std

    if config.normalize_input:
        data['pgv_lres'] = norm(data['pgv_lres'], *stats['pgv_lres'])
    if config.normalize_output:
        data['pgv_hres'] = norm(data['pgv_hres'], *stats['pgv_hres'])

    data['pgv_grad_x']   = norm(data['pgv_grad_x'], *stats['pgv_grad_x'])
    data['pgv_grad_y']   = norm(data['pgv_grad_y'], *stats['pgv_grad_y'])
    data['distance_map'] = norm(data['distance_map'], *stats['distance_map'])
    data['azimuth_map']  = norm(data['azimuth_map'], *stats['azimuth_map'])
    data['depth_map']    = norm(data['depth_map'], *stats['depth_map'])
    return data


def assemble_input(data: Dict[str, np.ndarray], config) -> np.ndarray:
    
    components = [data['pgv_lres']]

    if config.inc_gradient:
        components.append(data['pgv_grad_x'])
        components.append(data['pgv_grad_y'])
    if config.inc_distance:
        components.append(data['distance_map'])
    components.append(data['azimuth_map'])
    components.append(data['depth_map'])

    return np.concatenate(components, axis=3)


def process_data(config) -> Tuple[np.ndarray, np.ndarray]:
    data = load_pgv_data(config)
    data = apply_transforms(data, config)

    # Determine stats_tag from config, fallback to data_tag if necessary
    stats_tag = config.stats_tag if config.stats_tag else config.data_tag
    stats_path = os.path.join(config.data_dir, f'data_stats_{stats_tag}.npz')

    if config.mode == 'train':
        stats = compute_statistics(data)
        save_stats(stats, stats_path)
    else:
        stats = load_stats(stats_path, data.keys())

    data = normalize_data(data, stats, config)
    inp_map = assemble_input(data, config)
    out_map = data['pgv_hres']

    return inp_map, out_map

