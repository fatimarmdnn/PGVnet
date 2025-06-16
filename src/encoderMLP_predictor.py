"""
Main entry point for training and testing the EncoderMLP model (Step 2)
Usage:
  python main.py --mode train
  python main.py --mode test --results_dir ./results/<run_id>


"""

import argparse
import torch
import numpy as np
import time
import uuid
from datetime import datetime
from preprocess import process_data   
from trainer import train, test
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'inference'], required=True)
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Required if mode is "test": path to directory with config.yaml and model checkpoint.')
    
    # Dataset and processing
    parser.add_argument("--data_tag", type=str, required = True, help="Tag for the dataset, e.g., '50_50_x6'")
    parser.add_argument("--stats_tag", type=str, default=None, help="Tag for the statistics file, defaults to data_tag if not provided")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--transform_input", type=bool, default=True)
    parser.add_argument("--transform_output", type=bool, default=True)
    parser.add_argument("--inc_gradient", type=bool, default=False)
    parser.add_argument("--inc_distance", type=bool, default=True)
    parser.add_argument("--normalize_output", type=bool, default=True)
    parser.add_argument("--normalize_input", type=bool, default=True)
    parser.add_argument("--downsample_factor", type=int, required=True, help="Downsample factor for the input data, e.g., 4, 6, or 8")
    parser.add_argument("--nx_patch", type=int, default=8)
    parser.add_argument("--ny_patch", type=int, default=8)
    parser.add_argument("--fourier_features", type=bool, default=False)
    parser.add_argument("--D", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=5)

    # Training
    parser.add_argument("--n_samp_pts_per_patch", type=int, default=512)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--num_epochs", type=int, default=1)
    # Model
    parser.add_argument("--enc_type", type=str, default="edsr")
    parser.add_argument("--in_channels", type=int, default=5)
    parser.add_argument("--out_channels", type=int, default=32)
    parser.add_argument("--conv_kernel_size", type=int, default=3)
    parser.add_argument("--attention_kernel_size", type=int, default=1)
    parser.add_argument("--num_features", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--nf", type=int, default=32)
    parser.add_argument("--activation", type=str, default="sine")

    # Misc
    parser.add_argument("--pad_size", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=3.0)

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    config = parse_args()

    seed = int(time.time() * 1000) % (2**32)
    set_seed(seed)

    if config.mode == 'train':
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        results_dir = os.path.join("./results", run_id)
        os.makedirs(results_dir, exist_ok=True)

        config.run_id = run_id
        config.results_dir = results_dir
        config.seed = seed

        print(f"Run ID: {run_id} | Seed: {seed}")

        inp_data, out_data = process_data(config)

        # Number of sources and repeats per source
        num_sources        = 50
        samples_per_source = 50
        total_samples = num_sources * samples_per_source

        # Randomly choose test sources (e.g., 10)
        num_test_sources = 10
        np.random.seed(seed)
        test_source_ids = np.random.choice(num_sources, num_test_sources, replace=False)

        # Compute test sample indices
        test_ids = np.concatenate([
            np.arange(s * samples_per_source, (s + 1) * samples_per_source)
            for s in test_source_ids])
        # Save test source IDs so we can reuse them in test mode
        np.save(os.path.join(results_dir, "test_ids.npy"), test_ids)


        # Remaining are train
        all_ids   = np.arange(total_samples)
        train_ids = np.setdiff1d(all_ids, test_ids)

        # Split data
        inp_data_test = inp_data[test_ids]
        out_data_test = out_data[test_ids]
        inp_data_train = inp_data[train_ids]
        out_data_train = out_data[train_ids]

        config.test_ids = test_ids

        train(config, inp_data_train, out_data_train)
        #test(inp_data_test, out_data_test, results_dir)
            
    elif config.mode == 'test':
        if not config.results_dir:
            raise ValueError("In test mode, you must provide --results_dir (from a previous training run).")

        config.seed = seed
        print(f"Testing only | Using results from: {config.results_dir} | Seed: {seed}")

        # Load test IDs (actual sample indices used during training)
        test_ids = np.load(os.path.join(config.results_dir, "test_ids.npy"))
        print(test_ids.shape)

        # Re-load full data (same as training)
        inp_data, out_data = process_data(config)

        # Extract test samples using exact saved indices
        inp_data_test = inp_data[test_ids]
        out_data_test = out_data[test_ids]

        preds, gts =  test(inp_data_test, out_data_test, results_dir=config.results_dir)
        np.save(os.path.join(config.results_dir, "test_preds.npy"), preds)
        np.save(os.path.join(config.results_dir, "test_gts.npy"), gts)   
        
    
    elif config.mode == 'inference':
        if not config.results_dir:
            raise ValueError("In inference mode, you must provide --results_dir (from a previous training run).")
        
        inp_data, out_data = process_data(config)
        print(inp_data.shape, out_data.shape)
        preds, gts = test(inp_data, out_data, results_dir=config.results_dir)
        print(f"Predictions shape: {preds.shape}")
        np.save(os.path.join(config.results_dir, "inference_preds.npy"), preds)


