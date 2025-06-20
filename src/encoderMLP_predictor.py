"""
EncoderMLP Main Entry Point
Usage:
  CLI:
    python encoderMLP_main.py --mode train --data_tag 50_50_x4 --downsample_factor 4
  Programmatic:
    from encoderMLP_main import run_train, run_test, run_inference
"""

import argparse
import os
import time
import uuid
import numpy as np
import torch
from datetime import datetime

from preprocess import process_data
from trainer import train, test


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_train(config):
    seed = int(time.time() * 1000) % (2**32)
    set_seed(seed)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    results_dir = os.path.join("./results", run_id)
    os.makedirs(results_dir, exist_ok=True)

    config.run_id = run_id
    config.results_dir = results_dir
    config.seed = seed

    print(f"Run ID: {run_id} | Seed: {seed}")

    inp_data, out_data = process_data(config)

    num_sources = int(config.data_tag.split('_')[0])
    samples_per_source = 50
    total_samples = num_sources * samples_per_source
    test_source_ids = np.random.choice(num_sources, 20, replace=False)
    test_ids = np.concatenate([
        np.arange(s * samples_per_source, (s + 1) * samples_per_source)
        for s in test_source_ids
    ])
    np.save(os.path.join(results_dir, "test_ids.npy"), test_ids)

    all_ids = np.arange(total_samples)
    train_ids = np.setdiff1d(all_ids, test_ids)

    inp_data_train = inp_data[train_ids]
    out_data_train = out_data[train_ids]

    config.test_ids = test_ids
    train(config, inp_data_train, out_data_train)


def run_test(config):
    set_seed(int(time.time() * 1000) % (2**32))

    if not config.results_dir:
        raise ValueError("Missing --results_dir in test mode")

    print(f"Testing using results from: {config.results_dir}")
    test_ids = np.load(os.path.join(config.results_dir, "test_ids.npy"))

    inp_data, out_data = process_data(config)
    inp_data_test = inp_data[test_ids]
    out_data_test = out_data[test_ids]

    preds, gts = test(inp_data_test, out_data_test, results_dir=config.results_dir)
    np.save(os.path.join(config.results_dir, "test_preds.npy"), preds)
    np.save(os.path.join(config.results_dir, "test_gts.npy"), gts)


# for testing on a unified test set across the different models
def run_inference(config):
    set_seed(int(time.time() * 1000) % (2**32))

    if not config.results_dir:
        raise ValueError("Missing --results_dir in inference mode")

    inp_data, out_data = process_data(config)
    preds, gts = test(inp_data, out_data, results_dir=config.results_dir)
    np.save(os.path.join(config.results_dir, "inference_preds.npy"), preds)


def parse_args():
    parser = argparse.ArgumentParser(description="EncoderMLP Trainer")

    parser.add_argument('--mode', type=str, choices=['train', 'test', 'inference'], required=True)
    parser.add_argument('--results_dir', type=str, default=None)

    # Dataset and processing
    parser.add_argument("--data_tag", type=str, required=True)
    parser.add_argument("--stats_tag", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--transform_input", type=bool, default=True)
    parser.add_argument("--transform_output", type=bool, default=True)
    parser.add_argument("--inc_gradient", type=bool, default=False)
    parser.add_argument("--inc_distance", type=bool, default=True)
    parser.add_argument("--normalize_output", type=bool, default=True)
    parser.add_argument("--normalize_input", type=bool, default=True)
    parser.add_argument("--downsample_factor", type=int, required=True)
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



if __name__ == "__main__":
    config = parse_args()

    if config.mode == "train":
        run_train(config)
    elif config.mode == "test":
        run_test(config)
    elif config.mode == "inference":    
        run_inference(config)