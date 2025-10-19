"""
EncoderMLP Main Entry Point
Handles training, testing, and inference for the EncoderMLP model.
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
    
    SEED = 20251005
    N_RUNS = 3
    TEST_FRAC = 0.10       # 10% test each run
    DISJOINT = True       
    
    set_seed(SEED)

    # ------------------ MASTER RUN SETUP ------------------
    rng_master = np.random.default_rng(SEED)
    master_run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    base_results_dir = os.path.join("./results", master_run_id)
    os.makedirs(base_results_dir, exist_ok=True)

    config.run_id = master_run_id
    config.results_dir = base_results_dir
    config.seed = SEED
    print(f"[MASTER] run_id={master_run_id} | seed={SEED}")

    # ------------------ LOAD DATA ------------------
    inp_data, out_data = process_data(config)
    N = len(inp_data)
    assert len(out_data) == N, "inp/out size mismatch"

    # ------------------ PREP SPLITS ------------------
    test_size = max(1, int(round(TEST_FRAC * N)))

    if DISJOINT:
        assert N_RUNS * test_size <= N, \
            f"DISJOINT=True requires N_RUNS*test_size <= N ({N_RUNS*test_size} > {N})"
        perm = rng_master.permutation(N)
        run_test_id_sets = [perm[i*test_size:(i+1)*test_size] for i in range(N_RUNS)]
    else:
        # Independent random 10% each run (test sets may overlap across runs)
        run_test_id_sets = []
        for i in range(N_RUNS):
            rng_i = np.random.default_rng(SEED + i)  # reproducible per run
            run_test_id_sets.append(rng_i.choice(N, size=test_size, replace=False))

    # ------------------ RUNS ------------------
    for run_idx, test_ids in enumerate(run_test_id_sets):
        fold_run_id = f"{master_run_id}_run{run_idx}_p{int(TEST_FRAC*100)}"
        results_dir = os.path.join(base_results_dir, fold_run_id)
        os.makedirs(results_dir, exist_ok=True)

        test_ids = np.sort(np.asarray(test_ids, dtype=np.int64))
        train_ids = np.setdiff1d(np.arange(N, dtype=np.int64), test_ids, assume_unique=False)

        # Save splits for reproducibility
        np.save(os.path.join(results_dir, "test_ids.npy"),  test_ids)
        np.save(os.path.join(results_dir, "train_ids.npy"), train_ids)

        # Update config for this run
        config.run_id = fold_run_id
        config.results_dir = results_dir
        config.seed = SEED + run_idx  # optional jitter per run

        print(f"[Run {run_idx}] test={len(test_ids)} ({TEST_FRAC:.0%}) | train={len(train_ids)} | dir={results_dir}")

        # Train once (your train() handles its own internal val split)
        model = train(config, inp_data[train_ids], out_data[train_ids])
        
    print("All runs complete.")



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