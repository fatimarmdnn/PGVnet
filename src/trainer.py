# trainer.py
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from dataset import PGVDatasetPatchesPoints
import train_utils, test_utils, plot_utils
import unet, edsr, implicit_net
import losses
import yaml
import argparse


def build_model(config):
    
    if  config.enc_type == 'unet':
        encnet = unet.AttentionUNet(in_channels=config.in_channels, out_channels=config.out_channels)
    elif config.enc_type == 'edsr':
        encnet = edsr.EDSREncoder(in_channels=config.in_channels, num_features=config.num_features,
                                    num_blocks=config.num_blocks, out_channels=config.out_channels)
    else:
        raise ValueError(f"Unsupported encoder type: {config.enc_type}")

    dim   = 2 + 4 * config.D if config.fourier_features else 4
    imnet = implicit_net.ImNet(dim=dim, in_features=config.out_channels, out_features=2,
                                nf=config.nf, activation=config.activation)
    return encnet, imnet


def train(config, inp_data_train, out_data_train):
    
    dataset    = PGVDatasetPatchesPoints(inp_data_train, out_data_train, config)
    train_size = int(config.split * len(dataset))
    val_size   = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    encnet, imnet = build_model(config)
    loss_fn       = losses.compute_loss(config.loss_type)

    params = list(encnet.parameters()) + list(imnet.parameters())
    opt_args = {  'lr': config.learning_rate,'weight_decay': getattr(config, 'weight_decay', 0)}

    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(params, **opt_args)
    elif config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(params, **opt_args)
    elif config.optimizer.lower() == 'sgd':
        opt_args['momentum'] = getattr(config, 'momentum', 0.9)
        optimizer = optim.SGD(params, **opt_args)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encnet.to(device)
    imnet.to(device)

    best_model_path = os.path.join(config.results_dir, 'best_model.pth')

    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0
    train_losses = []
    valid_losses = []

    for epoch in tqdm(range(config.num_epochs), desc="Training Epochs"):
        
        train_loss = train_utils.train_step(encnet, imnet, train_loader, loss_fn, optimizer, device, config)
        val_loss   = train_utils.eval_step(encnet, imnet, val_loader, loss_fn, device, config)

        train_losses.append(train_loss['total_loss'])
        valid_losses.append(val_loss)

        print(f"Epoch {epoch}: Train Loss {train_loss['total_loss']:.4f} | Valid Loss {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'enc_state_dict': encnet.state_dict(),
                'imnet_state_dict': imnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve > patience:
                print("Early stopping triggered")
                break

    np.save(os.path.join(config.results_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(config.results_dir, 'valid_losses.npy'), valid_losses)

    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.legend()
    plt.savefig(os.path.join(config.results_dir, 'learning_curves.png'))
    plt.close()

    save_config(config, filename="config.yaml", results_dir=config.results_dir)


def test(inp_data_test, out_data_test, results_dir, checkpoint_path=None, plot=True):
    
    config = load_config(filename="config.yaml", results_dir=results_dir)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(results_dir, 'best_model.pth')

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    enc_net, imnet = build_model(config)

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    enc_net.load_state_dict(checkpoint['enc_state_dict'])
    imnet.load_state_dict(checkpoint['imnet_state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_net.to(device)
    imnet.to(device)

    field_reconstructed, field_ground_truth, _ = test_utils.test_step(
        inp_data_test, out_data_test, imnet, enc_net, config, device)

    if plot:
        num_sims = field_reconstructed.shape[0]
        all_pairs = [(sim_nb, comp) for sim_nb in range(num_sims) for comp in range(2)]

        # Randomly select 20 unique pairs
        selected_pairs = random.sample(all_pairs, 20)

        for sim_nb, comp in selected_pairs:
            image = plot_utils.generate_test_images(field_ground_truth, field_reconstructed, sim_nb, comp)
            image.save(os.path.join(results_dir, f'test_image_sim{sim_nb}_comp{comp}.png'))

    test_ssim = plot_utils.compute_ssim_field(field_ground_truth, field_reconstructed)
    print(f"Test SSIM: {test_ssim:.4f}")
    with open(os.path.join(results_dir, 'test_metrics.txt'), 'a') as f:
        f.write(f"Average SSIM: {test_ssim:.4f}\n")

    return field_reconstructed, field_ground_truth



def numpy_representer(dumper, data):
    return dumper.represent_sequence("!numpy", data.tolist())

def numpy_constructor(loader, node):
    return np.array(loader.construct_sequence(node))

# Register custom handlers
yaml.add_representer(np.ndarray, numpy_representer)
yaml.add_constructor("!numpy", numpy_constructor)

def save_config(args, filename="config.yaml", results_dir="results"):
    """
    Save argparse.Namespace to a YAML config file, excluding specific keys.
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    # Exclude 'test_ids' (or any other unwanted keys)
    config_dict = {k: v for k, v in vars(args).items() if k != 'test_ids'}

    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f)

    print(f"Configuration saved to {filepath}.")

def load_config(filename="config.yaml", results_dir="results"):
    """
    Load YAML config file and convert it to argparse.Namespace.
    """
    filepath = os.path.join(results_dir, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Config file not found at {filepath}")

    with open(filepath, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return argparse.Namespace(**config_dict)