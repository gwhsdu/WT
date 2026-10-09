import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import create_data_loaders
from utils import set_seed, ensure_dir, save_checkpoint, load_checkpoint
from models.persistence import Persistence
from models.plain_cnn import PlainCNN
from models.unet import UNet
from models.bicnn_style import BiCNNStyle
# from models.pod_mlp import PodMLP
from models.fno import FNO2d

def model_count(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )
def synchronize_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def train_model(model, train_loader, val_loader, device,
                model_name, max_epochs=6000, lr=1e-3, patience=50,
                checkpoint_dir='./outputs/checkpoints/seed42',
                weight_decay=0, scheduler_factor=0.8, scheduler_patience=50,
                scheduler_cooldown=100, min_lr=5e-5):

    ensure_dir(checkpoint_dir)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor,
                                  patience=scheduler_patience, cooldown=scheduler_cooldown, min_lr=min_lr)

    # Loss function
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X, p, Y in train_loader:
            X, p, Y = X.to(device), p.to(device), Y.to(device)
            optimizer.zero_grad()
            Y_pred = model(X, p)
            loss = criterion_mse(Y_pred, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, p, Y in val_loader:
                X, p, Y = X.to(device), p.to(device), Y.to(device)
                Y_pred = model(X, p)
                #loss = criterion_l1(Y_pred, Y) + criterion_mse(Y_pred, Y)
                loss =  criterion_mse(Y_pred, Y)
                val_loss += loss.item() * X.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        if epoch % 10 == 0:
            print(f'Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

    # Load best model
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, optimizer, checkpoint_path)
        print(f'Loaded best checkpoint from {checkpoint_path}')
    else:
        print('Warning: No checkpoint found, using last model.')

    return model, train_losses, val_losses


def prepare_podmlp(model, train_dataset, device):

    wind_list = []
    for case_idx, time_idx in train_dataset.samples:
        wind_list.append(train_dataset.wind[case_idx, time_idx])
    wind_stack = torch.stack(wind_list, dim=0)  # (N_train, 3, L, W)
    # Fit PCA
    model.fit(wind_stack.to(device))
    return model


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data paths
    wind_path = './data/wind.pt'
    param_path = './data/aa.pt'

    # Batch size adjustment
    batch_size = 256
    # Try to allocate dummy tensors to test memory
    try:
        dummy_x = torch.randn(batch_size, 3, 81, 33, device=device)
        dummy_p = torch.randn(batch_size, 1, device=device)
        dummy_y = torch.randn(batch_size, 3, 81, 33, device=device)
        del dummy_x, dummy_p, dummy_y
        torch.cuda.empty_cache()
        print(f"Using batch size {batch_size}")
    except RuntimeError as e:
        print(f"Batch size {batch_size} too large, reducing.")
        # Try smaller sizes
        for bs in [128, 64, 32, 16, 8]:
            try:
                dummy_x = torch.randn(bs, 3, 81, 33, device=device)
                dummy_p = torch.randn(bs, 1, device=device)
                dummy_y = torch.randn(bs, 3, 81, 33, device=device)
                del dummy_x, dummy_p, dummy_y
                torch.cuda.empty_cache()
                batch_size = bs
                print(f"Using batch size {batch_size}")
                break
            except RuntimeError:
                continue
        else:
            print("Cannot allocate even with batch size 8, exiting.")
            return

    # Create data loaders
    (train_loader, val_loader, test_loader,
     ds_train, ds_val, ds_test,
     wind_mean, wind_std, param_mean, param_std) = create_data_loaders(
        wind_path, param_path, batch_size=batch_size, seed=42)

    print(f"Train samples: {len(ds_train)}")
    print(f"Val samples: {len(ds_val)}")
    print(f"Test samples: {len(ds_test)}")

    # Model configurations (hyperparameters and architecture adjustments)
    model_configs = [
        {
            'name': 'plain_cnn',
            'model_class': PlainCNN,
            'model_args': {'param_dim': 1, 'channels': [64, 128, 128, 64, 3]},
            'hyper': {'lr': 1e-3, 'weight_decay': 0, 'max_epochs': 6000, 'patience': 50,
                      'scheduler_factor': 0.8, 'scheduler_patience': 50, 'scheduler_cooldown': 100, 'min_lr': 5e-5}
        },
        {
            'name': 'unet',
            'model_class': UNet,
            'model_args': {'param_dim': 1, 'base_width': 64},
            'hyper': {'lr': 1e-3, 'weight_decay': 0, 'max_epochs': 6000, 'patience': 50,
                      'scheduler_factor': 0.8, 'scheduler_patience': 50, 'scheduler_cooldown': 100, 'min_lr': 5e-5}
        },
        {
            'name': 'bicnn_style',
            'model_class': BiCNNStyle,
            'model_args': {'param_dim': 1, 'bg_channels': [64, 128, 128], 'fg_channels': [64, 128, 128],
                           'fusion_channels': [256, 128, 64, 3]},
            'hyper': {'lr': 1e-3, 'weight_decay': 0, 'max_epochs': 6000, 'patience': 50,
                      'scheduler_factor': 0.8, 'scheduler_patience': 50, 'scheduler_cooldown': 100, 'min_lr': 5e-5}
        },
        {
            'name': 'fno',
            'model_class': FNO2d,
            'model_args': {'in_channels': 3,'param_dim': 1,'width': 64,'modes_h': 16,'modes_w': 12,'n_layers': 4,'padding_h': 8,
                           'padding_w': 4,'use_grid': True,'projection_width': 128,},
            'hyper': {'lr': 1e-3,'weight_decay': 0,'max_epochs': 6000,'patience': 50,'scheduler_factor': 0.8,
                'scheduler_patience': 50,'scheduler_cooldown': 100,'min_lr': 5e-5,}
        },
    ]


    models_to_run = ['plain_cnn','unet','bicnn_style','fno']

    model_configs = [c for c in model_configs if c['name'] in models_to_run]

    # Train each model with its configuration
    training_summary = []
    for config in model_configs:
        name = config['name']
        print(f"\n=== Training {name} with config ===")
        # Instantiate model
        model = config['model_class'](**config['model_args']).to(device)
        parameter_count = model_count(model)

        print(f"Model parameters: {parameter_count:,}")
        print(f"Model parameters (M): {parameter_count / 1e6:.3f} M")
        # if name == 'pod_mlp':
        #     model = prepare_podmlp(model, ds_train, device)
        
        start_time = time.time()
        # Train with hyperparameters
        trained_model, train_loss, val_loss = train_model(
            model, train_loader, val_loader, device,
            model_name=name, **config['hyper'])


        training_time = time.time() - start_time

        print(
            f"Training time for {name}: "
            f"{training_time:.2f} s "
            f"({training_time / 60:.2f} min, "
            f"{training_time / 3600:.2f} h)"
        )
        training_summary.append({
            'name': name,
            'parameters': parameter_count,
            'epochs': len(train_loss),
            'training_seconds': training_time,
            'best_val_loss': min(val_loss) if val_loss else float('nan'),
        })


        checkpoint_path = f'./outputs/checkpoints/seed42/{name}_final.pt'
        torch.save(trained_model.state_dict(), checkpoint_path)
        print(f"Saved final model to {checkpoint_path}")


    print("\nTraining completed.")
    print("\n=== Model Training Summary ===")
    print(
        f"{'Model':<16}"
        f"{'Parameters':>16}"
        f"{'Params(M)':>12}"
        f"{'Epochs':>10}"
        f"{'Time(min)':>12}"
        f"{'Best Val':>14}"
    )

    for result in training_summary:
        print(
            f"{result['name']:<16}"
            f"{result['parameters']:>16,}"
            f"{result['parameters'] / 1e6:>12.3f}"
            f"{result['epochs']:>10}"
            f"{result['training_seconds'] / 60:>12.2f}"
            f"{result['best_val_loss']:>14.6f}"
        )


if __name__ == '__main__':
    main()