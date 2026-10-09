import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import sys
import json
import numpy as np
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import create_data_loaders
from models.plain_cnn import PlainCNN
from models.plain_cnn_with_inflow import PlainCNNWithInflow

def model_count(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )
def synchronize_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def parse_args():
    parser = argparse.ArgumentParser(description='Train wind forecasting model')
    parser.add_argument('--model', type=str, choices=['inflow'], default='inflow',
                        help='Model type: baseline (no inflow) or inflow (two-branch)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 500)')
    parser.add_argument('--epochs', type=int, default=6000,
                        help='Maximum number of epochs (default: 6000)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--early_stop_patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--lr_patience', type=int, default=50,
                        help='LR scheduler patience (default: 50)')
    parser.add_argument('--lr_factor', type=float, default=0.8,
                        help='LR reduction factor (default: 0.8)')
    parser.add_argument('--lr_cooldown', type=int, default=100,
                        help='LR scheduler cooldown (default: 100)')
    parser.add_argument('--min_lr', type=float, default=5e-5,
                        help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--seed', type=int, default=10,
                        help='Random seed (default: 42)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for logs (default: ./logs)')
    parser.add_argument('--checkpoint_dir', type=str, default='./outputs/checkpoints/seed10',
                        help='Directory for model checkpoints (default: ./outputs/checkpoints)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory for output files (default: ./outputs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(model_type, param_dim=1):
    if model_type == 'baseline':
        model = PlainCNN(param_dim=param_dim)
        print(f"Created baseline model with param_dim={param_dim}")
    else:
        model = PlainCNNWithInflow(param_dim=param_dim)
        print(f"Created inflow model with param_dim={param_dim}")
    return model


def train_epoch(model, train_loader, optimizer, criterion, device, model_type):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        if model_type == 'baseline':
            wind_t, inflow, status, wind_tp1 = batch
            # Baseline does not use inflow
            wind_t = wind_t.to(device)
            status = status.to(device)
            wind_tp1 = wind_tp1.to(device)
            optimizer.zero_grad()
            pred = model(wind_t, status)
            loss = criterion(pred, wind_tp1)
        else:
            wind_t, inflow, status, wind_tp1 = batch
            wind_t = wind_t.to(device)
            inflow = inflow.to(device)
            status = status.to(device)
            wind_tp1 = wind_tp1.to(device)
            optimizer.zero_grad()
            pred = model(wind_t, inflow, status)
            loss = criterion(pred, wind_tp1)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(model, val_loader, criterion, device, model_type):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if model_type == 'baseline':
                wind_t, inflow, status, wind_tp1 = batch
                wind_t = wind_t.to(device)
                status = status.to(device)
                wind_tp1 = wind_tp1.to(device)
                pred = model(wind_t, status)
                loss = criterion(pred, wind_tp1)
            else:
                wind_t, inflow, status, wind_tp1 = batch
                wind_t = wind_t.to(device)
                inflow = inflow.to(device)
                status = status.to(device)
                wind_tp1 = wind_tp1.to(device)
                pred = model(wind_t, inflow, status)
                loss = criterion(pred, wind_tp1)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, stats,test_dataset = create_data_loaders(
        batch_size=args.batch_size,
        shuffle_train=True,
        num_workers=0
    )
    print(f"Training samples: {stats['train_samples']}")
    print(f"Validation samples: {stats['val_samples']}")
    print(f"Test samples: {stats['test_samples']}")

    # Create model
    param_dim = stats['N_turbine']
    model = create_model(args.model, param_dim=param_dim)
    model.to(device)

    model_name_map = {
        'baseline': 'plain_cnn',
        'inflow': 'plain_cnn_with_inflow',
    }

    model_name = model_name_map[args.model]

    parameter_count = model_count(model)

    print(f"Model parameters: {parameter_count:,}")
    print(f"Model parameters (M): {parameter_count / 1e6:.3f} M")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        cooldown=args.lr_cooldown,
        min_lr=args.min_lr

    )

    # Resume training if checkpoint provided
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    def save_checkpoint(
            model,
            optimizer,
            epoch,
            checkpoint_path,
    ):


        checkpoint = {
            'epoch': epoch + 1,

            'model_name': model_name,
            'model_type': args.model,

            'model_state_dict': (
                model.state_dict()
            ),

            'optimizer_state_dict': (
                optimizer.state_dict()
            ),

            'scheduler_state_dict': (
                scheduler.state_dict()
            ),

            'best_val_loss': best_val_loss,

            'epochs_no_improve': (
                epochs_no_improve
            ),

            'args': vars(args),

            'stats': stats,
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        print(
            f"Checkpoint saved to "
            f"{checkpoint_path}"
        )

    if args.resume:
        if os.path.isfile(args.resume):

            checkpoint = torch.load(
                args.resume,
                map_location=device,
            )

            start_epoch = checkpoint['epoch']

            model.load_state_dict(
                checkpoint['model_state_dict'],
                strict=True,
            )

            optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )

            scheduler.load_state_dict(
                checkpoint['scheduler_state_dict']
            )

            # Ensure the current command-line min_lr is effective,
            # even when an old scheduler state is resumed.
            scheduler.min_lrs = [
                args.min_lr
                for _ in optimizer.param_groups
            ]

            best_val_loss = checkpoint.get(
                'best_val_loss',
                float('inf'),
            )

            epochs_no_improve = checkpoint.get(
                'epochs_no_improve',
                0,
            )

            print(
                f"Resumed from epoch "
                f"{start_epoch}"
            )
            #------------------------------------#
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Checkpoint {args.resume} not found. Starting from scratch.")

    # Logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'train_{args.model}_{timestamp}.log')
    with open(log_file, 'w') as f:
        f.write(f"Training started at {timestamp}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Device: {device}\n")
        f.write("-" * 50 + "\n")

    # Training loop
    print("Starting training...")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.model)
        val_loss = validate(model, val_loader, criterion, device, args.model)
        epoch_time = time.time() - epoch_start

        # Update learning rate
        scheduler.step(val_loss)

        # Logging
        log_line = f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        print(log_line)
        with open(log_file, 'a') as f:
            f.write(log_line + "\n")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0


            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'{model_name}_best.pt',
            )

            save_checkpoint(
                model,
                optimizer,
                epoch,
                checkpoint_path,
            )

            print(f"  -> Best model saved to {checkpoint_path}")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break


        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                (
                    f'{model_name}_'
                    f'epoch{epoch + 1}.pt'
                ),
            )

            save_checkpoint( model,optimizer,epoch,checkpoint_path,)
    training_time = time.time() - start_time
    print(
        f"Training time for {args.model}: "
        f"{training_time:.2f} s "
        f"({training_time / 60:.2f} min, "
        f"{training_time / 3600:.2f} h)"
    )


    final_path = os.path.join(
        args.checkpoint_dir,
        f'{model_name}_final.pt',
    )

    save_checkpoint( model,optimizer,epoch,final_path, )

    print(f"Final model saved to " f"{final_path}")
    # print(f"Final model saved to {final_path}")

    # Evaluate on test set
    test_loss = validate(model, test_loader, criterion, device, args.model)
    print(f"Test Loss: {test_loss:.6f}")
    with open(log_file, 'a') as f:
        f.write(f"Test Loss: {test_loss:.6f}\n")

    # Save training summary
    # Convert tensor stats to serializable
    serializable_stats = {}
    for key, value in stats.items():
        if torch.is_tensor(value):
            serializable_stats[key] = value.cpu().tolist()
        else:
            serializable_stats[key] = value
    summary = {
        'model': args.model,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'final_epoch': epoch + 1,
        'total_epochs': epoch + 1 - start_epoch,
        'early_stopped': epochs_no_improve >= args.early_stop_patience,
        'args': vars(args),
        'stats': serializable_stats
    }
    summary_path = os.path.join(args.output_dir, f'summary_{args.model}_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()