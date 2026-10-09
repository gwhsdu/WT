import torch
import numpy as np
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import create_data_loaders
from utils import set_seed, ensure_dir, create_param_map, concatenate_wind_param
from models.plain_cnn import PlainCNN
from models.plain_cnn_with_inflow import PlainCNNWithInflow



def select_rollout_indices(dataset_test, num_samples=100, rollout_steps=120, seed=42):
    """
    Randomly select starting points from test dataset that allow full rollout.
    Returns list of tuples (case_idx, start_time_idx).
    """
    set_seed(seed)
    # Collect all valid starting indices
    valid_indices = []
    # Need to know time indices for test split
    split_idx = sorted(set(t for _, t in dataset_test.indices))
    split_set = set(split_idx)
    for case in range(dataset_test.N_case):
        for t in split_idx:
            # Check that t, t+1, ..., t+rollout_steps-1 are all in test split
            valid = all((t + k) in split_set for k in range(rollout_steps+1))
            if valid:
                valid_indices.append((case, t))
    # Random sample
    if len(valid_indices) < num_samples:
        print(f"Warning: only {len(valid_indices)} valid starting points, using all.")
        selected = valid_indices
    else:
        selected = random.sample(valid_indices, num_samples)
    return selected





# def recursive_predict(model, start_x, start_p, steps=100, device='cpu'):
#     """
#     Perform recursive prediction using model.
#     Args:
#         model: trained model
#         start_x: initial wind field (1, 3, L, W)
#         start_p: parameter (1, N_turbine)
#         steps: number of steps to rollout
#         device: device
#     Returns:
#         predictions: tensor (steps, 3, L, W) (including step1 prediction)
#     """
#     model.eval()
#     x = start_x.clone()
#     preds = []
#     with torch.no_grad():
#         for _ in range(steps):
#             y_pred = model(x.to(device), start_p.to(device))
#             preds.append(y_pred.squeeze(0).cpu())
#             x = y_pred  # use prediction as next input
#     return torch.stack(preds, dim=0)  # (steps, 3, L, W)

def recursive_predict(model, start_x, start_p,steps=120, device='cpu'):
    """
    Perform recursive prediction using model.
    Args:
        model: trained model
        start_x: initial wind field (1, 3, L, W)
        start_p: parameter (1, N_turbine)
        steps: number of steps to rollout
        device: device
    Returns:
        predictions: tensor (steps, 3, L, W) (including step1 prediction)
    """
    x_mean = torch.load("./outputs/data_norm/wind_mean.pt")
    x_std = torch.load("./outputs/data_norm/wind_std.pt")
    p_mean = torch.load("./outputs/data_norm/param_mean.pt")
    p_std = torch.load("./outputs/data_norm/param_std.pt")
    x_mean = x_mean.view(1, -1, 1, 1)
    x_std = x_std.view(1, -1, 1, 1)
    model.eval()
    x = start_x.clone()
    x = (x - x_mean) / x_std
    start_p = (start_p-p_mean)/p_std
    preds = []
    with torch.no_grad():
        for i in range(steps):
            y_pred = model(x.to(device), start_p.to(device))
            preds.append(y_pred.squeeze(0).cpu())
            x = y_pred  # use prediction as next input
    return torch.stack(preds, dim=0)  # (steps, 3, L, W)


def recursive_predict_inflow(model, start_x, start_p,inflow, steps=120, device='cpu'):
    """
    Perform recursive prediction using model.
    Args:
        model: trained model
        start_x: initial wind field (1, 3, L, W)
        start_p: parameter (1, N_turbine)
        steps: number of steps to rollout
        device: device
    Returns:
        predictions: tensor (steps, 3, L, W) (including step1 prediction)
    """
    x_mean = torch.load("./outputs/data_norm/wind_mean.pt")
    x_std = torch.load("./outputs/data_norm/wind_std.pt")
    p_mean = torch.load("./outputs/data_norm/param_mean.pt")
    p_std = torch.load("./outputs/data_norm/param_std.pt")
    x_mean = x_mean.view(1, -1, 1, 1)
    x_std = x_std.view(1, -1, 1, 1)
    model.eval()
    x = start_x.clone()
    x = (x - x_mean) / x_std
    start_p = (start_p-p_mean)/p_std
    preds = []
    with torch.no_grad():
        for i in range(steps):
            inflow_input = (inflow[i]-x_mean) / x_std
            # inflow_input = (inflow[i+1] - inflow[i]) / x_std
            y_pred = model(x.to(device), inflow_input.to(device),start_p.to(device))
            preds.append(y_pred.squeeze(0).cpu())
            x = y_pred  # use prediction as next input
    return torch.stack(preds, dim=0)  # (steps, 3, L, W)

# --------------- 新增噪声 ------------------- #
def get_relative_inflow_error(
    horizon,
    initial_error=0.02,
    maximum_error=0.20,
    saturation_horizon=100,
):
    """
    Return the relative inflow forecast error at a given horizon.

    Args:
        horizon:
            Forecast horizon h, starting from 1.

        initial_error:
            Relative standard deviation at h=1.

        maximum_error:
            Maximum relative standard deviation.

        saturation_horizon:
            Horizon at which the error reaches maximum_error.
    """

    if horizon < 1:
        raise ValueError(
            f"horizon must be >= 1, got {horizon}"
        )

    if saturation_horizon <= 1:
        raise ValueError(
            "saturation_horizon must be greater than 1"
        )

    growth_fraction = min(
        (horizon - 1)
        / (saturation_horizon - 1),
        1.0,
    )

    relative_error = (
        initial_error
        + (maximum_error - initial_error)
        * growth_fraction
    )

    return relative_error

def recursive_predict_inflow_noise(
    model,
    start_x,
    start_p,
    inflow,
    inflow_mean,
    steps=120,
    device='cpu',
    delta_t=1.0,
    error_time_scale=10.0,
):
    """
    Recursive prediction with horizon-dependent,
    temporally correlated Gaussian inflow errors.

    Perturbed future inflow:

        I_tilde(t+i, c)
        =
        I_true(t+i, c)
        + abs(I_mean(c)) * r(i) * z(i, c)

    where r(i) increases from 2% to 20%, and z(i, c)
    follows an AR(1) process.
    """

    if steps > inflow.shape[0]:
        raise ValueError(
            f"Requested {steps} steps, but only "
            f"{inflow.shape[0]} inflow steps are available"
        )

    if error_time_scale <= 0.0:
        raise ValueError(
            "error_time_scale must be positive"
        )

    x_mean = torch.load(
        "./outputs/data_norm/wind_mean.pt"
    )

    x_std = torch.load(
        "./outputs/data_norm/wind_std.pt"
    )

    p_mean = torch.load(
        "./outputs/data_norm/param_mean.pt"
    )

    p_std = torch.load(
        "./outputs/data_norm/param_std.pt"
    )

    x_mean = x_mean.view(1, -1, 1, 1)
    x_std = x_std.view(1, -1, 1, 1)

    model.eval()

    # Normalize initial wind field and turbine parameters.
    x = (
        start_x.clone()
        - x_mean
    ) / x_std

    start_p = (
        start_p
        - p_mean
    ) / p_std

    # Component-wise error scales:
    #
    # S_c = abs(mean inflow of component c)
    #
    # Shape: (1, 3, 1, 1)
    inflow_scale = (
        inflow_mean
        .view(1, -1, 1, 1)
        .abs()
        .to(
            dtype=inflow.dtype,
            device=inflow.device,
        )
    )

    # AR(1) correlation coefficient.
    rho = float(
        np.exp(
            -delta_t / error_time_scale
        )
    )

    innovation_scale = np.sqrt(
        1.0 - rho ** 2
    )

    # z_1,c ~ N(0, 1)
    #
    # Three velocity components use three independent
    # temporally correlated random processes.
    #
    # Shape: (1, 3, 1, 1)
    z = torch.randn(
        1,
        inflow.shape[1],
        1,
        1,
        dtype=inflow.dtype,
        device=inflow.device,
    )

    preds = []

    with torch.no_grad():
        for i in range(steps):
            # Python index i starts at 0.
            # Forecast horizon starts at 1.
            horizon = i + 1

            # Update AR(1) process from horizon 2.
            if horizon > 1:
                innovation = torch.randn_like(z)

                z = (
                    rho * z
                    + innovation_scale * innovation
                )

            # r(i): 2% at horizon 1, increasing linearly
            # to 20% at horizon 100.
            relative_error = (
                get_relative_inflow_error(
                    horizon=horizon,
                    initial_error=0.02,
                    maximum_error=0.20,
                    saturation_horizon=100,
                )
            )

            # sigma_i,c = abs(I_mean_c) * r(i)
            error_std = (
                inflow_scale
                * relative_error
            )

            # epsilon_i,c = sigma_i,c * z_i,c
            #
            # Shape: (1, 3, 1, 1)
            inflow_error = (
                error_std * z
            )

            # True future inflow:
            #
            # inflow[i]: (3, 1, W)
            # after unsqueeze: (1, 3, 1, W)
            true_inflow = (
                inflow[i]
                .unsqueeze(0)
            )

            # Add error in physical scale.
            #
            # The error is automatically broadcast along W.
            #
            # Shape: (1, 3, 1, W)
            perturbed_inflow = (
                true_inflow
                + inflow_error
            )

            # Normalize only after adding the error.
            inflow_input = (
                perturbed_inflow
                - x_mean
            ) / x_std

            y_pred = model(
                x.to(device),
                inflow_input.to(device),
                start_p.to(device),
            )

            preds.append(
                y_pred.squeeze(0).cpu()
            )

            # Recursive prediction.
            x = y_pred

    return torch.stack(
        preds,
        dim=0,
    )


# def run_rollout_for_model(model_name, model, indices, dataset_test, device,
#                           rollout_steps=120, save_dir='./outputs/predictions'):
def run_rollout_for_model(
        model_name,
        model,
        indices,
        dataset_test,
        device,
        rollout_steps=120,
        save_dir='./outputs/predictions',
        inflow_mean=None,
):
    """
    Run rollout for a specific model.
    Saves predictions as tensor of shape (100, rollout_steps, 3, L, W).
    """
    if (
            model_name == 'plain_cnn_with_inflow_noise'
            and inflow_mean is None
    ):
        raise ValueError(
            "inflow_mean must be provided for "
            "plain_cnn_with_inflow_noise"
        )
    # if model_name ==  'plain_cnn_with_inflow_noise':
    #     inflow_total = torch.load('./data/wind.pt')
    #     inflow_total = inflow_total.reshape(-1,3,81,33)[:,:,:1,:]
    #     inflow_mean = inflow_total.mean(dim=(0, 2, 3), keepdim=True)
    ensure_dir(save_dir)
    predictions = []
    for case_idx, start_t in indices:
        # Get initial wind field (normalized)
        start_x = dataset_test.wind[case_idx, start_t].unsqueeze(0)  # (1,3,L,W)
        start_p = dataset_test.status[case_idx].unsqueeze(0)         # (1, N_turbine)
        start_inflow = dataset_test.wind[case_idx, start_t+1:start_t+rollout_steps+1,:,:1,:]
        # start_inflow = dataset_test.wind[case_idx, start_t :start_t + rollout_steps + 1, :, :1, :]
        # Recursive predict
        if model_name == 'plain_cnn':
          pred = recursive_predict(model, start_x, start_p, rollout_steps, device)
        elif model_name == 'plain_cnn_with_inflow':
          pred = recursive_predict_inflow(model, start_x, start_p, start_inflow, rollout_steps, device)
        elif model_name ==  'plain_cnn_with_inflow_noise':
          #noise = torch.randn_like(start_inflow) * (0.05 * inflow_mean)
          #start_inflow_noise = start_inflow + noise
          # pred = recursive_predict_inflow_noise(model, start_x, start_p, start_inflow, inflow_mean,rollout_steps, device)
          pred = recursive_predict_inflow_noise(
              model=model,
              start_x=start_x,
              start_p=start_p,
              inflow=start_inflow,
              inflow_mean=inflow_mean,
              steps=rollout_steps,
              device=device,
              delta_t=1.0,
              error_time_scale=10.0,
          )
        else:
          print("No match found model_name.")
        predictions.append(pred)  # (steps, 3, L, W)
    # Stack across samples
    predictions = torch.stack(predictions, dim=0)  # (num_samples, steps, 3, L, W)
    # Denormalize to original scale
    predictions_denorm = dataset_test.denormalize_wind(predictions)
    # Save
    # Save
    if model_name == 'plain_cnn':
        save_path = os.path.join(save_dir, f'Plain CNN_rollout.pt')
    elif model_name == 'plain_cnn_with_inflow':
        save_path = os.path.join(save_dir, f'CNN with inflow_rollout.pt')
    elif model_name == 'plain_cnn_with_inflow_noise':
        save_path = os.path.join(save_dir, f'CNN with perturbed inflow_rollout.pt')
    torch.save(predictions_denorm, save_path)
    print(f"Saved {model_name} predictions to {save_path}")
    return predictions_denorm


# def load_checkpoint(model, checkpoint_path, device):
#     """Load model checkpoint (state_dict)."""
#     #checkpoint = torch.load(checkpoint_path, map_location=device)
#     #if 'model_state_dict' in checkpoint:
#     #    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#     #else:
#     #    # Assume checkpoint is directly state_dict
#     #    model.load_state_dict(checkpoint, strict=False)
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#
#     if 'model_state' in checkpoint:
#         state_dict = checkpoint['model_state']
#     elif 'model_state_dict' in checkpoint:
#         state_dict = checkpoint['model_state_dict']
#     else:
#         state_dict = checkpoint
#
#     model.load_state_dict(state_dict, strict=True)
#     return model


def load_checkpoint(model,checkpoint_path,device,):
    """
    Load a best model checkpoint for rollout.
    """

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path,map_location=device,)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint[
            'model_state_dict'
        ]

    elif 'model_state' in checkpoint:
        # Compatibility with old checkpoints.
        state_dict = checkpoint[
            'model_state'
        ]

    else:
        # Compatibility with a directly saved state_dict.
        state_dict = checkpoint

    model.load_state_dict(state_dict,strict=True,)

    if isinstance(checkpoint, dict):
        checkpoint_epoch = checkpoint.get(
            'epoch',
            'unknown',
        )

        checkpoint_val_loss = checkpoint.get(
            'best_val_loss',
            'unknown',
        )

        print(f"Loaded checkpoint: "f"{checkpoint_path}")

        print(f"Checkpoint epoch: "f"{checkpoint_epoch}")

        print(f"Checkpoint best val loss: " f"{checkpoint_val_loss}" )

    return model

def main():
    set_seed(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    wind_path = './data/wind.pt'
    param_path = './data/aa.pt'
    # Use batch size 1 for data loading (just need dataset)
    # (_, _, test_loader,
    #  status,  ds_test) = create_data_loaders(
    #     batch_size=1)
    ( train_loader,val_loader,test_loader,status,ds_test,) = create_data_loaders(batch_size=1)
    ds_train = train_loader.dataset

    train_inflow_samples = []

    for case_idx, time_idx in ds_train.indices:
        inflow_sample = ds_train.wind[
                        case_idx,
                        time_idx,
                        :,
                        :1,
                        :,
                        ]

        train_inflow_samples.append(
            inflow_sample
        )

    train_inflow = torch.stack(
        train_inflow_samples,
        dim=0,
    )

    # Shape:
    #     train_inflow: (N_train, 3, 1, W)
    #     inflow_mean:  (1, 3, 1, 1)
    inflow_mean = train_inflow.mean(
        dim=(0, 2, 3),
        keepdim=True,
    )

    print(
        "Training inflow component means:",
        inflow_mean.view(-1),
    )

    print(
        "Training inflow error scales:",
        inflow_mean.abs().view(-1),
    )

    # -----------------  新增  ------------------- #
    rollout_steps = 120

    # ------------------------------------------ #


    # Select rollout indices
    # indices = select_rollout_indices(ds_test, num_samples=100, rollout_steps=120, seed=42)
    indices = select_rollout_indices(ds_test,num_samples=100,rollout_steps=rollout_steps,seed=42,)
    # Save indices
    ensure_dir('./outputs/predictions')
    indices_path = './outputs/predictions/rollout_indices.pt'
    torch.save(indices, indices_path)
    print(f"Saved rollout indices to {indices_path}")
    print(f"Selected {len(indices)} starting points.")

    # Load ground truth for these indices
    ground_truth = []
    for case_idx, start_t in indices:
        # Collect wind fields for steps 1..10 (i.e., start_t+1 to start_t+10)
        truth = torch.stack([ds_test.wind[case_idx, start_t + k] for k in range(1, 121)], dim=0)
        ground_truth.append(truth)  # (steps, 3, L, W)
    ground_truth = torch.stack(ground_truth, dim=0)  # (num_samples, steps, 3, L, W)
    #ground_truth_denorm = ds_test.denormalize_wind(ground_truth)
    gt_path = './outputs/predictions/ground_truth_rollout.pt'
    #torch.save(ground_truth_denorm, gt_path)
    torch.save(ground_truth, gt_path)
    print(f"Saved ground truth to {gt_path}")

    # Load trained models with architecture matching training configuration
    checkpoint_dir = './outputs/checkpoints/seed42'
    # Persistence (no checkpoint)
    #persistence = Persistence().to(device)

    # Architecture configurations (must match training config in train.py)
    model_configs = {
        'plain_cnn': {'class': PlainCNN, 'args': {'param_dim': 3, 'channels': [64, 128, 128, 64, 3]}},
        'plain_cnn_with_inflow':{'class': PlainCNNWithInflow, 'args': {'param_dim': 3}},
        'plain_cnn_with_inflow_noise':{'class': PlainCNNWithInflow, 'args': {'param_dim': 3}},
       
    }

    plain_cnn = model_configs['plain_cnn']['class'](**model_configs['plain_cnn']['args']).to(device)
    plain_cnn = load_checkpoint(plain_cnn, os.path.join(checkpoint_dir, 'plain_cnn_best.pt'), device)
    
    plain_cnn_with_inflow = model_configs['plain_cnn_with_inflow']['class'](**model_configs['plain_cnn_with_inflow']['args']).to(device)
    # plain_cnn_with_inflow = load_checkpoint(plain_cnn_with_inflow, os.path.join(checkpoint_dir, 'best_inflow.pt'), device)
    inflow_checkpoint = os.path.join(checkpoint_dir,'plain_cnn_with_inflow_best.pt')

    plain_cnn_with_inflow = load_checkpoint(plain_cnn_with_inflow,inflow_checkpoint,device)
    
   

    # Run rollout for each model
    models = {
        'plain_cnn': plain_cnn,
        'plain_cnn_with_inflow': plain_cnn_with_inflow,
        'plain_cnn_with_inflow_noise': plain_cnn_with_inflow,    
    }
    #models = {
    #    'plain_cnn': plain_cnn,
    #    'unet': unet,
    #    'bicnn_style': bicnn,
    #    'pod_mlp': pod_mlp,
    #}
    for name, model in models.items():
        print(f"\nRollout for {name}...")
        # run_rollout_for_model(name, model, indices, ds_test, device)
        run_rollout_for_model(model_name=name,model=model,indices=indices,dataset_test=ds_test,device=device,rollout_steps=rollout_steps,inflow_mean=inflow_mean,)

    print("\nRollout completed.")


if __name__ == '__main__':
    main()