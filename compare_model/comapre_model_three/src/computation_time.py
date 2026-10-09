import os
import sys
import time
import random
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import create_data_loaders
from utils import set_seed
from models.plain_cnn import PlainCNN
from models.fno import FNO2d


# ============================================================
# Basic settings
# ============================================================

SEED = 42

# Number of recursive steps used for timing
ROLLOUT_STEPS = 100

# Number of complete rollout runs used for GPU warm-up
WARMUP_RUNS = 5

# Number of repeated timing runs
REPEAT_RUNS = 20

# Data
WIND_PATH = './data/wind.pt'
PARAM_PATH = './data/aa.pt'

# Checkpoints
CHECKPOINT_DIR = './outputs/checkpoints/seed42'


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(model, checkpoint_path, device):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=False
        )
    else:
        model.load_state_dict(
            checkpoint,
            strict=False
        )

    return model


# ============================================================
# Count trainable parameters
# ============================================================

def count_trainable_parameters(model):
    """
    Count the number of trainable parameters.
    """
    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return num_params


# ============================================================
# Select one test initial condition
# ============================================================

def select_test_sample(dataset_test):

    # Use the first test case
    case_idx = 0

    # Use the first available time index in the test split
    start_t = int(dataset_test.test_idx[0])

    start_x = dataset_test.wind[
        case_idx,
        start_t
    ].unsqueeze(0)

    start_p = dataset_test.params[
        case_idx
    ].unsqueeze(0)

    print(
        f"Timing sample: case={case_idx}, "
        f"time index={start_t}"
    )

    return start_x, start_p


# ============================================================
# Measure recursive inference time
# ============================================================

def measure_rollout_time(
    model,
    start_x,
    start_p,
    x_mean,
    x_std,
    p_mean,
    p_std,
    device,
    rollout_steps=100,
    warmup_runs=5,
    repeat_runs=20,
):
    model.eval()

    # --------------------------------------------------------
    # Move data and normalization statistics to GPU
    # before timing
    # --------------------------------------------------------

    start_x = start_x.to(
        device=device,
        dtype=torch.float32
    )

    start_p = start_p.to(
        device=device,
        dtype=torch.float32
    )

    x_mean = x_mean.to(
        device=device,
        dtype=torch.float32
    ).view(1, -1, 1, 1)

    x_std = x_std.to(
        device=device,
        dtype=torch.float32
    ).view(1, -1, 1, 1)

    p_mean = p_mean.to(
        device=device,
        dtype=torch.float32
    )

    p_std = p_std.to(
        device=device,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Normalize input before timing
    # --------------------------------------------------------

    x0 = (start_x - x_mean) / x_std
    p0 = (start_p - p_mean) / p_std

    # --------------------------------------------------------
    # GPU warm-up
    # --------------------------------------------------------

    print(
        f"Warm-up: {warmup_runs} runs "
        f"x {rollout_steps} rollout steps"
    )

    with torch.inference_mode():

        for _ in range(warmup_runs):

            x = x0.clone()

            for _ in range(rollout_steps):
                x = model(x, p0)

    # Make sure all CUDA operations have finished
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    total_times = []

    print(
        f"Timing: {repeat_runs} runs "
        f"x {rollout_steps} rollout steps"
    )

    with torch.inference_mode():

        for run_idx in range(repeat_runs):

            # Reset initial condition
            x = x0.clone()

            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            # Recursive rollout
            for _ in range(rollout_steps):
                x = model(x, p0)


            if device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            elapsed_time = end_time - start_time

            total_times.append(elapsed_time)

            print(
                f"Run {run_idx + 1:02d}: "
                f"{elapsed_time:.6f} s"
            )

    total_times = np.asarray(
        total_times,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_total_time = np.mean(total_times)

    if repeat_runs > 1:
        std_total_time = np.std(
            total_times,
            ddof=1
        )
    else:
        std_total_time = 0.0

    # Average time per rollout step
    step_times = total_times / rollout_steps

    mean_step_time = np.mean(step_times)

    if repeat_runs > 1:
        std_step_time = np.std(
            step_times,
            ddof=1
        )
    else:
        std_step_time = 0.0

    return (
        mean_total_time,
        std_total_time,
        mean_step_time,
        std_step_time,
        total_times,
    )


# ============================================================
# Print model results
# ============================================================

def evaluate_model_computation_time(
    model_name,
    model,
    start_x,
    start_p,
    x_mean,
    x_std,
    p_mean,
    p_std,
    device,
):
    """
    Count model parameters and measure recursive inference time.
    """

    print("\n")
    print("=" * 70)
    print(f"Model: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Number of parameters
    # --------------------------------------------------------

    num_params = count_trainable_parameters(model)

    num_params_m = num_params / 1e6

    print(
        f"Trainable parameters: "
        f"{num_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{num_params_m:.4f} M"
    )

    # --------------------------------------------------------
    # Computational time
    # --------------------------------------------------------

    (
        mean_total_time,
        std_total_time,
        mean_step_time,
        std_step_time,
        total_times,
    ) = measure_rollout_time(
        model=model,
        start_x=start_x,
        start_p=start_p,
        x_mean=x_mean,
        x_std=x_std,
        p_mean=p_mean,
        p_std=p_std,
        device=device,
        rollout_steps=ROLLOUT_STEPS,
        warmup_runs=WARMUP_RUNS,
        repeat_runs=REPEAT_RUNS,
    )

    print("\nResults")
    print("-" * 70)

    print(
        f"{ROLLOUT_STEPS}-step rollout time:"
    )

    print(
        f"{mean_total_time:.6f} "
        f"+/- {std_total_time:.6f} s"
    )

    print(
        f"\nAverage computational time "
        f"per rollout step:"
    )

    print(
        f"{mean_step_time:.8f} "
        f"+/- {std_step_time:.8f} s"
    )

    print(
        f"{mean_step_time * 1000:.4f} "
        f"+/- {std_step_time * 1000:.4f} ms"
    )

    return {
        'model': model_name,
        'parameters': num_params,
        'parameters_M': num_params_m,
        'mean_total_time_s': mean_total_time,
        'std_total_time_s': std_total_time,
        'mean_step_time_s': mean_step_time,
        'std_step_time_s': std_step_time,
    }


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print("=" * 70)
    print("Computational efficiency evaluation")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if device.type == 'cuda':

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(device)}"
        )

        print(
            f"CUDA version: "
            f"{torch.version.cuda}"
        )

    print(
        f"PyTorch version: "
        f"{torch.__version__}"
    )

    print(
        f"Rollout steps: "
        f"{ROLLOUT_STEPS}"
    )

    print(
        f"Warm-up runs: "
        f"{WARMUP_RUNS}"
    )

    print(
        f"Timing repetitions: "
        f"{REPEAT_RUNS}"
    )

    # --------------------------------------------------------
    # Load dataset
    #
    # Data-loading time is NOT included in inference timing.
    # --------------------------------------------------------

    print("\nLoading dataset...")

    (
        _,
        _,
        test_loader,
        ds_train,
        ds_val,
        ds_test,
        wind_mean,
        wind_std,
        param_mean,
        param_std,
    ) = create_data_loaders(
        WIND_PATH,
        PARAM_PATH,
        batch_size=1,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Select one fixed test input
    # --------------------------------------------------------

    start_x, start_p = select_test_sample(
        ds_test
    )

    print(
        f"Input wind shape: "
        f"{tuple(start_x.shape)}"
    )

    print(
        f"Parameter shape: "
        f"{tuple(start_p.shape)}"
    )

    # --------------------------------------------------------
    # Model configurations
    #
    # Must be exactly the same as training configuration.
    # --------------------------------------------------------

    model_configs = {

        'plain_cnn': {
            'class': PlainCNN,
            'args': {
                'param_dim': 3,
                'channels': [
                    64,
                    128,
                    128,
                    64,
                    3,
                ],
            },
        },

        'fno': {
            'class': FNO2d,
            'args': {
                'in_channels': 3,
                'param_dim': 3,
                'width': 64,
                'modes_h': 16,
                'modes_w': 12,
                'n_layers': 4,
                'padding_h': 8,
                'padding_w': 4,
                'use_grid': True,
                'projection_width': 128,
            },
        },
    }

    # --------------------------------------------------------
    # Plain CNN
    # --------------------------------------------------------

    print("\nLoading Plain CNN...")

    plain_cnn = model_configs[
        'plain_cnn'
    ]['class'](
        **model_configs[
            'plain_cnn'
        ]['args']
    ).to(device)

    plain_cnn = load_checkpoint(
        plain_cnn,
        os.path.join(
            CHECKPOINT_DIR,
            'plain_cnn_best.pt'
        ),
        device,
    )

    # --------------------------------------------------------
    # FNO
    # --------------------------------------------------------

    print("Loading FNO...")

    fno = model_configs[
        'fno'
    ]['class'](
        **model_configs[
            'fno'
        ]['args']
    ).to(device)

    fno = load_checkpoint(
        fno,
        os.path.join(
            CHECKPOINT_DIR,
            'fno_best.pt'
        ),
        device,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    models = {
        'Plain CNN': plain_cnn,
        'FNO': fno,
    }

    results = []

    for model_name, model in models.items():

        result = evaluate_model_computation_time(
            model_name=model_name,
            model=model,
            start_x=start_x,
            start_p=start_p,
            x_mean=wind_mean,
            x_std=wind_std,
            p_mean=param_mean,
            p_std=param_std,
            device=device,
        )

        results.append(result)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("Final summary")
    print("=" * 90)

    print(
        f"{'Model':<20}"
        f"{'Parameters (M)':>20}"
        f"{'100-step time (s)':>25}"
        f"{'Time/step (ms)':>20}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['model']:<20}"
            f"{result['parameters_M']:>20.4f}"
            f"{result['mean_total_time_s']:>25.6f}"
            f"{result['mean_step_time_s'] * 1000:>20.4f}"
        )

    print("=" * 90)


# ============================================================
# Run
# ============================================================

if __name__ == '__main__':
    main()