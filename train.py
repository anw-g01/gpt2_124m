import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from itertools import cycle     # creates an infinitely looping cycle over an iterator
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from dataset import TinyShakespeare
from config import *    # import all global variables (all caps)
from model import GPT2_124M, GPT2Config

# Ensure a GPU is available and properly configured.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {DEVICE.type.upper()}")
# assert DEVICE.type == "cuda", "DEVICE must be 'cuda' to run the training script."

def train(model: torch.nn.Module) -> tuple:
    """
    Train a PyTorch model using gradient accumulation, mixed precision, and cosine decay learning rate scheduling.
    Selected hyperparameters are close to GPT-2 and GPT-3 model choices by OpenAI, released in any papers.
    - The function uses gradient accumulation with mini-batch processing, if GPU memory is constrained.
    - Mixed precision training (with `bfloat16`) is implemented for efficiency with modern GPUs.
    - Learning rate scheduling includes a linear warm-up phase (to `LEARNING_RATE`) followed by a cosine decay rate.
    - Validation is performed periodically based on `VAL_INTERVAL` (see `config.py`).
    -----
    Args:
        `model` (`torch.nn.Module`): The PyTorch model to be trained. Assumes compatibility with a `CUDA` device.
    Returns:
        `tuple`: A tuple containing:
            - `model` (`torch.nn.Module`): The trained model with updated weights.
            - `train_losses` (`np.ndarray`): Array of training losses recorded at each iteration.
            - `val_losses` (`np.ndarray`): Array of validation losses recorded every `VAL_INTERVAL` iterations.
            - `learning_rates` (`np.ndarray`): Learning rate values tracked at each iteration.
    
    """

    torch.manual_seed(2001)     # for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2001)

    torch.set_float32_matmul_precision("high")     # set global tensor dtype as TensorFloat32 

    # ---------- LOAD DATA ---------- # 

    print(f"\nloading data...\n")
    if CHUNK_SAMPLING:
        train_loader = DataLoader(
            TinyShakespeare(
                BLOCK_SIZE,
                pct=PCT_DATA,
                train_split=TRAIN_SPLIT,
                batch_size=BATCH_SIZE,
                verbose=False
            ),
            batch_size=None,
            shuffle=True
        )
    else:
        train_loader = DataLoader(
            TinyShakespeare(
                BLOCK_SIZE,
                pct=PCT_DATA,
                train_split=TRAIN_SPLIT,
                verbose=False
            ),
            batch_size=BATCH_SIZE,
            shuffle=True
        )
    val_loader = DataLoader(
        TinyShakespeare(
            BLOCK_SIZE,
            pct=PCT_DATA,
            train_split=TRAIN_SPLIT,
            split="val",
        ),
        batch_size=BATCH_SIZE,      # no chunk sampling used as only VAL_ACCUM_STEPS batches used 
        shuffle=False
    )
    val_iter = cycle(iter(val_loader))          # infinite iterator over val_loader
    train_iter = cycle(iter(train_loader))      # infinite iterator over train_loader
    print(f"chunk sampling for training: {CHUNK_SAMPLING}")

    print("\n*------------ TRAINING ------------*")
    print(f"training tokens per batch: {TOKENS_PER_BATCH:,}")
    print(f"mini-batch size: {BATCH_SIZE} ({GRAD_ACCUM_STEPS} accumulation steps)")
    batches_per_epoch_train = int(math.ceil(len(train_loader) / GRAD_ACCUM_STEPS))
    print(f"batches per epoch: {batches_per_epoch_train:,} ({len(train_loader):,} mini-batches)")

    print("\n*------------ VALIDATION ------------*")
    print(f"validation tokens per batch: {BATCH_SIZE * BLOCK_SIZE * VAL_ACCUM_STEPS:,}")
    print(f"mini-batch size: {BATCH_SIZE} ({VAL_ACCUM_STEPS} accumulation steps)")
    batches_per_epoch_val= int(math.ceil(len(val_loader) / VAL_ACCUM_STEPS))
    print(f"batches per epoch: {batches_per_epoch_val:,} ({len(val_loader):,} mini-batches)")

    # ---------- MODEL INSTANCE ---------- #

    print(f"\nloading model, optimiser and scheduler...\n")
    model.to(DEVICE)     # GPT(GPTConfig(vocab_size=50304)) in main.py
    print(f"no. of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimiser = model.configure_optim(WEIGHT_DECAY, LEARNING_RATE, DEVICE.type)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimiser,
        T_max=(MAX_STEPS - WARMUP_STEPS),   # decay starts after warmup
        eta_min=0.1*LEARNING_RATE           # minimum learning rate
    )

    print(f"\ncompiling model...")
    model = torch.compile(model)

    # ---------- MAIN LOOP ---------- # 

    print(f"\nrunning validation every {VAL_INTERVAL} iterations")
    print(f"running {ITERATIONS:,} total iterations...\n")
    train_losses = np.empty(ITERATIONS)
    val_losses = np.full(ITERATIONS, np.nan)    # initialise with NaNs (due to interval usage)
    learning_rates = np.empty(ITERATIONS)

    for i in range(ITERATIONS):

        # ----- TRAINING LOOP ----- #
        t0 = time.time()    # capture starting time
        model.train()
        X, y = next(train_iter)
        X_train, y_train = X.to(DEVICE), y.to(DEVICE)

        # ----- GRADIENT ACCUMULATION ----- #
        optimiser.zero_grad()                                                       # reset gradients
        train_loss = 0                                                              # accumulated train loss
        for _ in range(GRAD_ACCUM_STEPS):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):     # mixed precision
                _, loss = model(X_train, y_train)
            loss /= GRAD_ACCUM_STEPS                                                # scale loss to mimic full total batch average
            train_loss += loss.detach()                                             # prevent carry over of computational graph
            loss.backward()                                                         # accumulate gradients (+=)
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)           # gradient clipping
        optimiser.step()                                                            # update parameters after GRAD_ACCUM_STEPS

        # ----- LEARNING RATE SCHEDULER ----- #
        if i < WARMUP_STEPS:
            lr = LEARNING_RATE * (i + 1) / WARMUP_STEPS      # linear warmup to LEARNING_RATE
            for param_group in optimiser.param_groups:
                param_group["lr"] = lr
        else:
            scheduler.step()                                    # update learning rate after WARM_UPSTEPS
            lr = scheduler.get_last_lr()[0]                     # use param group 0

        # ----- TRACK METRICS ----- #
        torch.cuda.synchronize()    # wait until CUDA operations queued on a GPU are completed before proceeding
        t1 = time.time()
        dt = (t1 - t0) * 1e3        # time difference in miliseconds
        tps = (X.numel() * GRAD_ACCUM_STEPS) / (t1 - t0)    # training tokens/second processed

        # ----- VALIDATION LOOP ----- #
        if i % VAL_INTERVAL == 0:
            t0_val = time.time()
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(VAL_ACCUM_STEPS):
                    X, y = next(val_iter)
                    X_val, y_val = X.to(DEVICE), y.to(DEVICE)
                    _, loss = model(X_val, y_val)
                    val_loss += loss.item() / VAL_ACCUM_STEPS    # equivalent to val_loss /= VAL_ACCUM_STEPS in the final iteration
            torch.cuda.synchronize()
            t1_val = time.time()
            dt_val = (t1_val - t0_val) * 1e3
            tps_val = (X.numel() * VAL_ACCUM_STEPS) / (t1_val - t0_val)
            val_losses[i] = val_loss
        
        # ----- LOG PROGRESS & STATS ----- #
        learning_rates[i] = lr      # populate arrays for plotting
        train_losses[i] = train_loss.item()
        if i % LOG_INTERVAL == 0:
            pct = (i + 1) / ITERATIONS * 100     # percentage completion
            progress_str = (
                f"\r{i + 1:,}/{ITERATIONS:,} ({pct:.0f}%) | "
                f"train_loss: {train_loss.item():.3f} | "
                f"{dt * 1e-3:,.2f} sec/batch ({dt / GRAD_ACCUM_STEPS:.1f} ms/iter) | "
                f"{tps:,.0f} tok/sec | "
                f"val_loss: {val_loss:.3f} | "
                f"{dt_val * 1e-3:.1f} sec ({tps_val:,.0f} tok/sec)"
            )
            print(progress_str, end="")
    print("\n\nTraining Complete.")
    return model, train_losses, val_losses, learning_rates

def plot_losses(train_losses: np.array, val_losses=None) -> None:
    """
    Plot training and validation losses over iterations during a training run.
    """
    x = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.title("GPT-2 (124M) Training on Tiny Shakespeare (~300K tokens)")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    marker = "." if x.shape[0] <= 50 else None
    plt.plot(
        x, train_losses, linewidth=1,
        label="train loss", marker=marker
    )
    if val_losses is not None:
        idx = np.isfinite(val_losses)     # bool: False if cell is NaN
        plt.plot(
            x[idx], val_losses[idx], linewidth=1,
            label="val loss", marker="."
        )
    plt.legend()
    plt.show()

def plot_lr(learning_rates: np.array):
    """
    Plot learning rates over a training run.
    """
    steps = np.arange(1, len(learning_rates) + 1)
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.title(f"Learning Rate Schedule (Cosine Decay)")
    plt.xlabel("iteration")
    plt.ylabel("learning rate")
    marker = "." if steps.shape[0] <= 50 else None
    plt.plot(
        steps, learning_rates,
        linewidth=1, color="tab:olive",
        label="learning rate", marker=marker
    )
    plt.legend()
    plt.show()


    
