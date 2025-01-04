import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from itertools import cycle     # creates an infinitely looping cycle over an iterator
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from dataset import TinyShakespeare
from config import *    # import all global variables (all in caps)
import os
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def initialise_ddp() -> tuple:
    """
    Set up DDP (Distributed Data Parallel) with `torch.distributed` to utilise multi-GPU training.
    The `torchrun` command will set the `env` variables: `RANK`, `LOCAL_RANK` and `WORLD_SIZE`.
    """
    is_ddp = all(key in os.environ for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])    # check if running in a distributed environment
    if is_ddp:
        assert torch.cuda.is_available(), "train.py script cannot be run without CUDA."    # CUDA must be available for DDP
        init_process_group(backend="nccl")                  # initialise the process group
        ddp_rank = int(os.environ["RANK"])                  # global process integer ID (e.g. 0-7 for 8 GPUs)
        ddp_local_rank = int(os.environ["LOCAL_RANK"])      # GPU ID on the current node (e.g. 0-7 if all on one machine)
        ddp_world_size = int(os.environ["WORLD_SIZE"])      # total no. of processes (i.e. no of GPUs)
        device = f"cuda:{ddp_local_rank}"                   # select appropriated GPU based on integer IDw
        torch.cuda.set_device(device)                       # set the device for current process 
        master_process = (ddp_rank == 0)                    # flag for the first GPU (for logging, checkpointing etc.) 
        print(f"using DDP with WORLD_SIZE: {ddp_world_size}\n")
    else:
        ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1  # fallback for non-DDP setup
        master_process = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using single device: {device}")
    return ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


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

    # get distributed parameters from environment variables (if using DDP)
    DDP_RANK, DDP_LOCAL_RANK, DDP_WORLD_SIZE, DEVICE, MASTER_PROCESS = initialise_ddp()

    torch.manual_seed(2001)                         # for consistent intantiations of models across all processes
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2001)

    torch.set_float32_matmul_precision("high")      # set global tensor dtype as TensorFloat32 

    # ---------- LOAD DATA ---------- # 

    train_loader, val_loader = load_shakespeare(DDP_WORLD_SIZE, DDP_LOCAL_RANK)    # load training and validation data
    train_iter, val_iter = cycle(train_loader), cycle(val_loader)                  # create infinite iterators

    if MASTER_PROCESS:    # print in command window for only one GPU
        print("\n*-------------- TRAINING --------------*")
        print(f"training tokens per batch: {TOKENS_PER_BATCH:,}")
        GRAD_ACCUM_STEPS = int(TOKENS_PER_BATCH / (BATCH_SIZE * BLOCK_SIZE * DDP_WORLD_SIZE))
        print(f"mini-batch size: {BATCH_SIZE} ({GRAD_ACCUM_STEPS} accumulation steps)")
        batches_per_epoch_train = int(math.ceil(len(train_loader) / GRAD_ACCUM_STEPS))
        print(f"batches per epoch: {batches_per_epoch_train:,} ({len(train_loader):,} mini-batches)")   # displays available batches for one epoch

        print("\n*-------------- VALIDATION --------------*")
        print(f"validation tokens per batch: {BATCH_SIZE * BLOCK_SIZE * VAL_ACCUM_STEPS:,}")
        print(f"mini-batch size: {BATCH_SIZE} ({VAL_ACCUM_STEPS} accumulation steps)")
        batches_per_epoch_val= int(math.ceil(len(val_loader) / VAL_ACCUM_STEPS))
        print(f"batches per epoch: {batches_per_epoch_val:,} ({len(val_loader):,} mini-batches)")

    # ---------- MODEL INSTANCE ---------- #

        print(f"\nloading model, optimiser and scheduler...\n")
        print(f"no. of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model.to(DEVICE)     # should be a GPT2_124M(GPTConfig(vocab_size=50304)) model instantiated in main.py
    optimiser = model.configure_optim(WEIGHT_DECAY, LEARNING_RATE, DEVICE.type)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimiser,
        T_max=(MAX_STEPS - WARMUP_STEPS),   # decay starts after warmup
        eta_min=0.1*LEARNING_RATE           # minimum learning rate
    )
    if MASTER_PROCESS:
        print(f"\ncompiling model...")
    model = torch.compile(model)    
    if DDP_WORLD_SIZE > 1:                                          # if using DDP
        model = DDP(model, device_ids=[DDP_LOCAL_RANK])             # wrap model in a PyTorch DDP container
    # raw_model = model.module if DDP_WORLD_SIZE > 1 else model     # to access the "raw" unwrapped model

    # ---------- MAIN TRAINING LOOP ---------- # 

    if MASTER_PROCESS:
        print(f"\nrunning validation every {VAL_INTERVAL} iterations")
        print(f"running {ITERATIONS:,} total iterations...\n")
        train_losses = np.empty(ITERATIONS)
        val_losses = np.full(ITERATIONS, np.nan)    # initialise with NaNs (due to interval usage)
        learning_rates = np.empty(ITERATIONS)

    import sys; sys.exit()
    
    for i in range(ITERATIONS):     # not using set_epoch() since iterations are used over epochs

        if MASTER_PROCESS:          # capture starting time for stats (master process only)
            t0 = time.time() if MASTER_PROCESS else None    

        model.train()
        X, y = next(train_iter)
        X_train, y_train = X.to(DEVICE), y.to(DEVICE)

        # ----- GRADIENT ACCUMULATION ----- #
        optimiser.zero_grad()                                                       # reset gradients
        train_loss = 0                                                              # accumulated train loss
        for micro_step in range(GRAD_ACCUM_STEPS):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):     # mixed precision
                _, loss = model(X_train, y_train)
            loss /= GRAD_ACCUM_STEPS                                                # scale loss to mimic full total batch average
            train_loss += loss.detach()                                             # prevent carry over of computational graph
            if DDP_WORLD_SIZE > 1:                                                  # could also use "contextlib.nullcontext()"
                if micro_step == GRAD_ACCUM_STEPS - 1:
                    loss.backward()             # synchronise gradients across all GPUs in the final accumulation step
                else:
                    with model.no_sync():       # context manager: accumulate gradients without synchronisation 
                        loss.backward()
            else:
                loss.backward()                 # no syncrhonisation for a single GPU
        if DDP_WORLD_SIZE > 1:                              # calculate and synchronise the average loss across all GPUs
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)    # all_reduce places the same final averaged result back on all GPUs
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
        if MASTER_PROCESS:
            t1 = time.time()
            dt = (t1 - t0) * 1e3        # time difference in miliseconds
            tps = (X.numel() * GRAD_ACCUM_STEPS * DDP_WORLD_SIZE) / (t1 - t0)    # training tokens/sec processed

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
            if MASTER_PROCESS:
                t1_val = time.time()
                dt_val = (t1_val - t0_val) * 1e3
                tps_val = (X.numel() * VAL_ACCUM_STEPS) / (t1_val - t0_val)     # validation tokens/sec processed
                val_losses[i] = val_loss
        
        # ----- LOG PROGRESS & STATS ----- #
        if MASTER_PROCESS:
            learning_rates[i] = lr                      # populate arrays for plotting
            train_losses[i] = train_loss.item()
            if i % LOG_INTERVAL == 0:
                pct = (i + 1) / ITERATIONS * 100        # percentage completion
                progress_str = (
                    f"\r{i + 1:,}/{ITERATIONS:,} ({pct:.0f}%) | "
                    f"train_loss: {train_loss.item():.3f} | "
                    f"{dt * 1e-3:,.2f} sec/batch ({dt / GRAD_ACCUM_STEPS:.1f} ms/iter) | "
                    f"{tps:,.0f} tok/sec | "
                    f"val_loss: {val_loss:.3f} | "
                    f"{dt_val * 1e-3:.1f} sec ({tps_val:,.0f} tok/sec)"
                )
                print(progress_str, end="")
    if MASTER_PROCESS:
        print("\n\nTraining Complete.")
    if DDP_WORLD_SIZE > 1:
        destroy_process_group()
    return model, train_losses, val_losses, learning_rates


def load_shakespeare(ddp_world_size: int, ddp_rank: int) -> tuple:
    """
    Loads training and validation `DataLoader` (PyTorch) objects for the `TinyShakespeare()` `Dataset`.
    For DDP, training data is split into equal-sized chunks across all GPUs (processes) using `DistributedSampler`.
    """
    print(f"\nloading data...\n")
    train_dataset = TinyShakespeare(        # load custom Dataset class for training
        block_size=BLOCK_SIZE,
        pct=PCT_DATA,
        train_split=TRAIN_SPLIT,
    )
    train_sampler = DistributedSampler(     # for DDP: divides the dataset into equal-sized chunks across all GPUs (processes)
        train_dataset,
        num_replicas=ddp_world_size,        # total no. of processes
        rank=ddp_rank,                      # current GPU integer ID
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,              # using DistributedSampler
        pin_memory=True
    )
    val_loader = DataLoader(                # validation dataset (no DistributedSampler used)
        TinyShakespeare(
            block_size=BLOCK_SIZE,
            pct=PCT_DATA,
            train_split=TRAIN_SPLIT,
            split="val",
            verbose=False
        ),
        batch_size=BATCH_SIZE,              # N.B. only VAL_ACCUM_STEPS batches used 
        shuffle=False
    )
    return train_loader, val_loader 


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
