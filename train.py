import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from config import *    # import all global variables (all in caps)
from dataset import TinyShakespeare, FineWebEdu
from model import GPT2_124M, GPT2Config
from tqdm_bars import tqdmGPT
from hellaswag import evaluate

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
        print(f"\nusing single device: {device}")
    return ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


def train(compile: bool = True) -> tuple:
    """
    Train a PyTorch model using gradient accumulation, mixed precision, and cosine decay learning rate scheduling.
    Selected hyperparameters are close to GPT-2 and GPT-3 model choices by OpenAI, released in any papers.
    - The function uses gradient accumulation with mini-batch processing, if GPU memory is constrained.
    - Mixed precision training (with `bfloat16`) is implemented for efficiency with modern GPUs.
    - Learning rate scheduling includes a linear warm-up phase (to `LEARNING_RATE`) followed by a cosine decay rate.
    - Validation is performed periodically based on `VAL_INTERVAL` (see `config.py`).
    -----
    Returns:
        `tuple`: A tuple containing:
            - `model` (`torch.nn.Module`): The trained model with updated weights, specificly `GPT2_124M(GPT2Config(vocab_size=50304))`. 
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

    # train_loader, val_loader = load_shakespeare(DDP_WORLD_SIZE, DDP_LOCAL_RANK)    # load training and validation data

    train_loader, val_loader = load_fineweb(DDP_WORLD_SIZE, DDP_LOCAL_RANK)    # load training and validation data
    train_iter, val_iter = cycle(train_loader), cycle(val_loader)              # create infinite iterators
    
    if MASTER_PROCESS:    # print in command window for only one GPU
        print("\n*-------------- TRAINING --------------*")
        print(f"effective batch: {TOKENS_PER_BATCH:,} tokens")
        
        tok_per_gpu = BATCH_SIZE * BLOCK_SIZE    # tokens processed per GPU per mini-batch
        GRAD_ACCUM_STEPS = int(TOKENS_PER_BATCH // (tok_per_gpu * DDP_WORLD_SIZE))  
        print(f"mini-batch size: [{BATCH_SIZE}, {BLOCK_SIZE}] ({GRAD_ACCUM_STEPS} acc. steps)")
        total_batches = len(train_loader) * DDP_WORLD_SIZE    # total mini-batches in ONE EPOCH for all GPUs
        # calculate no. of complete batches (called chunks) per epoch for all GPUs:
        chunks_per_epoch_train = int(math.ceil(total_batches / (GRAD_ACCUM_STEPS * DDP_WORLD_SIZE)))    
        chunks_per_gpu = int(math.ceil(chunks_per_epoch_train / DDP_WORLD_SIZE))
        print(f"DataLoader batches: {total_batches:,} ({len(train_loader):,} per GPU)")
        print(f"=> {chunks_per_epoch_train:,} chunks/epoch ({chunks_per_gpu:,} per GPU)")

        print("\n*-------------- VALIDATION --------------*")
        val_effective_batch = BATCH_SIZE * BLOCK_SIZE * VAL_ACCUM_STEPS * DDP_WORLD_SIZE
        print(f"effective batch: {val_effective_batch:,} tokens")
        print(f"mini-batch size: [{BATCH_SIZE}, {BLOCK_SIZE}] (with {VAL_ACCUM_STEPS} acc. steps)")
        total_val_batches = len(val_loader) * DDP_WORLD_SIZE
        chunks_per_epoch_val = int(math.ceil(total_val_batches / (VAL_ACCUM_STEPS * DDP_WORLD_SIZE)))
        val_chunks_per_gpu = int(math.ceil(chunks_per_epoch_val / DDP_WORLD_SIZE))
        print(f"DataLoader batches: {total_val_batches:,} ({len(val_loader):,} per GPU)")
        print(f"=> {chunks_per_epoch_val:,} chunks/epoch ({val_chunks_per_gpu:,} per GPU)")

    # ---------- MODEL INSTANCE ---------- #
        print(f"\nloading model, optimiser and scheduler...\n")

    iters_per_epoch = len(train_loader)             # no. of available mini-batches in one epoch (per GPU)    
    total_iterations = iters_per_epoch * EPOCHS     # total no. of iterations across full training (per GPU)

    model = GPT2_124M(GPT2Config(vocab_size=50304)).to(DEVICE)      # increase vocab size to (2^7 * 3 * 131)
    optimiser = model.configure_optim(WEIGHT_DECAY, MAX_LEARNING_RATE, DEVICE.type)
    scheduler = CosineAnnealingLR(
        optimizer=optimiser,
        T_max=(total_iterations - WARMUP_STEPS),    # iterations over which decay starts (after linear warmup phase)
        eta_min=0.1*MAX_LEARNING_RATE               # set minimum learning rate to 10% of the maximum
    )
    
    model = torch.compile(model) if compile else model      # compile model if specified
    # if using DDP, wrap model in a PyTorch DDP container
    if DDP_WORLD_SIZE > 1:                                          
        model = DDP(model, device_ids=[DDP_LOCAL_RANK])             
    # raw_model = model.module if DDP_WORLD_SIZE > 1 else model       # to access the "raw" unwrapped model

    if MASTER_PROCESS:
        print(f"\ncompiling model...")
        print(f"no. of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---------- MAIN TRAINING LOOP ---------- # 

        print(f"\ntraining for {EPOCHS} epochs ({iters_per_epoch:,} iterations per epoch per GPU)")
        print(f"running {total_iterations:,} parallel iterations across {DDP_WORLD_SIZE} GPUs...\n")
        print(f"\nrunning validation and HellaSwag eval every {VAL_INTERVAL} iterations")
        train_losses = np.zeros(total_iterations)
        val_losses = np.full(total_iterations, np.nan)  # initialise with NaNs (due to interval usage)
        learning_rates = np.zeros(total_iterations)
        hellaswag_scores = np.zeros(total_iterations)   # store HellaSwag scores during evaluation
    
    pbar = tqdmGPT(     # create a custom tqdm bar for printing/logging stats (see tqdm_bars.py)
        iterable=range(total_iterations),
        n_tokens=(BATCH_SIZE * BLOCK_SIZE),         # custom input: tokens processed in input batch
        acc_steps=GRAD_ACCUM_STEPS,                 # custom input: gradient accumulation steps
        desc="train_loss: ? | val_losses: ? |",
        total=total_iterations,
        disable=(DDP_LOCAL_RANK != 0),    # show progress bar for only the first GPU process (DDP)
    )

    for i in pbar:    # pbar acts as a normal iterator when disabled (for non-master GPU processes)

        epoch = i // iters_per_epoch    # current epoch number
        local_i = i % iters_per_epoch   # current iteration within the current epoch

        # ----- TRAINING - GRADIENT ACCUMULATION ----- #
        model.train()
        optimiser.zero_grad()                                                       # reset gradients
        train_loss = 0                                                              # accumulated train loss
        for micro_step in range(GRAD_ACCUM_STEPS):
            X, y = next(train_iter)                                                 # get next training mini-batch
            X_train, y_train = X.to(DEVICE), y.to(DEVICE)                           # move to GPU
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):     # mixed precision (use bfloat16)
                _, loss = model(X_train, y_train)
            loss /= GRAD_ACCUM_STEPS                                                # scale loss to mimic full total batch average
            train_loss += loss.detach()         # keep as a tensor (for performing all_reduce)
            
            if DDP_WORLD_SIZE > 1:                                                  # could also use "contextlib.nullcontext()"
                if micro_step == GRAD_ACCUM_STEPS - 1:
                    loss.backward()             # synchronise gradients across all GPUs in the final accumulation step
                else:
                    with model.no_sync():       # context manager: accumulate gradients without synchronisation 
                        loss.backward()
            else:
                loss.backward()                 # no synchronisation for a single GPU
        # calculate and synchronise the average loss across all GPUs:
        if DDP_WORLD_SIZE > 1:                                              
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)    # all_reduce places the same final averaged result back on all GPUs
        # clip gradients to prevent exploding gradients and update model parameters:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)          
        optimiser.step()                                                    
        
        # ----- LEARNING RATE SCHEDULER ----- #
        if i < WARMUP_STEPS:
            lr = MAX_LEARNING_RATE * (i + 1) / WARMUP_STEPS     # linear warmup to LEARNING_RATE
            for param_group in optimiser.param_groups:
                param_group["lr"] = lr
        else:   # cosine decay begins with scheduler (after WARM_UPSTEPS)
            scheduler.step()                    
            lr = scheduler.get_last_lr()[0] 

        # ----- GET VALIDATION LOSS + HELLASWAG EVAL ----- #
        # run validation every VAL_INTERVAL iterations OR in the final iteration of each epoch
        if (i % VAL_INTERVAL == 0) or (local_i == iters_per_epoch - 1): 
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(VAL_ACCUM_STEPS):   
                    X, y = next(val_iter)
                    X_val, y_val = X.to(DEVICE), y.to(DEVICE)
                    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):     
                        _, loss = model(X_val, y_val)
                    loss /= VAL_ACCUM_STEPS
                    val_loss += loss.detach()           
            if DDP_WORLD_SIZE > 1:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)     # average and synchronise validation loss across all GPUs
            if MASTER_PROCESS:
                val_losses[i] = val_loss                            # rank 0 stores the average validation loss (calculated across all GPUs)
            # ----- HellaSwag EVALUATION ----- #    # N.B. NEEDS UPDATING TO USE DDP
            n_correct, n_total = evaluate(model, DDP_WORLD_SIZE, DDP_LOCAL_RANK)    # evaluate on HellaSwag dataset
            if DDP_WORLD_SIZE > 1:
                n_correct = value_reduce(n_correct, DEVICE)         # sum and then synchronise correct counts across all GPUs
                n_total = value_reduce(n_total, DEVICE)             # sum and then synchronise total counts across all GPUs
            if MASTER_PROCESS:
                score = n_correct / n_total * 100   # percentage accuracy on HellaSwag dataset
                hellaswag_scores[i] = score         # store HellaSwag score
        
        # ----- LOG PROGRESS & STATS ----- #
        if MASTER_PROCESS:
            learning_rates[i] = lr                      
            train_losses[i] = train_loss.item()
            if i % LOG_INTERVAL == 0:
                pbar.set_description_str(
                    f"epoch: {epoch + 1}/{EPOCHS} | "
                    f"train_loss: {train_loss.item():.3f} | "
                    f"val_loss: {val_loss:.3f}"
                )

    # ---------- TRAINING COMPLETE ---------- #
    if MASTER_PROCESS:
        print("\n\nTraining Complete.")     # print completion message
    if DDP_WORLD_SIZE > 1:                  # if using DDP
        destroy_process_group()             # clean up DDP process group

    return model, train_losses, val_losses, learning_rates


def value_reduce(value: float, device: torch.device) -> float:
    """
    Helper function to reduce a single value across all GPU processes.
    Used for summing HellaSwag `n_correct` and `n_total` counts across all GPUs.
    
    A single value is converted into a PyTorch `tensor` since `dist.all_reduce()` 
    only operates with tensors. The value is summed and synchronised across all 
    GPU processes and the final value is extracted and returned.
    """
    tensor = torch.tensor(value, dtype=torch.long, device=device)   # convert a single value to tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def cycle(iterable):
    """
    Infinitely cycles over an iterable object (e.g. a `DataLoader`) using a generator.
    Used over `itertools.cycle()` to prevent memory leaks for large datasets like `FineWebEdu()`.
    """
    iterator = iter(iterable)
    while True:                         
        try:    
            yield next(iterator)        # yield the next item in the iterator
        except StopIteration:           # iterator reaches the end
            iterator = iter(iterable)   # reset the iterator


def load_fineweb(ddp_world_size: int, ddp_rank: int) -> tuple:
    """
    Loads training and validation `DataLoader` (PyTorch) objects for the `FineWebEdu()` dataset.

    For DDP, a `DistributedSampler` splits all available `global_idx` indicies (defined in `__len__`)
    amongst GPU processes which independently handles its batch of shard loading and processing. 

    All shuffling must be set to False to prevent constant shard loading. Utilising `self.cache()` to
    store the current shard being processed improves continuous iteration until the next shard.
    The validation set only occurs over one shard file, hence only a single shard has to be loaded once
    and so shuffling can occur without major overheads.
    """
    # training dataset:
    train_dataset = FineWebEdu(BATCH_SIZE, BLOCK_SIZE, split="train")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=ddp_world_size,
        rank=ddp_rank,
        shuffle=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,            # must be set to None
        sampler=train_sampler,      # using a DistributedSampler
        pin_memory=True,
    )
    # validation dataset:
    val_dataset = FineWebEdu(BATCH_SIZE, BLOCK_SIZE, split="val")
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=ddp_world_size,
        rank=ddp_rank,
        shuffle=True        # shuffling does not affect speed as only one shard is loaded
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        sampler=val_sampler,
        pin_memory=True,
    )
    return train_loader, val_loader


def load_shakespeare(ddp_world_size: int, ddp_rank: int) -> tuple:
    """
    Loads training and validation `DataLoader` (PyTorch) objects for the `TinyShakespeare()` dataset.
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
