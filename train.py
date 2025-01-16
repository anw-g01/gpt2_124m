import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import numpy as np
import math
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams["font.size"] = 9
plt.rcParams["lines.linewidth"] = 1
from config import *    # import global variables from config.py (all in capital letters)
from datasets import TinyShakespeare, FineWebEdu
from model import GPT2_124M, GPT2Config
from tqdm_bars import tqdmGPT
from hellaswag import evaluate


def initialise_ddp() -> tuple:
    """
    Set up DDP (Distributed Data Parallel) with `torch.distributed` to utilise multi-GPU training.
    The `torchrun` command will set the `env` variables: `RANK`, `LOCAL_RANK` and `WORLD_SIZE`.

    Returns:
    --
    A `tuple` containing:
    - `ddp_rank` (`int`): global process integer ID (e.g. `0-7` for `8` GPUs).
    - `ddp_local_rank` (`int`): GPU ID on the current node (e.g. `0-7` if all on one machine).
    - `ddp_world_size` (`int`): total number of processes (i.e. number of GPUs).
    - `device` (`torch.device`): device to be used for the current process.
    - `master_process` (`bool`): flag indicating if the current process is the master process (first GPU).
    """
    # check if running in a distributed environment (e.g. using torchrun):
    using_ddp = all(key in os.environ for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])    
    if using_ddp:
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


def train_gpt2(
        compile: bool = False,
        verbose: str = True
    ) -> tuple:
    """
    Parameters:
    --
    - `compile` (`bool`): whether to compile the model using `torch.compile()`. Default is `False`.
    - `verbose` (`str`): whether to print logs & metrics to the command line every `CHECKPOINT_INTVERAL` steps. Default is `True`.
    This is in addition to the custom `tqdmGPT` progress bar which will always be displayed during training by default. 
    
    Returns:
    --
    - `model` (`torch.nn.Module`): The trained model with updated weights, specificly `GPT2_124M(GPT2Config(vocab_size=50304))`. 
    """

    # get distributed parameters from environment variables (if using DDP)
    DDP_RANK, DDP_LOCAL_RANK, DDP_WORLD_SIZE, DEVICE, MASTER_PROCESS = initialise_ddp()

    torch.manual_seed(2001)    # for consistent intantiations of models across all processes
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2001)
    torch.set_float32_matmul_precision("high")    # set global tensor dtype as TensorFloat32 

    # ---------- LOAD DATA ---------- # 

    # train_loader, val_loader = load_shakespeare(DDP_WORLD_SIZE, DDP_LOCAL_RANK)     # tiny/big shakespeare dataset
    train_loader, val_loader = load_fineweb(DDP_WORLD_SIZE, DDP_LOCAL_RANK)         # load training and validation data
    train_iter, val_iter = cycle(train_loader), cycle(val_loader)                   # create infinite iterators
    
    if MASTER_PROCESS:    # only GPU rank 0 prints to the command window
        print("\n*-------------- TRAINING --------------*")
        print(f"effective batch: {TOKENS_PER_BATCH:,} tokens")
        tok_per_gpu = BATCH_SIZE * BLOCK_SIZE    # tokens processed per GPU per mini-batch
        GRAD_ACCUM_STEPS = int(TOKENS_PER_BATCH // (tok_per_gpu * DDP_WORLD_SIZE))  
        print(f"mini-batch size: [{BATCH_SIZE}, {BLOCK_SIZE}] ({GRAD_ACCUM_STEPS} acc. steps per GPU)")
        total_batches = len(train_loader) * DDP_WORLD_SIZE    # total mini-batches in ONE EPOCH for all GPUs
        # calculate no. of complete batches (called chunks) per epoch for all GPUs:
        chunks_per_epoch_train = int(math.ceil(total_batches / (GRAD_ACCUM_STEPS * DDP_WORLD_SIZE)))    
        chunks_per_gpu = int(math.ceil(chunks_per_epoch_train / DDP_WORLD_SIZE))
        print(f"DataLoader batches: {total_batches:,} ({len(train_loader):,} per GPU)")
        print(f"=> {chunks_per_epoch_train:,} chunks/epoch ({chunks_per_gpu:,} per GPU)")

        print("\n*-------------- VALIDATION --------------*")
        val_effective_batch = BATCH_SIZE * BLOCK_SIZE * VAL_ACCUM_STEPS * DDP_WORLD_SIZE
        print(f"effective batch: {val_effective_batch:,} tokens")
        print(f"mini-batch size: [{BATCH_SIZE}, {BLOCK_SIZE}] (with {VAL_ACCUM_STEPS} acc. steps per GPU)")
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
        n_validations = total_iterations // VAL_INTERVAL        # no. of validation runs (roughly due to epoch-ends)
        print(f"\nrunning validation and HellaSwag eval every {VAL_INTERVAL} iterations (total ~{n_validations:,})\n")
        n_checkpoints = n_validations // CHECKPOINT_INTERVAL    # no. of model checkpoints to write
        print(f"writing model checkpoints every {CHECKPOINT_INTERVAL} validation steps (total ~{n_checkpoints:,}) \n")
        # pre-allocate arrays to store training metrics for plotting:
        train_losses = np.zeros(total_iterations)               # store training losses (for plotting)
        val_losses = np.full(total_iterations, np.nan)          # initialise with NaNs due to interval storing
        hellaswag_scores = np.full(total_iterations, np.nan)    # store HellaSwag scores (interval storage)
        learning_rates = np.zeros(total_iterations)             # store learning rates (optional plotting)
        # create a log directory to store model checkpoints:
        os.makedirs(LOG_DIR, exist_ok=True)                     # create directory if it doesn't exist

    # create a custom tqdm bar for printing/logging stats (see tqdm_bars.py):
    pbar = tqdmGPT(     
        iterable=range(total_iterations),
        n_tokens=(BATCH_SIZE * BLOCK_SIZE),     # custom input: training tokens processed in input batch
        acc_steps=GRAD_ACCUM_STEPS,             # custom input: gradient accumulation steps
        disable=(DDP_LOCAL_RANK != 0),          # show progress bar for only the first GPU process (DDP)
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
            train_loss += loss.detach()         # accumulate as single-value tensor (only for logging after performing all_reduce)
            # accumulate gradients:
            if DDP_WORLD_SIZE > 1:                                                  # could also use "contextlib.nullcontext()"
                if micro_step == GRAD_ACCUM_STEPS - 1:
                    loss.backward()             # synchronise gradients across all GPUs in the final accumulation step
                else:
                    with model.no_sync():       # context manager: accumulate gradients without synchronisation 
                        loss.backward()
            else:
                loss.backward()                 # no synchronisation for a single GPU
        # clip gradients to prevent exploding gradients and update model parameters:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)          
        optimiser.step()                                                    
        # update the learning rate (linear warmup + cosine decay):
        if i < WARMUP_STEPS:
            lr = MAX_LEARNING_RATE * (i + 1) / WARMUP_STEPS     # linear warmup to LEARNING_RATE
            for param_group in optimiser.param_groups:
                param_group["lr"] = lr
        else:   # cosine decay begins with scheduler (after WARM_UPSTEPS)
            scheduler.step()                    
            lr = scheduler.get_last_lr()[0] 

        # ----- GET VALIDATION LOSS + HELLASWAG EVALUATION ----- #
        # run validation every VAL_INTERVAL iterations OR in the final iteration of each epoch:
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
            # run HellaSwag evaluation:
            n_correct, n_total = evaluate(model, DDP_WORLD_SIZE, DDP_LOCAL_RANK)    # evaluate on HellaSwag dataset

        # ----- ALL-REDUCE - DDP COMMUNICATION (FOR LOGGING) ----- #
        if DDP_WORLD_SIZE > 1:      # only if using DDP
            # average and synchronise single-value loss tensors across all GPUs:
            # (all_reduce places the same final averaged result back on all GPUs)
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)   # only for logging (loss.backward() already built-in gradient synchronisation)
            # only synchronise eval metrics on runs where validation is performed: 
            if (i % VAL_INTERVAL == 0) or (local_i == iters_per_epoch - 1):                 
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)     
                # sum and then synchronise HellaSwag evaluation counts across all GPUs: 
                # (using helper function due to non-tensor values)
                n_correct = value_reduce(n_correct, DEVICE)         
                n_total = value_reduce(n_total, DEVICE)

        # ----- STORE AND LOG METRICS (FOR PRINTING & PLOTTING) ----- #
        if MASTER_PROCESS:                          # only GPU rank 0 stores & prints
            train_losses[i] = train_loss.item()     # store training loss
            learning_rates[i] = lr                  # store learning rate
            # if at the end of an epoch, print a new line message to the command window
            if local_i == iters_per_epoch - 1:                                  # end of an epoch     
                i_pct = (i + 1) / total_iterations * 100                        # percentage of total iterations so far
                pbar.refresh()    # force update the progress bar
                pbar.write(
                    f"\n*--- epoch {epoch + 1}/{EPOCHS} complete | "            # no. of epochs completed so far
                    f"i: {i + 1:,}/{total_iterations:,} ({i_pct:.1f}%) ---*\n"  # iterations completed so far
                )
            # only store eval metrics on runs where validation is performed:
            if (i % VAL_INTERVAL == 0) or (local_i == iters_per_epoch - 1):                                                   # if validation was performed
                val_losses[i] = val_loss.item()                                         # store validation loss          
                hellaswag_scores[i] = n_correct / n_total * 100                         # store HellaSwag accuracy score (in %)
                # print stats to the command window on validation runs:
                if verbose and not (local_i == iters_per_epoch - 1):                    # DON'T print on epoch end
                    i_pct = (i + 1) / total_iterations * 100                            # percentage of total iterations so far
                    t = datetime.timedelta(seconds=int(pbar.format_dict["elapsed"]))    # total elapsed time
                    pbar.refresh()                                                      # force update the progress bar
                    pbar.write(                                                         # print to the command window
                        f"\nepoch {epoch + 1}/{EPOCHS} | "
                        f"i: {i + 1:,}/{total_iterations:,} ({i_pct:.1f}%) | "
                        f"train_loss: {train_losses[i]:.3f} | val_loss: {val_losses[i]:.3f} | "
                        f"HellaSwag: {hellaswag_scores[i]:.1f}% | elapsed: [{t}]\n"
                    )
                # --- WRITE MODEL CHECKPOINT TO LOG_DIR --- #
                # write a model checkpoint every CHECKPOINT_INTERVAL validations (NOT iterations) or end of epochs:
                if (i % (VAL_INTERVAL * CHECKPOINT_INTERVAL) == 0) or (local_i == iters_per_epoch - 1):
                    raw_model = model.module if DDP_WORLD_SIZE > 1 else model           # access the "raw" unwrapped model if DDP
                    # create a sub-directory for each checkpoint inside LOG_DIR:
                    prefix = "end" if (local_i == iters_per_epoch - 1) else "val"       # "end" prefix for epoch end
                    filename = get_checkpoint_filename(prefix, epoch + 1, i)            # get a standardised filename
                    checkpoint_dir = os.path.join(LOG_DIR, filename)                    # create a new checkpoint directory
                    checkpoint_path = os.path.join(checkpoint_dir, f"model.pt")         # path to save PyTorch model weights
                    # dictionary to store all metrics:
                    checkpoint = {      
                        "epoch": epoch + 1, 
                        "iteration": i,
                        "model_state_dict": raw_model.state_dict(),
                        # "optimiser_state_dict": optimiser.state_dict(),
                        # "scheduler_state_dict": scheduler.state_dict(),
                    }
                    torch.save(checkpoint, checkpoint_path)     # save model checkpoint
                    # save further metrics as numpy arrays:
                    for name, arr in [
                        ("train_losses", train_losses), ("val_losses", val_losses),
                        ("hellaswag_scores", hellaswag_scores), ("learning_rates", learning_rates)
                    ]:
                        np.save(os.path.join(checkpoint_dir, f"{name}.npy"), arr)   # save numpy array to directory

    # ---------- TRAINING COMPLETE ---------- #
    if MASTER_PROCESS:
        pbar.close()                                # close the tqdmGPT progress bar
        print("\n\n*--- TRAINING COMPLETE ---*")    # print completion message
    # if using DDP, clean up the process group:
    if DDP_WORLD_SIZE > 1:                          
        destroy_process_group()                     

    return model


def value_reduce(value: float, device: torch.device) -> float:
    """
    Helper function to reduce a single value across all GPU processes.
    Used for summing HellaSwag `n_correct` and `n_total` counts across all GPUs.
    
    A single value is converted into a PyTorch `tensor` since `dist.all_reduce()` 
    only operates with tensors. The value is summed and synchronised across all 
    GPU processes and the final value is extracted and returned.
    """
    tensor = torch.tensor(value, dtype=torch.long, device=device)   # convert a single value to tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)                   # sum and synchronise across all GPUs
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


def get_checkpoint_filename(prefix: str, epoch: int, iteration: int) -> str:
    """
    Returns a standardised filename (`str) for saving model checkpoint files.
    Edit the format string to change the naming convention consistently during
    training, as well as for accessing files for graph plotting.
    """
    return f"{prefix}_checkpoint_epoch_{epoch:02d}_iter_{iteration:05d}"


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
    train_dataset = TinyShakespeare(BLOCK_SIZE, pct=PCT_DATA, train_split=TRAIN_SPLIT) # load custom Dataset class for training
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
    val_dataset = TinyShakespeare(
        BLOCK_SIZE, pct=PCT_DATA, train_split=TRAIN_SPLIT,
        split="val", verbose=False
    )
    val_sampler = DistributedSampler(     
        val_dataset,
        num_replicas=ddp_world_size,        
        rank=ddp_rank,                      
        shuffle=True
    )
    val_loader = DataLoader(                
        val_dataset,
        batch_size=BATCH_SIZE,              
        sampler=val_sampler,              
        pin_memory=True
    )
    return train_loader, val_loader 


def display_graphs(
        epoch_idx: int,
        iter_idx: int,
        log_dir: str = LOG_DIR,
        checkpoint_type: str = "end",
        plot_lr: bool = True,
        save: bool = True,
        format: str = "png"
    ) -> None:
    """
    Plot training results given a model checkpoint directory.

    Loads training and validation losses, HellaSwag evaluation scores, and learning rates
    from the specified checkpoint directory. The figure consists of two subplots: one for
    training and validation losses, and one for HellaSwag evaluation scores. Optionally, 
    it can also plot the learning rate on the same subplot as the losses. The figure can
    be saved as an image file in the specified `format` type.

    Parameters:
    --
    - `epoch_idx` (`int`): epoch index (starting from `1`) for the checkpoint.
    - `iter_idx` (`int`): iteration (starting from `1`) index for the checkpoint.
    - `log_dir` (`str`): directory where the logs and checkpoints are stored.
    - `checkpoint_type` (`str`): prefix of the checkpoint file name (must be `"end"` or `"val"`).
    - `plot_lr` (`bool`): whether to plot the learning rate on the same plot as the losses.
    - `save` (`bool`): whether to save the plot as an image file.
    - `format` (`str`): format to save the plot image (e.g., `"png"`, `"jpg"`).    
    """
    assert checkpoint_type in ["end", "val"], "checkpoint_type must be 'end' or 'val'"
    # get the checkpoint directory from the filename (for the specified epoch and iteration):
    filename = get_checkpoint_filename(checkpoint_type, epoch_idx, iter_idx)
    checkpoint_dir = os.path.join(log_dir, filename)
    assert os.path.exists(checkpoint_dir), f"checkpoint directory does not exist: {checkpoint_dir}"     # check if the directory exists
    # load numpy arrays from the checkpoint directory:
    train_losses = np.load(os.path.join(checkpoint_dir, "train_losses.npy"))
    val_losses = np.load(os.path.join(checkpoint_dir, "val_losses.npy"))
    hellaswag_scores = np.load(os.path.join(checkpoint_dir, "hellaswag_scores.npy"))
    learning_rates = np.load(os.path.join(checkpoint_dir, "learning_rates.npy"))        
    
    # ----- MAIN FIGURE ----- #
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(f"GPT2 Training Results: {filename}")  # title for the entire figure
    x = np.arange(1, len(train_losses) + 1)             # x-values for all plots
    
    # --- LEFT SUBPLOT (training and validation losses) --- #
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("iteration (per GPU)")
    axs[0].set_ylabel("loss")
    axs[0].grid(True)
    axs[0].plot(
        x, train_losses,
        label="train_loss", color="tab:olive"
    )
    # for val_loss, only plot non-NaN values due to interval storage:
    val_idx = np.isfinite(val_losses)           # cell turns False if cell was NaN
    axs[0].plot(
        x[val_idx], val_losses[val_idx],        # only select elements where val_idx is True (i.e. non-NaN)  
        label="val_loss", color="tab:purple"
    )
    axs[0].axhline(y=3.292, color="tab:red", linestyle="--", label="OpenAI GPT-2 (124M) baseline")
    axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))    # comma separator for x-axis tickers
    # optional: plot learning rates in a twin axis (with losses figure):
    if plot_lr:
        lr_color = "tab:cyan"    
        ax_lr = axs[0].twinx()                              # create a second y-axis for learning rates
        ax_lr.set_ylabel("learning rate", color=lr_color)
        ax_lr.plot(
            x, learning_rates, label="learning_rate",
            color=lr_color, alpha=0.5                       # set alpha for transparency    
        )
        ax_lr.tick_params(axis="y", labelcolor=lr_color)    # set axis tick colour
        ax_lr.legend()                                      # set legend location
        # combine legends for twin axes:
        lines, labels = axs[0].get_legend_handles_labels()
        lines_lr, labels_lr = ax_lr.get_legend_handles_labels()
        axs[0].legend(lines + lines_lr, labels + labels_lr, loc="upper right")
    else:
        axs[0].legend()                                     # legend for losses
    
    # --- RIGHT SUBPLOT (HellaSwag evaluation scores) --- #
    axs[1].set_title("HellaSwag Evaluation")
    axs[1].set_xlabel("iteration (per GPU)")
    axs[1].set_ylabel("accuracy (%)")
    axs[1].grid(True)
    axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))    # comma separator for x-axis tickers
    hs_idx = np.isfinite(hellaswag_scores)    # bool: False if cell is NaN
    axs[1].plot(x[hs_idx], hellaswag_scores[hs_idx])
    # plot baseline scores from other models:
    axs[1].axhline(y=29.6, color="tab:red", linestyle="--", label="OpenAI GPT-2 (124M) baseline")
    axs[1].axhline(y=48.9, color="tab:blue", linestyle="--", label="OpenAI GPT-2 (1.56B) baseline")
    axs[1].axhline(y=54.7, color="tab:green", linestyle="--", label="OpenAI GPT-3 (1.56B) baseline")
    axs[1].legend()
    plt.tight_layout()  
    if save:
        plt.savefig(
            f"figure_{filename}", 
            dpi=300, bbox_inches="tight",
            format=format
        )
    plt.show()