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
from config import *    # import global variables from config.py (all in capital letters)
from model import GPT2_124M, GPT2Config
from tqdm_bars import tqdmGPT
from hellaswag import HellaSwag, hs_eval 
from fineweb import FineWebEdu
from shakespeare.shakespeare import Shakespeare


def train_gpt2(
        compile: bool = False,
        eval: bool = True,
        verbose: bool = True,
    ) -> GPT2_124M:
    """
    Parameters:
    --
        `compile` (`bool`): whether to compile the model using `torch.compile()`. Default is `False`.
        `eval` (`bool`): whether to run HellaSwag evaluation on the validation set. Default is `True`.
        `verbose` (`str`): whether to print validation logs & metrics to the command line every
        `CHECKPOINT_INTVERAL` steps. Default is `True`. This does not have any effect on the custom 
        `tqdmGPT` progress bar which will always be displayed during training by default. Print logs 
        will also always occur to signify the end of an epoch by default.
    
    Returns:
    --
        `model` (`GPT2_124M`): The trained model (a 'torch.nn.Module`) with updated weights, specificly a `GPT2_124M(GPT2Config(vocab_size=50304))` model class. 
    """

    # get distributed parameters from environment variables (if using DDP)
    RANK, LOCAL_RANK, WORLD_SIZE, DEVICE, MASTER_PROCESS = _setup_ddp()

    torch.manual_seed(2001)    # for consistent intantiations of models across all processes
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2001)
    torch.set_float32_matmul_precision("high")    # set global tensor dtype as TensorFloat32 

    # ---------- LOAD DATA ---------- # 
    # train_loader, val_loader = _load_shakespeare(WORLD_SIZE, RANK, size="tiny")    # (option to load a shakespeare dataset)
    train_loader, val_loader = _load_fineweb(WORLD_SIZE, RANK)          # load training and validation data
    train_iter, val_iter = _cycle(train_loader), _cycle(val_loader)     # create infinite iterators
    hs_loader = _load_hellaswag(WORLD_SIZE, RANK)                       # load HellaSwag dataloader for evaluation

    tok_per_gpu = BATCH_SIZE * BLOCK_SIZE                                               # training tokens processed per GPU per mini-batch
    grad_accum_steps = int(TOKENS_PER_BATCH // (tok_per_gpu * WORLD_SIZE))              # accumulation steps until backpropagation
    total_train_mini_batches = len(train_loader) * WORLD_SIZE                           # total mini-batches in one overall (across all GPUs)
    # calculate no. of COMPLETE training batches (not mini-batches):
    # using floor because remaining mini-batches wouldn't complete a full batch:
    train_batches_per_epoch = int(math.floor(len(train_loader) / grad_accum_steps))     # essentially the steps per epoch per GPU

    # only GPU rank 0 prints to the command window
    if MASTER_PROCESS:    
        print("\n*-------------- TRAINING --------------*")
        print(f"effective batch: {TOKENS_PER_BATCH:,} tokens")
        print(f"mini-batch size: [{BATCH_SIZE}, {BLOCK_SIZE}]")
        print(f"no. of mini-batches: {total_train_mini_batches:,} ({len(train_loader):,} per GPU)")
        print(f"using {grad_accum_steps} accumulation step(s) per GPU")
        print(f"=> {train_batches_per_epoch:,} batches/epoch per GPU")     

        print("\n*------------- VALIDATION -------------*")
        print(f"running {VAL_ACCUM_STEPS} accumulation steps per GPU ")
        print(f"mini-batch size: [{BATCH_SIZE}, {BLOCK_SIZE}]")
        val_effective_batch = BATCH_SIZE * BLOCK_SIZE * VAL_ACCUM_STEPS * WORLD_SIZE
        print(f"=> effective batch: {val_effective_batch:,} tokens")
        total_val_mini_batches = len(val_loader) * WORLD_SIZE
        print(f"no. of mini-batches: {total_val_mini_batches:,} ({len(val_loader):,} per GPU)")
        val_batches_per_epoch = int(math.floor(total_val_mini_batches / (VAL_ACCUM_STEPS * WORLD_SIZE)))
        print(f"=> {val_batches_per_epoch:,} batches/epoch per GPU")

        if eval:    # if running evaluation (eval=True)
            print("\n*----------- HELLASWAG EVAL -----------*")
            hs_examples = len(hs_loader) * WORLD_SIZE
            print(f'total examples in "val" set: {hs_examples:,}')
            print(f"using DistributedSampler with {WORLD_SIZE} GPU(s)")
            # sampler uses drop_last=True if not exactly divisible by world_size (see hellaswag.py)
            print(f"=> {len(hs_loader):,} unique examples per GPU")    

    # ---------- MODEL INSTANCE ---------- #
        print(f"\nloading model, optimiser and scheduler...\n")

    steps_per_epoch = train_batches_per_epoch           # no. of steps (batches) per epoch (per GPU) - equivalent to no. of training batches
    total_steps = int(steps_per_epoch * EPOCHS)         # total no. of complete training batches to process for full training (per GPU)

    model = GPT2_124M(GPT2Config(vocab_size=50304)).to(DEVICE)      # increase vocab size to (2^7 * 3 * 131)
    
    # configure an optimiser and scheduler: 
    optimiser = model.configure_optim(WEIGHT_DECAY, MAX_LEARNING_RATE, DEVICE)
    scheduler = CosineAnnealingLR(
        optimizer=optimiser,
        T_max=(total_steps - WARMUP_STEPS),     # iteration steps over which decay starts (after linear warmup phase)
        eta_min=0.1*MAX_LEARNING_RATE           # set minimum learning rate to 10% of the maximum
    )
        
    model = torch.compile(model) if compile else model    # compile model if specified
    # if using DDP, wrap model in a PyTorch DDP container
    if WORLD_SIZE > 1:                                          
        model = DDP(model, device_ids=[LOCAL_RANK])    
    
    if MASTER_PROCESS:
        print(f"compiling model: {compile}")
        print(f"no. of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---------- MAIN TRAINING LOOP ---------- # 

        print(f"\ntraining for {EPOCHS} epoch(s) ({train_batches_per_epoch:,} steps/epoch per GPU)")
        print(f"total: {total_steps:,} steps/GPU (in parallel across {WORLD_SIZE} GPU(s))")
        n_validations = total_steps // VAL_INTERVAL        # no. of validation runs (roughly due to epoch-ends)
        print(f"running validation (and/or HellaSwag eval) every {VAL_INTERVAL} steps (total ~{n_validations:,})")
        n_checkpoints = n_validations // CHECKPOINT_INTERVAL    # no. of model checkpoints to write
        print(f"writing model checkpoints every {CHECKPOINT_INTERVAL} validation runs (total ~{n_checkpoints:,}) \n")
        
        # pre-allocate arrays to store training metrics for plotting:
        train_losses = np.zeros(total_steps)                # store training losses (for plotting)
        val_losses = np.full(total_steps, np.nan)           # initialise with NaNs due to interval storing
        hellaswag_scores = np.full(total_steps, np.nan)     # store HellaSwag scores (interval storage)
        learning_rates = np.zeros(total_steps)              # store learning rates (optional plotting)
        # create a log directory to store model checkpoints:
        os.makedirs(LOG_DIR, exist_ok=True)                 # create directory if it doesn't exist

    # create a custom tqdm bar for printing/logging stats (see tqdm_bars.py):
    pbar = tqdmGPT(     
        iterable=range(total_steps),
        n_tokens=BATCH_SIZE * BLOCK_SIZE * WORLD_SIZE * grad_accum_steps,   # custom input: training tokens processed in input batch
        disable=(RANK != 0),      # show progress bar for only the first GPU process (DDP)
        miniters=1,                     # update progress bar every 'x' iterations, default is 100 for tqdmGPT (see tqdm_bars.py)
    )

    for i in pbar:    # pbar acts as a normal iterator when disabled (for non-master GPU processes)

        epoch = i // steps_per_epoch    # current epoch number (starts at 0)
        local_i = i % steps_per_epoch   # current iteration within the current epoch

        # ----- TRAINING - GRADIENT ACCUMULATION ----- #
        model.train()
        optimiser.zero_grad()                                                       # reset gradients
        train_loss = 0                                                              # accumulated train loss
        for j in range(grad_accum_steps):
            X, y = next(train_iter)                                                 # get next training mini-batch
            X_train, y_train = X.to(DEVICE), y.to(DEVICE)                           # move to GPU
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):     # mixed precision (use bfloat16)
                _, loss = model(X_train, y_train)
            loss /= grad_accum_steps                                                # scale loss to mimic full total batch average
            train_loss += loss.detach()         # accumulate as single-value tensor (only for logging after performing all_reduce)
            # accumulate gradients:
            if WORLD_SIZE > 1:                                                  # could also use "contextlib.nullcontext()"
                if j == grad_accum_steps - 1:
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
        # run validation every VAL_INTERVAL steps OR in the final step of an epoch:
        if (i % VAL_INTERVAL == 0) or (local_i == steps_per_epoch - 1): 
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(VAL_ACCUM_STEPS):   
                    X, y = next(val_iter)
                    X_val, y_val = X.to(DEVICE), y.to(DEVICE)
                    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):     
                        _, loss = model(X_val, y_val)
                    loss /= VAL_ACCUM_STEPS
                    val_loss += loss.detach()           
            # run HellaSwag evaluation if eval=True:
            if eval:
                n_correct, n_total = hs_eval(hs_loader, model)      # evaluate on HellaSwag dataset

        # ----- ALL-REDUCE - DDP COMMUNICATION (FOR LOGGING) ----- #
        if WORLD_SIZE > 1:      # only if using DDP
            # average and synchronise single-value loss tensors across all GPUs:
            # (all_reduce places the same final averaged result back on all GPUs)
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)   # only for logging (loss.backward() already built-in gradient synchronisation)
            # only synchronise eval metrics on runs where validation is performed: 
            if eval and ( (i % VAL_INTERVAL == 0) or (local_i == steps_per_epoch - 1) ):                 
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)     
                # sum and then synchronise HellaSwag evaluation counts across all GPUs: 
                # (using helper function due to non-tensor values)
                n_correct = _value_reduce(n_correct, DEVICE)         
                n_total = _value_reduce(n_total, DEVICE)

        # ----- STORE AND LOG METRICS (FOR PRINTING & PLOTTING) ----- #
        if MASTER_PROCESS:                          # only GPU rank 0 stores & prints
            train_losses[i] = train_loss.item()     # store training loss
            learning_rates[i] = lr                  # store learning rate
            # if at the end of an epoch, print a new line message to the command window
            if local_i == steps_per_epoch - 1:          # end of an epoch     
                i_pct = (i + 1) / total_steps * 100     # percentage of total steps so far
                pbar.refresh()    # force update the progress bar
                pbar.write(
                    f"\n*----- EPOCH {epoch + 1}/{EPOCHS} COMPLETE | "            # no. of epochs completed so far
                    f"step: {i + 1:5,}/{total_steps:,} ({i_pct:4.1f}%) -----*\n"  # steps completed so far
                )
            # only store eval metrics on runs where validation is performed:
            if (i % VAL_INTERVAL == 0) or (local_i == steps_per_epoch - 1):                 # if validation was performed
                val_losses[i] = val_loss.item()                                             # store validation loss          
                hellaswag_scores[i] = n_correct / n_total * 100 if eval else None           # store HellaSwag accuracy score (in %)
                # print stats to the command window on validation runs (but not if it's a the end of an epoch):
                if verbose and not (local_i == steps_per_epoch - 1):                        # DON'T print on epoch end
                    i_pct = (i + 1) / total_steps * 100                                     # percentage of total steps so far
                    t = datetime.timedelta(seconds=int(pbar.format_dict["elapsed"]))        # total elapsed time
                    pbar.refresh()                                                          # force update the progress bar
                    hs_log = f"HellaSwag: {hellaswag_scores[i]:.1f}% | " if eval else ""    # only print eval log string if eval=True
                    pbar.write(                                                             # print to the command window
                        f"epoch {epoch + 1}/{EPOCHS} | "
                        f"step: {i + 1:5,}/{total_steps:,} ({i_pct:4.1f}%) | "
                        f"train_loss: {train_losses[i]:6.3f} | val_loss: {val_losses[i]:6.3f} | "
                        f"{hs_log}[{t}]"
                    )
                # --- WRITE MODEL CHECKPOINT TO LOG_DIR --- #
                # write a model checkpoint every CHECKPOINT_INTERVAL validations (NOT steps) or end of epochs:
                if (i % (VAL_INTERVAL * CHECKPOINT_INTERVAL) == 0) or (local_i == steps_per_epoch - 1):
                    # create a sub-directory for each checkpoint inside LOG_DIR:
                    prefix = "end" if (local_i == steps_per_epoch - 1) else "val"                   # "end" prefix for epoch end
                    filename = _get_checkpoint_filename(prefix, epoch + 1, i + 1, WORLD_SIZE)   # get a standardised filename
                    # create a new checkpoint folder for each checkpoint:
                    checkpoint_dir = os.path.join(LOG_DIR, filename)     
                    os.makedirs(checkpoint_dir, exist_ok=True)                                  # create directory if it doesn't exist
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint.pt")      # path to save PyTorch model weights
                    # dictionary to store all metrics:
                    checkpoint = {      
                        "epoch": epoch + 1, 
                        "step": i,
                        "model_state_dict": (model.module if WORLD_SIZE > 1 else model).state_dict(),   # must use raw model if using DDP
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
        print("\n*----- TRAINING COMPLETE -----*")  # print completion message
        # additional stats:
        t = datetime.timedelta(seconds=int(pbar.format_dict["elapsed"]))    # total elapsed time
        print(f"\ntotal elapsed time: {t}")
        avg_iter_per_sec = total_steps / t.total_seconds()
        print(f"average: {avg_iter_per_sec:.1f} batches/s processed ({1/avg_iter_per_sec:.2f} s/batch)")    # one step = one batch
        pbar.close()                                # close the tqdmGPT progress bar
        
    # if using DDP, clean up the process group:
    if WORLD_SIZE > 1:                          
        destroy_process_group()                     

    return model

# ---------- HELPER FUNCTIONS ---------- #

def _setup_ddp() -> tuple:
    """
    Initialise DDP (Distributed Data Parallel) with `torch.distributed` to utilise multi-GPU training.
    The `torchrun` command will set the `env` variables: `RANK`, `LOCAL_RANK` and `WORLD_SIZE`.

    Returns:
    --
    `tuple`: a tuple containing:
        `rank` (`int`): global process integer ID (e.g. `0-7` for `8` GPUs).
        `local_rank` (`int`): GPU ID on the current node (e.g. `0-7` for `16` GPUs across two nodes with `8` GPUs each).
        `world_size` (`int`): total number of processes (i.e. number of GPUs).
        `device` (`torch.device`): device to be used for the current process.
        `master_process` (`bool`): flag indicating if the current process is the master process (first GPU).
    """
    # check if running in a distributed environment (e.g. using torchrun):
    using_ddp = all(key in os.environ for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])    
    if using_ddp:
        assert torch.cuda.is_available(), "train.py cannot be run without CUDA."    # CUDA must be available for DDP
        init_process_group(backend="nccl")                  # initialise the process group
        rank = dist.get_rank()                              # global process integer ID
        local_rank = int(os.environ["LOCAL_RANK"])          # GPU ID on the current node 
        world_size = dist.get_world_size()                  # total no. of processes
        device = f"cuda:{local_rank}"                       
        torch.cuda.set_device(device)                       # set the device for current process on current node
        master_process = (rank == 0)                        # flag for the first GPU (for logging, checkpointing etc.) 
        if master_process:
            print(f"\nusing DDP with WORLD_SIZE: {world_size}")
    else:
        rank, local_rank, world_size = 0, 0, 1      # fallback for non-DDP setup
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nusing single device: {device.upper()}")
    return rank, local_rank, world_size, device, master_process


def _test_ddp() -> None:
    """
    Perform a simple test to verify a `DDP` setup.
    
    Execute on command prompt:
        `torchrun --standalone --nproc_per_node=8 train.py`

    N.B. Full training runs are executed with `main.py`, NOT `train.py`.
    """

    rank, _, world_size, device, master_process = _setup_ddp()

    if world_size == 1:
        print(f"\ninvalid DDP with {world_size=}, exiting test...")
        return      # break

    if master_process:
        print(f"\nrunning DDP test with {world_size=}...")
        print(f"master process running on {device}\n")

    # create a tensor on each GPU
    tensor = torch.tensor([1.0], device=device) * (rank + 1)
    print(f"{rank=} ({device=}) |  INITIAL: {tensor}")

    # perform an all-reduce operation to sum tensors across all GPUs
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"{rank=} ({device=}) | AFTER ALL-REDUCE: {tensor}")

    dist.destroy_process_group()    # clean up and shut down DDP process


def _value_reduce(value: float, device: torch.device) -> float:
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


def _cycle(iterable):
    """
    Infinitely cycles over an iterable object (e.g. a `DataLoader`) using a generator.
    Used over `itertools.cycle()` to prevent memory leaks for large datasets like `FineWebEdu()`.
    See: https://github.com/pytorch/pytorch/issues/23900
    """
    iterator = iter(iterable)
    while True:                         
        try:    
            yield next(iterator)        # yield the next item in the iterator
        except StopIteration:           # iterator reaches the end
            iterator = iter(iterable)   # reset the iterator


def _get_checkpoint_filename(prefix: str, epoch: int, step: int, ddp_world_size: int) -> str:
    """Returns a standardised filename string."""
    return f"{prefix}_checkpoint_gpus_{int(ddp_world_size):02d}_epoch_{int(epoch):02d}_step_{int(step):05d}"


def _load_fineweb(ddp_world_size: int = 1, ddp_rank: int = 0) -> tuple:
    """
    Loads training and validation `DataLoader` (PyTorch) objects for the `FineWebEdu()` dataset.

    Supports DDP training with a `DistributedSampler` that splits all available `global_idx` indicies 
    (defined in `__len__`) amongst GPU processes which independently handles its batch of shard loading 
    and processing. 

    All shuffling must be set to `False` to prevent constant shard loading. Utilising `self.cache()` to
    store the current shard being processed improves continuous iteration until the next shard.
    The validation set only occurs over one shard file, hence only a single shard has to be loaded once
    and so shuffling can occur without major overheads.

    Args:
    --
        `ddp_world_size` (`int`): total number of processes for Distributed Data Parallel (DDP) training. Default is `1`.
        `ddp_rank` (`int`): rank of the current process for DDP training. Default is `0`.
    
    Returns:
    --
        `tuple`: A tuple containing the training `DataLoader` and validation `DataLoader`.
    """
    # training dataset:
    train_dataset = FineWebEdu(BATCH_SIZE, BLOCK_SIZE, split="train", verbose=(ddp_rank == 0))
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
    val_dataset = FineWebEdu(BATCH_SIZE, BLOCK_SIZE, split="val", verbose=(ddp_rank == 0))
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


def _load_hellaswag(ddp_world_size: int, ddp_rank: int, split: str = "val") -> DataLoader:
    """
    Returns a HellaSwag `DataLoader` to iterate over for evaluation.

    The `DataLoader` MUST be set with `batch_size=None` as 
    each example is inherently a batch of `4` candidate ending tokens.

    With the `DistributedSampler`, if `ddp_world_size > 1` and `drop_last=True` the sampler
    will "drop the tail of the data to make it evenly divisible across the number of replicas"
    (PyTorch documentation). This will unfortunately remove `len(data_loader) % ddp_world_size`
    samples for each GPU process (if not perfectly divisible). In comparison, if `drop_last=False`
    the sampler will "add extra indices to make the data evenly divisible across the replicas" 
    (PyTorch documentation). As a result, `drop_last=True` is chosen prevent running duplicate 
    samples; the final accuracy score will only be negligibly affected.
    
    Example calculation with HellaSwag `'val'` dataset:
    - `10,042` examples split across `8` GPUs
    - with `drop_last=True`, `DistributedSampler` will create `1,255` examples per GPU
    - `1255 * 8` = `10,040` examples (`2` remainders are dropped)
    - `2 / 10,042` = `0.02%` of the total dataset is not processed (negligible).
    - if `drop_last=False`, `DistributedSampler` will create `1,256` examples per GPU
    - `1256 * 8` = `10,048` examples (`6` remainders are re-added)

    Example with code:
    ```
    # example usage of `load_hellaswag()` with DDP:
    ddp_world_size = 8
    split = "val"
    hs_loader = HellaSwag(split=split)      # without DDP
    print(f"total examples in '{split}' set: {len(hs_loader):,}")
    # using DistributedSampler with DDP:
    print(f"using DistributedSampler with {8} GPUs:")
    print(f"each GPU gets a DataLoader with the following examples:")
    total = 0
    for ddp_rank in range(ddp_world_size):
        hs_loader = load_hellaswag(ddp_world_size, ddp_rank, split)     # with DDP
        total += len(hs_loader)
        print(f"rank {ddp_rank}: {len(hs_loader):,}")
    print(f"total examples across all GPUs: {total:,}")

    >>> total examples in 'val' set: 10,042
    >>> using DistributedSampler with 8 GPUs:
    >>> each GPU gets a DataLoader with the following examples:
    >>> rank 0: 1,255
    >>> rank 1: 1,255
    >>> rank 2: 1,255
    >>> rank 3: 1,255
    >>> rank 4: 1,255
    >>> rank 5: 1,255
    >>> rank 6: 1,255
    >>> rank 7: 1,255
    >>> total examples across all GPUs: 10,040
    ```
    """
    hs_dataset = HellaSwag(split=split)
    hs_sampler = DistributedSampler(
        hs_dataset,
        num_replicas=ddp_world_size,
        rank=ddp_rank,
        shuffle=False,      # no need to shuffle for evaluation
        drop_last=True      # drop remainders with DDP, ensure non-duplicated results
    )
    hs_loader = DataLoader(
        hs_dataset,
        batch_size=None,    # must be set to None
        sampler=hs_sampler,
        pin_memory=True 
    )
    return hs_loader


def _load_shakespeare(ddp_world_size: int = 1, ddp_rank: int = 0, size: str = "tiny") -> tuple:

    """
    Loads training and validation `DataLoader` (PyTorch) iterators for the `Shakespeare()` dataset.
    Supports DDP training with a `DistributedSampler` that splits all available data indicies 
    (defined in `__len__()`) amongst GPU processes.

    See documentation within the `Shakespeare()` `Dataset` class for detailled information on the dataset.

    Args:
    --
        `ddp_world_size` (`int`): total number of processes for Distributed Data Parallel (DDP) training. Default is `1`.
        `ddp_rank` (`int`): rank of the current process for DDP training. Default is `0`.
        `size` (`str`): size of the dataset to load, either `"tiny"` or `"large"`. Default is `"tiny"`.

    Returns:
    --
        `tuple`: A tuple containing the training `DataLoader` and validation `DataLoader`.
    """
    print(f"\nloading data...\n")
    train_dataset = Shakespeare(BLOCK_SIZE, size, pct=PCT_DATA, train_split=TRAIN_SPLIT)
    train_sampler = DistributedSampler(     
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
    val_dataset = Shakespeare(BLOCK_SIZE, size, split_type="val", train_split=TRAIN_SPLIT, pct=PCT_DATA)
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


if __name__ == "__main__":
    # verify DDP setup with a simple test
    # `torchrun --standalone --nproc_per_node=8 train.py`
    _test_ddp()     