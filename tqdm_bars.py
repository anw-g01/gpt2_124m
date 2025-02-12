from tqdm import tqdm
import time         # using time.sleep() for a dummy training loop
import datetime
import os           # for creating directories (model checkpoints testing)
import torch        # using torch.load()  
import numpy as np  # using np.save()


class tqdmGPT(tqdm):
    """
    Custom `tqdm` progress bar for monitoring progress and tracking metrics during model training.
    
    Attributes:
    --
        `n_tokens` (`int`): number of tokens processed per step.
    """

    def __init__(self, *args, n_tokens: int, **kwargs):
        bar_format_str = (
            "|{bar:15}| {percentage:.1f}% | step: {n_fmt}/{total_fmt} | "
            "{tok/s} | {s/batch} "
            "[{elapsed}<{remaining}, {batches/s}]"
        )
        params = {
            "bar_format": bar_format_str,
            "ascii": "-█",
            "miniters": 100,                    # minimum number iterations between updates
            "colour": "red"
        }
        for key, value in params.items():
            kwargs.setdefault(key, value)
        self.n_tokens = n_tokens                # no. of tokens processed
        super().__init__(*args, **kwargs)       # pass to constructor of parent class

    @property
    def format_dict(self):
        d = super().format_dict     # get format dictionary from parent tqdm class
        d["n_fmt"] = f"{d['n']:,}" if d["n"] else "?"                 # current iteration (tokens processed) in billions
        d["total_fmt"] = f"{d['total']:,}" if d["total"] else "?"     # total iterations (tokens to process) in billions
        if d["rate"]:               # if rate exists (avoid division by zero)
            tpb = 1 / d["rate"]     # time per batch
            if tpb < 1:             # if less than a second
                d["s/batch"] = f"{tpb * 1e3:.1f} ms/step"          # milliseconds per step (batch)
            else:
                d["s/batch"] = f"{tpb:.2f} s/step"                 # in milliseconds if faster
            d["batches/s"] = f"{d['rate']:,.2f} steps/s"            # batches per second (forward + backward passes)
        else:
            d["s/batch"], d["batches/s"] = "? s/step", "? steps/s"
        # calculate training tokens processed per second:
        if d["elapsed"]:
            token_rate = (self.n_tokens * self.n) / d["elapsed"]       # self.n keeps track of current iteration (from PARENT tqdm class)
            if token_rate < 1e6:
                d["tok/s"] = f"{token_rate:,.0f} tok/s"
            else:
                d["tok/s"] = f"{token_rate * 1e-6:,.2f}M tok/s"
        else:
            d["tok/s"] = "? tok/s"
        return d


class tqdmFW(tqdm):
    """
    Custom `tqdm` progress bar for monitoring progress during FineWeb-Edu `sample-10BT` dataset processing.

    Created solely for shard downloading progress in `load_fineweb.py`.
    """
    
    def __init__(self, *args, **kwargs):
        params = {
            "bar_format": "{desc}[{bar:10}] {n_fmt}/{total_fmt} ({percentage:.1f}% complete) | [{elapsed}<{remaining}, {rate_fmt}]",
            "ascii": "->=",
            "mininterval": 3,
        }
        for key, value in params.items():
            kwargs.setdefault(key, value)
        super().__init__(*args, **kwargs)       # pass to constructor of parent class

    @property
    def format_dict(self):
        d = super().format_dict
        d["n_fmt"] = f"{d['n'] * 1e-9:.2f}B" if d["n"] else "?"                 # current iteration (tokens processed) in billions
        d["total_fmt"] = f"{d['total'] * 1e-9:.2f}B" if d["total"] else "?"     # total iterations (tokens to process) in billions
        if (d["rate"] is not None) and (d["rate"] < 1e6):                       # rate of processing tokens
            d["rate_fmt"] = f"{d['rate'] * 1e-3:.2f}k tok/sec" if d["rate"] else "?"    # in thousands
        else:
            d["rate_fmt"] = f"{d['rate'] * 1e-6:.2f}M tok/sec" if d["rate"] else "?"    # in millions
        return d
    

class tqdmHS(tqdm):
    """
    Custom `tqdm` progress bar for monitoring progress during evaluation on a HellaSwag dataset.

    Created for the `hs_eval()` function in `hellaswag.py` - only displays if `verbose=True`.
    """

    def __init__(self, *args, **kwargs):
        bar_format_str = (
            "[{bar:15}] {n_fmt}/{total_fmt} ({percentage:.1f}%) | "
            "{desc} "
            "[{elapsed}<{remaining}, {rate_fmt} examples/s]"
        )
        params = {
            "bar_format": bar_format_str,
            "ascii": "-=",
            "mininterval": 1,
            "colour": "cyan"
        }
        for key, value in params.items():
            kwargs.setdefault(key, value)
        super().__init__(*args, **kwargs)    

    @property
    def format_dict(self):
        d = super().format_dict    
        d["n_fmt"] = f"{d['n']:,}" if d["n"] else "?"               # current example
        d["total_fmt"] = f"{d['total']:,}" if d["total"] else "?"   # total examples
        d["rate_fmt"] = f"{d['rate']:,.1f}" if d['rate'] else "?"   # examples processed per second
        return d
    

def test_tqdmGPT():
    """
    Example usage of `tqdmGPT()` with a dummy training loop.
    """
    
    steps_per_epoch, epochs = 239, 3
    total_steps = epochs * steps_per_epoch
    val_interval = 47               # interval for print logging (simulted on validation runs)
    checkpoint_interval = 4
    verbose = True                  # print validation stats to the command window
    log_dir = "dummy_checkpoints"   # name of directory to store dummy model checkpoints

    print(f"\nrunning dummy training loop...")
    print(f"training for {epochs} epochs ({steps_per_epoch:,} steps/epoch)")
    print(f"total: {total_steps:,} steps")
    
    # validation and checkpoint intervals:
    n_validations = total_steps // val_interval
    n_checkpoints = n_validations // checkpoint_interval
    print(f"\nrunning validation (and HellaSwag if eval=True) every {val_interval} steps (total ~{n_validations:,})")
    print(f"writing model checkpoints every {checkpoint_interval} validation runs (total ~{n_checkpoints:,}) \n")
    os.makedirs(log_dir, exist_ok=True)     # create a dummy model checkpoint directory

    pbar = tqdmGPT(
            iterable=range(total_steps),
            n_tokens=64 * 1024,     
            miniters=1,
    )

    for i in pbar:
        
        epoch = i // steps_per_epoch    # current epoch
        local_i = i % steps_per_epoch   # current step index within current epoch
        
        # using random dummy values:
        train_loss, val_loss = 10-i/207, 11-i/234 
        hs_score = 25 + i/126
        
        # if at the end of an epoch, print a new line message to the command window:
        if local_i == steps_per_epoch - 1:              # end of an epoch     
            i_pct = (i + 1) / total_steps * 100         # percentage of total iterations so far
            pbar.refresh()                              # force update the progress bar
            pbar.write(
                f"\n*----- EPOCH {epoch + 1}/{epochs} COMPLETE | "            # no. of epochs completed so far
                f"STEP: {i + 1:,}/{total_steps:,} ({i_pct:4.1f}%) -----*"  # iterations completed so far
            )

        # only store eval metrics on runs where validation is performed:
        if (i % val_interval == 0) or (local_i == steps_per_epoch - 1):                # if validation was performed
            # --- arrays would be stored here with loss and accuracy values --- #
            # print stats to the command window on validation runs (but not if it's a the end of an epoch):
            if verbose and not (local_i == steps_per_epoch - 1):                    # DON'T print on epoch end
                i_pct = (i + 1) / total_steps * 100                            # percentage of total iterations so far
                t = datetime.timedelta(seconds=int(pbar.format_dict["elapsed"]))    # total elapsed time
                pbar.refresh()                                                      # force update the progress bar
                pbar.write(                                                         # print to the command window
                    f"epoch {epoch + 1}/{epochs} | "
                    f"step: {i + 1:6,}/{total_steps:,} ({i_pct:4.1f}%) | "
                    f"train_loss: {train_loss:6.3f} | val_loss: {val_loss:6.3f} | "
                    f"HellaSwag: {hs_score:.1f}% | [{t}]"
                )
            # --- WRITE MODEL CHECKPOINT TO LOG_DIR --- #
            # write a model checkpoint every CHECKPOINT_INTERVAL validations (NOT steps) or end of epochs:
            if (i % (val_interval * checkpoint_interval) == 0) or (local_i == steps_per_epoch - 1):
                if i == 0: continue
                prefix = "end" if (local_i == steps_per_epoch - 1) else "val"   # "end" prefix for epoch end
                file_name = f"{prefix}_checkpoint_gpus_{8:02d}_epoch_{int(epoch)+1:02d}_step_{int(i + 1):05d}"
                checkpoint_dir = os.path.join(log_dir, file_name)
                if verbose:
                    pbar.write(f'\nwriting checkpoint: "{file_name}"\n')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint.pt")      # path to save PyTorch model weights
                checkpoint = {      
                    "epoch": epoch + 1, 
                    "i": i,
                    "step": i + 1,
                    "model_state_dict": dict(),   # empty dictionary as model state_dict
                }
                # save model checkpoint and a dummy numpy array file:
                torch.save(checkpoint, checkpoint_path)     
                losses = np.random.rand(total_steps)
                np.save(os.path.join(checkpoint_dir, "random_arr.npy"), losses) 

        time.sleep(0.12)    # simulate a pause after every step
    
    print("\n*----- TRAINING COMPLETE -----*")

    # additional stats:
    t = datetime.timedelta(seconds=int(pbar.format_dict["elapsed"]))    # total elapsed time
    print(f"total elapsed time: {t}")
    avg_iter_per_sec = total_steps / t.total_seconds()
    print(f"{avg_iter_per_sec:.1f} batches/s processed ({1/avg_iter_per_sec:.2f} s/batch)")
    pbar.close()


if __name__ == "__main__":
    test_tqdmGPT()

    """
    EXAMPLE VERBOSE LOGGING ON COMMAND LINE:

    training for 3 epochs (239 iterations/epoch)
    total: 717 iterations

    running validation (and/or HellaSwag eval) every 47 steps (total ~15)
    writing model checkpoints every 4 validation runs (total ~3) 

    epoch 1/3 | step:      1/717 ( 0.1%) | train_loss: 10.000 | val_loss: 11.000 | HellaSwag: 25.0% | [0:00:00]
    epoch 1/3 | step:     48/717 ( 6.7%) | train_loss:  9.773 | val_loss: 10.799 | HellaSwag: 25.4% | [0:00:05]
    epoch 1/3 | step:     95/717 (13.2%) | train_loss:  9.546 | val_loss: 10.598 | HellaSwag: 25.7% | [0:00:12]
    epoch 1/3 | step:    142/717 (19.8%) | train_loss:  9.319 | val_loss: 10.397 | HellaSwag: 26.1% | [0:00:18]
    epoch 1/3 | step:    189/717 (26.4%) | train_loss:  9.092 | val_loss: 10.197 | HellaSwag: 26.5% | [0:00:24]

    writing checkpoint: val_checkpoint_gpus_08_epoch_01_step_00189

    epoch 1/3 | step:    236/717 (32.9%) | train_loss:  8.865 | val_loss:  9.996 | HellaSwag: 26.9% | [0:00:30]

    *----- EPOCH 1/3 COMPLETE | STEP: 239/717 (33.3%) -----*

    writing checkpoint: end_checkpoint_gpus_08_epoch_01_step_00239

    epoch 2/3 | step:    283/717 (39.5%) | train_loss:  8.638 | val_loss:  9.795 | HellaSwag: 27.2% | [0:00:36]
    epoch 2/3 | step:    330/717 (46.0%) | train_loss:  8.411 | val_loss:  9.594 | HellaSwag: 27.6% | [0:00:42]
    epoch 2/3 | step:    377/717 (52.6%) | train_loss:  8.184 | val_loss:  9.393 | HellaSwag: 28.0% | [0:00:48]

    writing checkpoint: val_checkpoint_gpus_08_epoch_02_step_00377

    epoch 2/3 | step:    424/717 (59.1%) | train_loss:  7.957 | val_loss:  9.192 | HellaSwag: 28.4% | [0:00:54]
    epoch 2/3 | step:    471/717 (65.7%) | train_loss:  7.729 | val_loss:  8.991 | HellaSwag: 28.7% | [0:01:00]

    *----- EPOCH 2/3 COMPLETE | STEP: 478/717 (66.7%) -----*

    writing checkpoint: end_checkpoint_gpus_08_epoch_02_step_00478

    epoch 3/3 | step:    518/717 (72.2%) | train_loss:  7.502 | val_loss:  8.791 | HellaSwag: 29.1% | [0:01:07]
    epoch 3/3 | step:    565/717 (78.8%) | train_loss:  7.275 | val_loss:  8.590 | HellaSwag: 29.5% | [0:01:13]

    writing checkpoint: val_checkpoint_gpus_08_epoch_03_step_00565

    epoch 3/3 | step:    612/717 (85.4%) | train_loss:  7.048 | val_loss:  8.389 | HellaSwag: 29.8% | [0:01:19]
    epoch 3/3 | step:    659/717 (91.9%) | train_loss:  6.821 | val_loss:  8.188 | HellaSwag: 30.2% | [0:01:25]
    epoch 3/3 | step:    706/717 (98.5%) | train_loss:  6.594 | val_loss:  7.987 | HellaSwag: 30.6% | [0:01:31]

    *----- EPOCH 3/3 COMPLETE | STEP: 717/717 (100.0%) -----*

    writing checkpoint: end_checkpoint_gpus_08_epoch_03_step_00717

    |███████████████| 100.0% | step: 717/717 | 504,765 tok/s | ? s/step [01:33<00:00, ? steps/s]        

    *----- TRAINING COMPLETE -----*
    total elapsed time: 0:01:33
    7.7 batches/s processed (0.13 s/batch)
    """