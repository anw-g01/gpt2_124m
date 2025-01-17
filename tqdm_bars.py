"""
Custom tqdm progress bars for logging stats with:
1. FineWeb-Edu Sample-10BT dataset processing with shards
2. GPT-2 (124) training loop.
"""

from tqdm import tqdm
import time    # using time.sleep() for a dummy training loop
import datetime


class tqdmGPT(tqdm):
    """
    Custom `tqdm` progress bar for the `GPT-2` training loop.
    
    Attributes:
    --
        `n_tokens` (`int`): number of tokens processed.
        `steps` (`int`): number of gradient accumulation steps (to calculate `iter/s`).
    """

    def __init__(self, *args, acc_steps: int, n_tokens: int, **kwargs):
        bar_format_str = (
            "[{bar:15}] {percentage:.1f}% | batch: {n_fmt}/{total_fmt} | "
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
        self.steps = acc_steps                  # no. of gradient accumulation steps (to find iterations rate)
        super().__init__(*args, **kwargs)       # pass to constructor of parent class

    @property
    def format_dict(self):
        d = super().format_dict     # get format dictionary from parent tqdm class
        d["n_fmt"] = f"{d['n']:,}" if d["n"] else "?"                 # current iteration (tokens processed) in billions
        d["total_fmt"] = f"{d['total']:,}" if d["total"] else "?"     # total iterations (tokens to process) in billions
        if d["rate"]:               # if rate exists (avoid division by zero)
            tpb = 1 / d["rate"]     # time per batch
            if tpb < 1:             # if less than a second
                d["s/batch"] = f"{tpb * 1e3:.1f} ms/batch"          # milliseconds per step (batch)
            else:
                d["s/batch"] = f"{tpb:.2f} s/batch"                 # in milliseconds if faster
            d["batches/s"] = f"{d['rate']:,.2f} batch/s"            # batches per second (forward + backward passes)
        else:
            d["s/batch"], d["ms/iter"], d["batches/s"] = "? s/batch", "? ms/iter", "? batches/s"
        # calculate training tokens processed per second:
        if d["elapsed"]:
            token_rate = (self.steps * self.n_tokens * self.n) / d["elapsed"]       # self.n keeps track of current iteration (from PARENT tqdm class)
            if token_rate < 1e6:
                d["tok/s"] = f"{token_rate:,.0f} tok/s"
            else:
                d["tok/s"] = f"{token_rate * 1e-6:,.2f}M tok/s"
        else:
            d["tok/s"] = "? tok/s"
        return d


class tqdmHS(tqdm):
    """
    Custom tqdm progress bar for evaluating on a HellaSwag dataset.
    Only displayed if `verbose=True` in `evaluate()` - see `hellaswag.py`.
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
    

class tqdmFW(tqdm):
    """Custom tqdm progress bar for FineWeb-Edu Sample-10BT dataset processing."""
    
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
    

def test_tqdmGPT():
    """
    Example usage of `tqdmGPT `with a dummy training loop.

    Example progress bar output (all within a single line):

    `[████-----------] 27.6% | batch: 1,021/3,705 | 2.68M tok/s | 18.2 ms/batch [00:17<00:48, 55.06 batch/s]`
    `[███████--------] 51.6% | batch: 1,912/3,705 | 2.71M tok/s | 17.8 ms/batch [00:32<00:31, 56.13 batch/s]`
    `[██████████-----] 72.7% | batch: 2,692/3,705 | 2.71M tok/s | 15.9 ms/batch [00:45<00:16, 63.04 batch/s]`
    `[██████████████-] 94.9% | batch: 3,515/3,705 | 2.71M tok/s | 16.3 ms/batch [00:59<00:03, 61.31 batch/s]`

    Example printing logs:

    ```
    training for 3 epochs (1,235 iterations/epoch)
    total: 3705 iterations

    epoch 1/3 | i:     1/3,705 ( 0.0%) | train_loss: 12.000 | val_loss: 14.000 | HellaSwag: 34.0% | [0:00:00]
    epoch 1/3 | i:   398/3,705 (10.7%) | train_loss: 10.698 | val_loss: 12.897 | HellaSwag: 35.6% | [0:00:06]
    epoch 1/3 | i:   795/3,705 (21.5%) | train_loss:  9.397 | val_loss: 11.794 | HellaSwag: 37.2% | [0:00:13]
    epoch 1/3 | i: 1,192/3,705 (32.2%) | train_loss:  8.095 | val_loss: 10.692 | HellaSwag: 38.8% | [0:00:20]

    *----- EPOCH 1/3 COMPLETE | i: 1,235/3,705 (33.3%) -----*

    epoch 2/3 | i: 1,589/3,705 (42.9%) | train_loss:  6.793 | val_loss:  9.589 | HellaSwag: 40.5% | [0:00:27]
    epoch 2/3 | i: 1,986/3,705 (53.6%) | train_loss:  5.492 | val_loss:  8.486 | HellaSwag: 42.1% | [0:00:33]
    epoch 2/3 | i: 2,383/3,705 (64.3%) | train_loss:  4.190 | val_loss:  7.383 | HellaSwag: 43.7% | [0:00:40]

    *----- EPOCH 2/3 COMPLETE | i: 2,470/3,705 (66.7%) -----*

    epoch 3/3 | i: 2,780/3,705 (75.0%) | train_loss:  2.889 | val_loss:  6.281 | HellaSwag: 45.3% | [0:00:47]
    epoch 3/3 | i: 3,177/3,705 (85.7%) | train_loss:  1.587 | val_loss:  5.178 | HellaSwag: 46.9% | [0:00:54]
    epoch 3/3 | i: 3,574/3,705 (96.5%) | train_loss:  0.285 | val_loss:  4.075 | HellaSwag: 48.5% | [0:01:00]

    *----- EPOCH 3/3 COMPLETE | i: 3,705/3,705 (100.0%) -----*

    [███████████████] 100.0% | batch: 3,705/3,705 | 3.72M tok/s | ? s/batch [00:46<00:00, ? batches/s]

    *----- TRAINING COMPLETE -----*
    ```
    """
    iters_per_epoch, epochs = 1235, 3
    total_iterations = epochs * iters_per_epoch
    print(f"\ntraining for {epochs} epochs ({iters_per_epoch:,} iterations/epoch)")
    print(f"total: {total_iterations} iterations\n")
    # print/checkpoint intervals:
    verbose = True           # print validation stats to the command window
    val_interval = 397       # interval for print logging (simulted on validation runs)

    pbar = tqdmGPT(
            iterable=range(total_iterations),
            n_tokens=11_561,
            acc_steps=4,
            miniters=1,
    )

    for i in pbar:
        
        epoch = i // iters_per_epoch    # current epoch
        local_i = i % iters_per_epoch   # current step index within current epoch
        
        train_loss, val_loss = 12-i/305, 14-i/360   # random dummy values
        hs_score = 34 + i/246
        
        # if at the end of an epoch, print a new line message to the command window:
        if local_i == iters_per_epoch - 1:                                  # end of an epoch     
            i_pct = (i + 1) / total_iterations * 100                        # percentage of total iterations so far
            pbar.refresh()    # force update the progress bar
            pbar.write(
                f"\n*----- EPOCH {epoch + 1}/{epochs} COMPLETE | "            # no. of epochs completed so far
                f"i: {i + 1:5,}/{total_iterations:,} ({i_pct:4.1f}%) -----*\n"  # iterations completed so far
            )

                # print stats to the command window on validation runs (but not if it's a the end of an epoch):
        if (i % val_interval == 0) or (local_i == iters_per_epoch - 1):                 # if validation was performed
                if verbose and not (local_i == iters_per_epoch - 1):                    # DON'T print on epoch end
                    i_pct = (i + 1) / total_iterations * 100                            # percentage of total iterations so far
                    t = datetime.timedelta(seconds=int(pbar.format_dict["elapsed"]))    # total elapsed time
                    pbar.refresh()                                                      # force update the progress bar
                    pbar.write(                                                         # print to the command window
                        f"epoch {epoch + 1}/{epochs} | "
                        f"i: {i + 1:5,}/{total_iterations:,} ({i_pct:4.1f}%) | "
                        f"train_loss: {train_loss:6.3f} | val_loss: {val_loss:6.3f} | "
                        f"HellaSwag: {hs_score:.1f}% | [{t}]"
                    )
        
        time.sleep(0.01)    # simulate a pause
    
    print("\n*----- TRAINING COMPLETE -----*")

    # additional stats:
    t = datetime.timedelta(seconds=int(pbar.format_dict["elapsed"]))    # total elapsed time
    print(f"\ntotal elapsed time: {t}")
    avg_iter_per_sec = total_iterations / t.total_seconds()
    print(f"{avg_iter_per_sec:.1f} batches/s processed ({1/avg_iter_per_sec:.2f} s/batch)")
    pbar.close()


if __name__ == "__main__":
    test_tqdmGPT()