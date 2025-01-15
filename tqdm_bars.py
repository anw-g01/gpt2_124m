"""
Custom tqdm progress bars for logging stats with:
1. FineWeb-Edu Sample-10BT dataset processing with shards
2. GPT-2 (124) training loop.
"""

from tqdm import tqdm
import time    # using time.sleep() for a dummy training loop
import datetime


class tqdmGPT(tqdm):
    """Custom `tqdm` progress bar for the `GPT-2` training loop."""

    def __init__(self, *args, acc_steps: int, n_tokens: int, **kwargs):
        bar_format_str = (
            "[{bar:15}] {percentage:.1f}% | iter: {n_fmt}/{total_fmt} | "
            "{tok/s} | {s/batch} ({ms/iter}) "
            "[{elapsed}<{remaining}, {batches/s}] | {desc}"
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
            d["ms/iter"] = f"{tpb / self.steps * 1e3:.1f} ms/iter"  # milliseconds per iter (per accumulation step)
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
    

if __name__ == "__main__":

    # Example usage of tqdmGPT progress bar with a dummy training loop:

    n_batches, n_epochs = 100, 3
    n_iterations = n_batches * n_epochs
    print(f"\ntraining for {n_epochs} epochs, with {n_batches:,} iterations per epoch...")
    print(f"total iterations in dummy loop: {n_batches:,}\n")

    pbar = tqdmGPT(
            iterable=range(n_iterations),
            n_tokens=241_561,
            acc_steps=4,
            miniters=1,
    )
    for i in pbar:
        epoch = i // n_batches    # current epoch
        local_i = i % n_batches   # current step index within current epoch
        train_loss, val_loss = 2-i/340, 3-i/270   # random dummy values
        hs_score = 34 + i/246
        if local_i == n_batches - 1:
            pct = (i + 1) / n_iterations * 100
            t = datetime.timedelta(seconds=int(pbar.format_dict["elapsed"]))    # total elapsed time
            pbar.refresh()
            pbar.write(
                f"\nepoch {epoch + 1}/{n_epochs} complete | i: {i + 1:,}/{n_iterations:,} ({pct:.1f}%) | "
                f"train_loss: {train_loss:.3f} | val_loss: {val_loss:.3f} | "
                f"HellaSwag: {hs_score:.1f}% | elapsed: [{t}]\n"
            )
        if i % 1000 == 0:
            pbar.set_description_str(
                f"t_loss: {train_loss:.3f} | "   # dummy values to test
                f"v_loss: {val_loss:.3f} "  
            )
        time.sleep(0.01)    # simulate a pause
    pbar.close()