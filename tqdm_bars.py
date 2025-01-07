"""
Custom tqdm progress bars for logging stats with:
1. FineWeb-Edu Sample-10BT dataset processing with shards
2. GPT-2 (124) training loop.
"""

from tqdm import tqdm

class tqdmFW(tqdm):
    """Custom and optional tqdm progress bar for FineWeb-Edu Sample-10BT dataset processing."""
    
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