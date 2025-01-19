import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import tiktoken
import os
from typing import Optional


class Shakespeare(Dataset):
    """
    A custom PyTorch `torch.utils.data.Dataset` class for loading two different Shakespeare datasets from text files:
    - `"tiny"`: `shakespeare_1.1M_chars.txt` (`~300k` GTP-2 tokens)
    - `"large"`: `shakespeare_5.4M_chars.txt` (`~1.6M` GTP-2 tokens)
    
    Links to datasets:
    - `tiny`: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    - `large`: https://gist.githubusercontent.com/blakesanie/dde3a2b7e698f52f389532b4b52bc254/raw/76fe1b5e9efcf0d2afdfd78b0bfaa737ad0a67d3/shakespeare.txt
    
    Implements both overlapping and non-overlapping samples within batches.
    If `batch_size=None`, overlapping sampling advances by a `+1` sliding window
    to create `(self.tokens.shape[0] - self.block_size) / self.batch_size` samples.

    With a specified `batch_size`, chunk sampling will advance by an index of
    `self.batch_size * self.block_size` with manual batch construction using
    `.view(self.batch_size, -1)`. The `batch_size` parameter WITHIN a `DataLoader`
    object must be set to `None` if using chunk sampling.

    Args:
    --
        `block_size` (`int`): Context (sequence) length.
        `size` (`str`): While text file to load, either `"tiny"` or `"large"`.
        `batch_size` (`Optional[int]`): Batch size for chunk sampling, otherwise use built-in PyTorch `DataLoader` batching.
        `split_type` (`str`): Which type of data split to load, either `"train"` or `"val"`. Default is `"train"`.
        `train_split` (`float`): Ratio of training data split. Default is `0.9`.
        `pct` (`float`): Percentage of the full dataset to use. Default is `1.0`.
        `verbose` (`bool`): Whether to print dataset information. Default is `False`.

    Methods:
    --
        `__len__()`: Returns the number of samples in the dataset.
        `__getitem__(idx: int)`: Returns batched input and target sequences if `batch_size` is specified, for the given index.
        `_tokenize()`: Tokenizes the Shakespeare dataset based on the specified size, percentage, and split type.
    """

    def __init__(
        self,
        block_size: int,
        size: str = "tiny",
        batch_size: Optional[int] = None,
        split_type: str = "train",
        train_split: float = 0.9,
        pct: float = 1.0,
        verbose: bool = False
    ):
        assert size.lower() in ["tiny", "large"], 'size must be either "tiny" or "large"'
        assert split_type.lower() in ["train", "val"], 'split_type must be either "train" or "val"'
        self.block_size = block_size        # context (sequence) length
        self.size = size
        self.split_type = split_type
        self.train_split = train_split
        self.pct = pct
        self.verbose = verbose
        self.tokens = self._tokenize()      # tokenize the text file and return as a tensor of tokens
        # ---
        self.batch_size = batch_size

    def __len__(self):
        if self.tokens.shape[0] == 0:   # if no items (only for 0 % validation split)
            return 0
        if self.batch_size is None:
            return self.tokens.shape[0] - self.block_size   # sliding window (overlapping samples)
        return (self.tokens.shape[0] - self.block_size) // (self.block_size * self.batch_size)

    def __getitem__(self, idx: int):
        if self.batch_size is None:
            X = self.tokens[idx: idx + self.block_size]
            y = self.tokens[idx + 1: idx + self.block_size + 1]
            return X, y
        # carry out MANUAL BATCHING (set batch_size=None in the DataLoader object)
        chunk = self.batch_size * self.block_size
        curr = idx * chunk      # position based on idx (previously removed statefullness)
        X = self.tokens[curr: curr + chunk]
        y = self.tokens[curr + 1: curr + chunk + 1]
        return X.view(self.batch_size, -1), y.view(self.batch_size, -1)
    
    def _tokenize(self):
        """
        Tokenizes the Shakespeare dataset based on the specified size, percentage, and split type.
        Reads the Shakespeare text file, encodes it into tokens using the GPT-2 tokenizer,
        and splits the tokens into training or validation sets based on the given split ratio.
        """
        file_name = "shakespeare_1.1M_chars.txt" if self.size == "tiny" else "shakespeare_5.4M_chars.txt"
        file_path = os.path.join("data", file_name)     # Shakespeare text files stored inside 'data' directory
        with open(file_path, "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = torch.tensor(enc.encode(text), dtype=torch.long)
        # use a percentage of the full dataset if specified:
        tokens = tokens[:int(self.pct * len(tokens))]     
        # split tokens into a train or validation set by the given split:
        n_train = int(self.train_split * len(tokens))
        tokens = tokens[:n_train] if self.split_type == "train" else tokens[n_train:]
        if self.verbose:
            n_test = len(tokens) - n_train
            print(f"no. of tokens: {len(tokens):,} ({self.pct * 100:.0f}% of full data)")
            print(f"train/val split: {n_train:,} ({self.train_split * 100:.0f}%), {n_test:,} ({(1 - self.train_split) * 100:.0f}%)")
            print(f"sequence (context) length: {block_size:,} tokens")
        return tokens
    

if __name__ == "__main__":

    # ----- DataLoader examples with TinyShakespeare ----- #

    chunk_sampling = False      # batch processing method
    batch_size = 16             # batch size
    block_size = 1024           # context (sequence) length
    size = "large"              # dataset size
    # simulate DDP:
    ddp_world_size = 8          # no. of GPU processes
    ddp_local_rank = 0          # run as the first GPU only

    if chunk_sampling:
        print(f"\nutilising chunk sampling (non-overlapping batches)")
        train_loader = DataLoader(
            Shakespeare(block_size, size, batch_size=batch_size, verbose=True),     # specify batch_size
            batch_size=None,        # DataLoader batch_size parameter must be set to None
            shuffle=False,
        )
    else:
        print(f"\nutilising overlapping samples across batches")
        train_loader = DataLoader(
            Shakespeare(block_size, size, verbose=True),    # NO specified batch_size within Dataset class
            batch_size=batch_size,      # specify DataLoader batch_size parameter
            shuffle=False
        )
    print(f"\ntokens per batch: {batch_size * block_size:,} (batch size {batch_size:,})")
    print(f"{len(train_loader):,} available batches per epoch")
    
    X, y = next(iter(train_loader))
    print(X.shape, y.shape)             # shape --> [batch_size, block_size]

    # ----- Usage with DistributedSampler ----- #

    print(f"\n------------")
    print(f"using DistributedSampler with DataLoader and overlapping samples...")
    train_dataset = Shakespeare(block_size, size)
    train_sampler = DistributedSampler(     # for DDP: divides the dataset into equal-sized chunks across all GPUs (processes)
        train_dataset,
        num_replicas=ddp_world_size,                     # total no. of processes (using 8 GPUs as an example)
        rank=ddp_local_rank,                             # current GPU integer ID (using the first GPU as an example)
        shuffle=True
    )
    train_loader_wDS = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,              # WITH DistributedSampler
    )
    train_loader= DataLoader(               # WITHOUT Distributed
        train_dataset,
        batch_size=batch_size,
        shuffle=True,                       
    )
    print("available batches per epoch: ", end="")
    print(f"{len(train_loader_wDS):,} (with) | {len(train_loader):,} (without)")