import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import tiktoken
import numpy as np
import os
from config import DATA_ROOT

# total no. of tokens in FineWebEdu sample-10BT: 9_982_590_278
from fineweb import SHARD_SIZE, LAST_SHARD_SIZE, TOTAL_TOKENS    
TRAIN_TOKENS = TOTAL_TOKENS - SHARD_SIZE    # total tokens in training set: 9_882_590_278


class TinyShakespeare(Dataset):
    """
    PyTorch Dataset for the Tiny Shakespeare dataset, found in `input.txt`.

    Implements both overlapping and non-overlapping samples within batches.
    If `batch_size=None`, overlapping sampling advances by a `+1` sliding window
    to create `(self.tokens.shape[0] - self.block_size) / self.batch_size` samples.

    With a specified `batch_size`, chunk sampling will advance by an index of
    `self.batch_size * self.block_size` with manual batch construction using
    `.view(self.batch_size, -1)`. The `batch_size` parameter WITHIN a `DataLoader`
    object must be set to `None` if using chunk sampling.

    Dataset link: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    """

    def __init__(self, block_size: int, batch_size=None, pct=1, split="train", train_split=0.9, verbose=True):
        assert split.lower() in ["train", "val"]
        with open("input.txt", "r") as f:   # in the same directory
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        self.data = torch.tensor(enc.encode(text), dtype=torch.long)
        self.data = self.data[:int(pct * len(self.data))]
        self.block_size = block_size    # context length
        n_train = int(train_split * len(self.data))
        self.tokens = self.data[:n_train] if split == "train" else self.data[n_train:]
        if verbose:
            n_test = len(self.data) - n_train
            print(f"no. of tokens: {len(self.data):,} ({pct * 100:.0f}% of full data)")
            print(f"train/val split: {n_train:,} ({train_split * 100:.0f}%), {n_test:,} ({(1 - train_split) * 100:.0f}%)")
            print(f"sequence (context) length: {block_size:,} tokens")
        # ---
        self.batch_size = batch_size

    def __len__(self):
        if self.tokens.shape[0] == 0:   # if no items (only for 0 % validation split)
            return 0
        if self.batch_size is None:
            return self.tokens.shape[0] - self.block_size   # sliding window (overlapping samples)
        return (self.tokens.shape[0] - self.block_size) // (self.block_size * self.batch_size)

    def __getitem__(self, idx):
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
    

class FineWebEdu(Dataset):
    """
    PyTorch Dataset for FineWeb-Edu (`sample-10BT`) dataset shards.

    Handles the loading of tokenized dataset shards stored as NumPy arrays in `DATA_ROOT`.
    Supports batching of data within a shard file as well as across shard boundaries.
    Returns input (`X`) and target (`y`) with shape `[batch_size, block_size]`.
    """

    def __init__(self, batch_size: int, block_size: int, split="train", dir=DATA_ROOT, verbose=True):
        self.verbose = verbose
        assert split.lower() in ["train", "val"], "split must be either 'train' or 'val'"
        self.split = split              # split to specify __len__() method
        self.batch_size = batch_size    # no. of samples to user in a forward pass
        self.block_size = block_size    # context (sequence) length
        self.root       = dir           # specify directory where the data shards are stored in config.py
        self.shards = self._get_shard_paths(split)    # load shards from directory based on split
        self.cache      = {}            # cache for validation set only

    def _get_shard_paths(self, split):
        """Get shard file names from the root directory (based on the split) to construct their full paths."""
        names = [f for f in os.listdir(self.root) if split in f]       # get individual shard file names
        shards = [os.path.join(self.root, f) for f in names]               # construct full paths, sorted by name (ascending order)
        assert len(shards) > 0, f"no shards found for split='{split}' in {self.root}"
        if self.verbose:
            print(f"found {len(shards):,} shards for '{split}' split")
        return shards   # return list of full paths to shards

    def _load_shard(self, shard_idx: int):
        """Loads a single shard as a PyTorch tensor of tokens, based on the `shard_idx`."""
        path = self.shards[shard_idx]       # get the full shard path
        if path in self.cache:              # check if shard is already loaded
            return self.cache[path]         # return the cached shard
        self.cache = {}                     # clear the cache (high memory consumption)
        arr = np.load(path)                             # load the shard as a numpy array
        tokens = torch.tensor(arr, dtype=torch.long)    # convert to PyTorch tensor with int64 dtype
        self.cache[path] = tokens                       # cache the loaded shard
        return tokens
    
    def __getitem__(self, idx: int):
        """Returns a single batch of input and target sequences based on the current index."""
        chunk_size = self.batch_size * self.block_size              # chunk size (tokens in one batch)
        global_idx = idx * chunk_size                               # starting position (across all shards)
        # determine corresponding shard index from global index:
        shard_idx = global_idx // SHARD_SIZE                        # get index of current shard file by the default shard size of 100M
        tokens = self._load_shard(shard_idx)                        # load the corresponding shard tokens
        # determine local index within the shard:
        if shard_idx == (len(self.shards) - 1):                         # if it's the last shard (which has <100M tokens)
            local_idx = global_idx - (TRAIN_TOKENS - LAST_SHARD_SIZE)   # local index within the last shard after 9.8B tokens
        else:
            local_idx = global_idx % SHARD_SIZE                         # index within others shard (espeically for last index which has <100M tokens)

        if local_idx + chunk_size >= tokens.shape[0]:           # if the current shard will be exhausted
            X1 = tokens[local_idx:]                             # store available tokens of current shard
            y1 = tokens[local_idx + 1:]                         # corresponding target sequence (next token for each sample)
            shard_idx = (shard_idx + 1) % len(self.shards)      # move to the next shard (circular indexing)
            tokens = self._load_shard(shard_idx)                # load the next shard
            rem = chunk_size - X1.shape[0]                      # remaining tokens needed for X to complete the chunk 
            
            if rem == 0:                            # if X was exactly filled but y wasn't
                X = X1                              # X is unchanged 
                y2 = tokens[:1]                     # y is missing one token (due to starting index +1)
                y = torch.cat((y1, y2), dim=0)      # concatenate y tensor
            elif rem > 0:                           # if X and y both need to be filled
                X2 = tokens[: rem]                  # get the remaining tokens from next shard to complete chunk
                y2 = tokens[: rem + 1]              # corresponding target sequence
                X = torch.cat((X1, X2), dim=0)      # concatenate X
                y = torch.cat((y1, y2), dim=0)      # concatenate y  
            else:
                X, y = X1, y1                       # X, y are unchanged (already filled)
            # print(f"--> {shard_idx=}, {local_idx=:,}, {rem=:,}")   # print for debugging
        else:                                                   # normal case (no shard boundary crossing)
            X = tokens[idx: idx + chunk_size]                   # get the input sequence
            y = tokens[idx + 1: idx + chunk_size + 1]           # get the target sequence (next token for each sample)
        return X.view(self.batch_size, -1), y.view(self.batch_size, -1)     # return with shapes [batch_size, block_size]

    def __len__(self):
        """
        Returns the number of available batches in one epoch.
        N.B. only 100M tokens (SHARD_SIZE) used for the validation set.
        Remaining tokens used for the training set.
        """
        if self.split == "val":
            return SHARD_SIZE // (self.batch_size * self.block_size)
        return ((TOTAL_TOKENS - SHARD_SIZE) // (self.batch_size * self.block_size)) 


if __name__ == "__main__":

    batch_size = 16             # samples per forward pass
    block_size = 1024           # context length

    # ----- DataLoader EXAMPLES with FineWebEdu Sample-10BT ----- #

    train_dataset = FineWebEdu(batch_size, block_size, split="train")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=8,
        rank=1,
        shuffle=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,            # must be set to None
        # sampler=train_sampler,      # using a DistributedSampler
        pin_memory=True,
    )

    print(f"\ntokens per batch: {batch_size * block_size:,} (batch size {batch_size:,})")
    print(f"{len(train_loader):,} available batches per epoch\n")
    
    train_iter = iter(train_loader)

    # example traversal through on epoch of the DataLoader
    n = len(train_loader) + (16 * 1024)
    for i in range(n):
        X, y = next(train_iter)
        progress_str = (
            f"\rbatch: {i + 1:,}/{n:,} | "
            f"{X.shape, y.shape} | "
        )
        print(progress_str, end="") 


    # ----- DataLoader EXAMPLES with FineWebEdu Sample-10BT ----- #

    # print(f"creating DataLoader for FineWebEdu Sample-10BT dataset..\n")
    # train_loader = DataLoader(
    #     FineWebEdu(batch_size, block_size, split="train"),
    #     batch_size=None,    # must be set to None
    #     shuffle=False,      # iterate through shards sequentially if shuffling=False
    # )

    # print(f"\ntokens per batch: {batch_size * block_size:,} (batch size {batch_size:,})")
    # print(f"{len(train_loader):,} available batches per epoch\n")
    
    # train_iter = iter(train_loader)

    # # example traversal through on epoch of the DataLoader
    # n = len(train_loader)
    # for i in range(n):
    #     X, y = next(train_iter)
    #     progress_str = (
    #         f"\rbatch: {i + 1:,}/{n:,} | "
    #         f"{X.shape, y.shape}"
    #     )
    #     print(progress_str, end="") 

    #-------------------------------------------------------
    # ----- DataLoader examples with TinyShakespeare ----- #

    # chunk_sampling = False      # batch processing method

    # if chunk_sampling:
    #     print(f"\nutilising chunk sampling (non-overlapping batches)")
    #     train_loader = DataLoader(
    #         TinyShakespeare(block_size=1024, batch_size=16),
    #         batch_size=None,    # must be set to None
    #         shuffle=False,
    #     )
    # else:
    #     print(f"\nutilising overlapping samples across batches")
    #     train_loader = DataLoader(
    #         TinyShakespeare(block_size),    # NO specified batch_size within Dataset class
    #         batch_size=batch_size,          # specify DataLoader batch size parameter
    #         shuffle=False
    #     )
    # print(f"\ntokens per batch: {batch_size * block_size:,} (batch size {batch_size:,})")
    # print(f"{len(train_loader):,} available batches per epoch")
    
    # X, y = next(iter(train_loader))
    # print(X.shape, y.shape)             # shape --> [batch_size, block_size]

    # # --- Usage with DistributedSampler:

    # print(f"\nusing DistributedSampler with DataLoader..")
    # train_dataset = TinyShakespeare(        # load custom Dataset class for training
    #     block_size=block_size,
    #     verbose=False
    # )
    # train_sampler = DistributedSampler(     # for DDP: divides the dataset into equal-sized chunks across all GPUs (processes)
    #     train_dataset,
    #     num_replicas=8,                     # total no. of processes (using 8 GPUs as an example)
    #     rank=0,                             # current GPU integer ID (using the first GPU as an example)
    #     shuffle=True
    # )
    # train_loader_wDS = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     sampler=train_sampler,              # WITH DistributedSampler
    # )
    # train_loader= DataLoader(               # WITHOUT Distributed
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,                       
    # )
    # print("\navailable batches per epoch: ", end="")
    # print(f"{len(train_loader):,} (without) | {len(train_loader_wDS):,} (with)")
