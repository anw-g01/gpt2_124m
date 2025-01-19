import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import os
from ..config import DATA_ROOT
from ..train import _cycle   
from load_fineweb import SHARD_SIZE, LAST_SHARD_SIZE


class FineWebEdu(Dataset):
    """
    A custom PyTorch `torch.utils.data.Dataset` class for FineWeb-Edu (`sample-10BT`) dataset shards.
    Handles the loading of tokenized dataset shards stored as NumPy arrays in `DATA_ROOT`.
    Supports batching of data within a shard file as well as across shard boundaries.
    
    Link to HuggingFace dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

    N.B. Shards should already be downloaded and stored (see `load_fineweb.py`). Only the first shard, 
    with `SHARD_SIZE` (`100M`) tokens, is used for the validation set. The training set uses the 
    remaining `99` shards. The first `98` shards of the training set hold `100M` tokens each, 
    while the last shard holds exactly `LAST_SHARD_SIZE` (`82,590,278`) tokens.

    Args:
    --
        `batch_size` (`int`): Number of mini-batch samples to use in a forward pass.
        `block_size` (`int`): Total context (sequence) length per mini-batch.
        `split` (`str`): Specifies the dataset split, either `"train"` or `"val"`. Default is `"train"`.
        `root` (`str`): Directory where the data shards are stored. Default is `DATA_ROOT` defined in `config.py`.
        `verbose` (`bool`): If `True`, prints number of shard files found during loading.

    Methods:
    --
        `_get_shard_paths()`: Gets shard file names from the root directory based on the `split` and constructs their full paths.
        `_load_shard(shard_idx: int)`: Loads a single shard as a PyTorch tensor of tokens based on the `shard_idx`.
        `__getitem__(idx: int)`: Returns a single batch of input and target sequences based on `idx` (a batch index).
        `__len__()`: Returns the total number of batches in the dataset, accounting for `LAST_SHARD_SIZE` if `split=Train`.
    """

    def __init__(
            self,
            batch_size: int,
            block_size: int,
            split: str = "train",
            dir: str = DATA_ROOT,
            verbose: bool = True
        ):
        assert split.lower() in ["train", "val"], "split must be either 'train' or 'val'"
        self.batch_size = batch_size    # no. of samples to user in a forward pass
        self.block_size = block_size    # context (sequence) length
        self.split = split              # split to specify __len__() method
        self.root = dir                 # specify directory where the data shards are stored in config.py
        self.verbose = verbose
        self.shards = self._get_shard_paths()    # load shards from directory based on split
        self.cache = {}                 # cache for validation set only

    def _get_shard_paths(self):
        """Get shard file names from the root directory (based on the split) to construct their full paths."""
        names = [f for f in os.listdir(self.root) if self.split in f]       # get individual shard file names
        shards = [os.path.join(self.root, f) for f in names]               # construct full paths, sorted by name (ascending order)
        assert len(shards) > 0, f"no shards found for split='{self.split}' in {self.root}"
        if self.verbose:
            if self.split == "val":      # validation set (single shard with 100M tokens)
                n = len(shards) * SHARD_SIZE * 1e-9
            else:
                n = ((len(shards) - 1) * SHARD_SIZE + LAST_SHARD_SIZE) * 1e-9
            print(f'found {len(shards):,} shard(s) for "{self.split}" split ({n:.2f} B tokens)')
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
        """Returns a single batch of input and target sequences based on `idx` - a batch index."""
        chunk_size = self.batch_size * self.block_size              # chunk size (tokens in one batch)
        global_idx = idx * chunk_size                               # starting position (across all shards)
        
        # determine corresponding shard index from global index:
        shard_idx = global_idx // SHARD_SIZE                                # get index of current shard file by the default shard size of 100M
        if shard_idx == len(self.shards) - 1:                               # if inside the last shard
            # subtract the total tokens in all previous shards (98 shards × 100M) to get the position in the last shard:
            local_idx = global_idx - ((len(self.shards) - 1) * SHARD_SIZE)  # index within the LAST shard
        else:
            local_idx = global_idx % SHARD_SIZE                             # index within all other shards 
        
        tokens = self._load_shard(shard_idx)                    # load the corresponding shard tokens
        if local_idx + chunk_size >= tokens.shape[0]:           # if the current shard will be exhausted
            X1 = tokens[local_idx:]                             # store available tokens of current shard
            y1 = tokens[local_idx + 1:]                         # corresponding target sequence (next token for each sample)
            shard_idx += 1                                      # move to the next shard (no circular indexing as the last shard will never cross boundaries)
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
        else:       # normal case (no shard boundary crossing)
            X = tokens[local_idx: local_idx + chunk_size]                   # get the input sequence
            y = tokens[local_idx + 1: local_idx + chunk_size + 1]           # get the target sequence (next token for each sample)
        return X.view(self.batch_size, -1), y.view(self.batch_size, -1)     # return with shapes [batch_size, block_size]

    def __len__(self):
        """
        Example calculations for `batch_size=16` and `block_size=1024`:
        ```
        >> idx=598,143 | global_idx=9,799,974,912 | shard_idx=97 | local_idx=99,974,912 | shard 97 size: 100.0M
        >> crossing shard: 97 -> 98 | local_idx=99,991,296 | rem=7,680 | shard 98 size: 82.6M

        >> idx=603,184 | global_idx=9,882,566,656 | shard_idx=98 | local_idx=82,566,656 | shard 98 size: 82.6M
        ```
        This shows that a final `local_idx` of `82,566,656 + 16,284 = 82,583,040` in the last shard was reached,
        resulting in a leftover of `82,590,278 - 82,583,040 = 7,238` tokens. These last remaining `7,238` tokens
        will never be used as they are less than a chunk size of `batch_size * block_size = 16,384` and because
        the total number of batches was calculated to fit within all available tokens using integer division.
        """
        chunk_size = self.batch_size * self.block_size
        if self.split == "val":
            return (len(self.shards) * SHARD_SIZE) // chunk_size   
        # else, for the training set, account for the last shard holding 82,590,278 tokens 
        return ((len(self.shards) - 1) * SHARD_SIZE + LAST_SHARD_SIZE) // chunk_size     


if __name__ == "__main__":

    batch_size = 64             # samples per forward pass
    block_size = 1024           # context length
    # simulate DDP:
    ddp_world_size = 8          # no. of GPU processes
    ddp_local_rank = 0          # run as the first GPU only

    # ----- DataLoader EXAMPLES with FineWebEdu Sample-10BT ----- #

    train_dataset = FineWebEdu(batch_size, block_size, split="train")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=ddp_world_size,
        rank=ddp_local_rank,
        shuffle=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,            # must be set to None
        sampler=train_sampler,      # using a DistributedSampler
        pin_memory=True,
        shuffle=False
    )

    print(f"\ntokens per mini-batch: {batch_size * block_size:,} (mini-batch size {batch_size:,})")
    print(f"{len(train_loader):,} available mini-batches per epoch (per GPU)\n")    # per GPU if using DDP
    
    train_iter = _cycle(train_loader)

    # example traversal through one epoch of the DataLoader
    n = len(train_loader) * 2
    for i in range(n):
        X, y = next(train_iter)     # get next X, y batch
        progress_str = (
            f"\rbatch: {i + 1:,}/{n:,} | "
            f"{X.shape, y.shape}"
        )
        print(progress_str, end="")