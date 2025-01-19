import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import tiktoken


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
    

if __name__ == "__main__":

    # ----- DataLoader examples with TinyShakespeare ----- #

    chunk_sampling = False      # batch processing method

    if chunk_sampling:
        print(f"\nutilising chunk sampling (non-overlapping batches)")
        train_loader = DataLoader(
            TinyShakespeare(block_size=1024, batch_size=16),
            batch_size=None,    # must be set to None
            shuffle=False,
        )
    else:
        print(f"\nutilising overlapping samples across batches")
        train_loader = DataLoader(
            TinyShakespeare(block_size),    # NO specified batch_size within Dataset class
            batch_size=batch_size,          # specify DataLoader batch size parameter
            shuffle=False
        )
    print(f"\ntokens per batch: {batch_size * block_size:,} (batch size {batch_size:,})")
    print(f"{len(train_loader):,} available batches per epoch")
    
    X, y = next(iter(train_loader))
    print(X.shape, y.shape)             # shape --> [batch_size, block_size]

    # --- Usage with DistributedSampler:

    print(f"\nusing DistributedSampler with DataLoader..")
    train_dataset = TinyShakespeare(        # load custom Dataset class for training
        block_size=block_size,
        verbose=False
    )
    train_sampler = DistributedSampler(     # for DDP: divides the dataset into equal-sized chunks across all GPUs (processes)
        train_dataset,
        num_replicas=8,                     # total no. of processes (using 8 GPUs as an example)
        rank=0,                             # current GPU integer ID (using the first GPU as an example)
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
    print("\navailable batches per epoch: ", end="")
    print(f"{len(train_loader):,} (without) | {len(train_loader_wDS):,} (with)")