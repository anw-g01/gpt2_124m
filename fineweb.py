"""
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Process the FineWeb-Edu Sample-10BT dataset by tokenzing documents and saving them in fixed-size shards.
Each shard is saved as a NumPy array with 100M tokens (available space), of data type `uint16`.
"""

import numpy as np
import tiktoken
import os
import sys
import multiprocessing as mp
import datasets
import time
from tqdm import tqdm

# ------ GLOBAL VARIABLES ------ #

LOCAL_DIR = "fineweb-edu-10BT"                  # directory to store data shards
DATASET_NAME = "HuggingFaceFW/fineweb-edu"      # HuggingFace dataset name
REMOTE_NAME = "sample-10BT"                     # specific subset
SHARD_SIZE = int(1e8)                           # 100M tokens/shard (100 total shards)

# construct a full path to the local data cache directory
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), LOCAL_DIR)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)          # create the directory if it doesn't exist
ENCODER = tiktoken.get_encoding("gpt2")             # initialise GPT-2 tokenizer 
EOT = ENCODER._special_tokens["<|endoftext|>"]      # end-of-text (EOT) token 

# ------ HELPER FUNCTIONS ------ #

def tokenize(row: dict) -> np.array:
    """
    Tokenizes the text (key) from a single document (row from dataset) and returns a NumPy array of `uint16` tokens.
    """
    tokens = [EOT]     # start token array with end-of-text (EOT) token
    tokens.extend(ENCODER.encode_ordinary(row["text"]))
    tokens = np.array(tokens)                               # convert to a NumPy array
    # check tokens are within the valid range of 16-bit unsigned integers
    assert (0 <= tokens).all() and (tokens < 2 ** 16).all(), "Invalid tokens for for dtype uint16."
    return tokens.astype(np.uint16)


def write_datafile(tokens: np.array, shard_idx: int) -> None:
    split = "val" if shard_idx == 0 else "train"    # make the first shard the validation set
    filename = os.path.join(DATA_CACHE_DIR, f"fineweb-edu_{split}_{shard_idx:06d}")     # create filename
    np.save(filename, tokens)


def print_stats(dataset: datasets.Dataset) -> tuple:
    """
    1. Calculates the total number of tokens in the entire dataset, including the end-of-text (EOT) token for each document.
    2. Calculates the number of shards required to store the dataset, based on the specified `SHARD_SIZE` global variable.
    """
    print(f"\ntotal no. of rows in dataset: {len(dataset):,}")          # no. of rows in the dataset (also see HuggingFace dataset page)
    # total_tokens = sum(row["token_count"] + 1 for row in dataset)       # +1 for the EOT token
    # summing each document takes too long; for sample-10Bt total tokens is: 9,982,590,278
    total_tokens = 9_982_590_278
    total_shards = int(np.ceil(total_tokens / SHARD_SIZE))
    print(f"total no. of tokens to process: {total_tokens:,} (~{int(total_tokens / len(dataset)):,} tok/doc)")
    print(f"no. of shards: {total_shards:,} (with SHARD_SIZE={int(SHARD_SIZE * 1e-6):,}M)")
    return total_tokens, total_shards

# ------ MAIN PROCESS LOOP FUNCTION ------ #

def main() -> None:
    """
    Process `FineWeb-Edu Sample-10BT` by tokenzing documents and saving them in fixed-size shards.
    Each shard is saved as a NumPy array with `100M` tokens, of data type `uint16`.
    Uses parallel processing from the `multiprocessing` library to improve tokenisation speed.
    """

    fw = datasets.load_dataset(DATASET_NAME, name=REMOTE_NAME, split="train")
    total_tokens, total_shards = print_stats(fw)    # get and print key dataset stats

    # --- MAIN PROCESS --- #

    n_proc = max(1, os.cpu_count() // 2)    # no. of worker processes - use half available CPU cores
    print(f"\nusing {n_proc} CPU cores\n")
    # create a process pool for parallel processing:
    # distribute tasks across multiple CPU cores to speed up workload

    with mp.Pool(processes=n_proc) as pool:               

        shard_idx = 0           # track current shard number   
        arr = np.empty(SHARD_SIZE, dtype=np.uint16)   # pre-allocated array (buffer) to store current shard tokens
        shard_tokens = 0        # track stored tokens in current shard
        tok_proc = 0            # total no. of ALL tokens processed

        pbar = tqdm(
            iterable=pool.imap(func=tokenize, iterable=fw, chunksize=64),   # iterate through each document, tokenizing each one in parallel
            bar_format="{desc}[{bar:10}] {percentage:.1f}% complete | [{elapsed}/{remaining}] ({rate_fmt})",
            desc=f"processing shard {shard_idx + 1}/{total_shards} | ",
            total=len(fw),      # main iterator: no. of documents (rows) in the dataset
            unit=" docs",       # units for the rate
            ascii="->=",        # custom ASCII progress bar
            mininterval=1       # minimum update interval for the progress bar (in seconds)
        )
        for tokens in pbar:
            
            if shard_tokens + len(tokens) < SHARD_SIZE:      # if tokens fit in current shard

                arr[shard_tokens: shard_tokens + len(tokens)] = tokens     # store all fittable tokens in array
                shard_tokens += len(tokens)      # tokens processed in current shard
                tok_proc += len(tokens)         # all-time processed tokens (never reset)

            else:
                # add tokens to any remaining leftover space
                remaining = SHARD_SIZE - shard_tokens                                        # calculate how many more tokens can fit
                arr[shard_tokens: shard_tokens + remaining] = tokens[:remaining]       # fill remaining space
                write_datafile(arr, shard_idx)                                       # write full shard to disk
                shard_idx += 1                                                              # move to next shard
                pbar.set_description_str(f"processing shard {shard_idx + 1}/{total_shards} | ")
                # start by populating the next shard with the leftovers of the current document (carry tokens over) 
                arr[: len(tokens) - remaining] = tokens[remaining:]
                shard_tokens = len(tokens) - remaining                           # reset token count to start with leftovers

        if shard_tokens != 0:
            write_datafile(arr, shard_idx)   # write any further remaining tokens as the last shard


if __name__ == "__main__":
    main()