"""
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Process the FineWeb-Edu Sample-10BT dataset by tokenzing documents and saving them in fixed-size shards.
Each shard is saved as a NumPy array with 100M tokens (available space), of data type `uint16`.
"""

import numpy as np
import tiktoken
import os
import multiprocessing as mp
import datasets
from tqdm_bars import tqdmFW

# ------ GLOBAL VARIABLES ------ #

LOCAL_DIR       = "fineweb-edu-10BT"                # directory to store data shards
DATASET_NAME    = "HuggingFaceFW/fineweb-edu"       # HuggingFace dataset name
REMOTE_NAME     = "sample-10BT"                     # specific subset
SHARD_SIZE      = int(1e8)                          # 100M tokens/shard except last shard (100 total shards)
TOTAL_TOKENS    = int(9.9e9)                             # 9.9B total tokens used from sample-10BT subset
DATA_CACHE_DIR  = os.path.join(os.path.dirname(__file__), LOCAL_DIR)    # construct a full path to the local data cache directory
ENCODER         = tiktoken.get_encoding("gpt2")                         # initialise GPT-2 tokenizer 
EOT             = ENCODER._special_tokens["<|endoftext|>"]              # end-of-text (EOT) token 

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


def calc_shard_num(dataset: datasets.Dataset) -> tuple:
    """Calculates the number of shards required to store the dataset."""
    print(f"\ntotal no. of rows in dataset: {len(dataset):,}")          # no. of rows in the dataset (also see HuggingFace dataset page)
    total_shards = int(np.ceil(TOTAL_TOKENS / SHARD_SIZE))
    print(f"total no. of tokens to process: {TOTAL_TOKENS:,} (~{int(TOTAL_TOKENS / len(dataset)):,} tok/doc)")
    print(f"no. of shards: {total_shards:,} (with SHARD_SIZE={int(SHARD_SIZE * 1e-6):,}M)")
    return total_shards


def main() -> None:
    """
    Process `FineWeb-Edu Sample-10BT` by tokenzing documents and saving them in fixed-size shards.
    Each shard is saved as a NumPy array with `100M` tokens, of data type `uint16`.
    Uses parallel processing from the `multiprocessing` library to improve tokenisation speed.
    """

    # --- SETUP --- #

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)          # create directory to store shards if it doesn't exist
    fw = datasets.load_dataset(DATASET_NAME, name=REMOTE_NAME, split="train")
    total_shards = calc_shard_num(fw)    # calculate the no. of shards that will be created

    # --- MAIN PROCESS --- #

    n_proc = max(1, os.cpu_count() // 2)    # no. of worker processes - use half available CPU cores
    print(f"\nusing {n_proc} CPU cores for tokenization...\n")
    # create a process pool for parallel processing:
    # distribute tasks across multiple CPU cores to speed up workload
    with mp.Pool(processes=n_proc) as pool:               

        shard_idx = 0           # track current shard number   
        arr = np.zeros(SHARD_SIZE, dtype=np.uint16)   # pre-allocated array (buffer) to store current shard tokens
        shard_tokens = 0        # track stored tokens in current shard
        tok_proc = 0            # total no. of ALL tokens processed

        pbar = tqdmFW(
            iterable=pool.imap(func=tokenize, iterable=fw, chunksize=32),   # iterate through each document, tokenizing each one in parallel
            desc=f"processing shard {shard_idx + 1}/{total_shards} | ",
            total=TOTAL_TOKENS,                                             # visualise progress of total tokens processed
        )
        for tokens in pbar:
            
            if shard_tokens + len(tokens) < SHARD_SIZE:                     # if tokens fit in current shard
                arr[shard_tokens: shard_tokens + len(tokens)] = tokens      # store all fittable tokens in array
                shard_tokens += len(tokens)                                 # tokens processed in current shard
                tok_proc += len(tokens)                                     # all-time processed tokens (never reset)
                pbar.update(len(tokens))                                    # update progress bar with the number of tokens processed
            else:
                # add tokens to any remaining leftover space
                remaining = SHARD_SIZE - shard_tokens                                   # calculate how many more tokens can fit
                arr[shard_tokens: shard_tokens + remaining] = tokens[:remaining]        # fill remaining space
                write_datafile(arr, shard_idx)                                          # write full shard to disk
                shard_idx += 1                                                          # move to next shard
                pbar.set_description_str(f"processing shard {shard_idx + 1}/{total_shards} | ")
                # start by populating the next shard with the leftovers of the current document (carry tokens over) 
                arr[: len(tokens) - remaining] = tokens[remaining:]
                shard_tokens = len(tokens) - remaining      # reset token count to start with leftovers

        # write any further remaining tokens as the last shard
        if shard_tokens != 0:
            write_datafile(arr[:shard_tokens], shard_idx)      # last shard holds 82,590,278 tokens (NOT 100M)

if __name__ == "__main__":
    main()