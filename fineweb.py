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

# ------ GLOBAL VARIABLES ------ #

LOCAL_DIR = "fineweb-edu-10BT"                  # directory to store data shards
DATASET_NAME = "HuggingFaceFW/fineweb-edu"      # HuggingFace dataset name
REMOTE_NAME = "sample-10BT"                     # specific subset
SHARD_SIZE = int(1e8)                           # 100M tokens/shard (100 total shards)

# construct a full path to the local data cache directory
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), LOCAL_DIR)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)      # create the directory if it doesn't exist
ENCODER = tiktoken.get_encoding("gpt2")         # GPT-2 tokenizer 

# ------ HELPER FUNCTIONS ------ #

def tokenize(row: dict) -> np.array:
    """
    Tokenizes the text (key) from a single document (row from dataset) and returns a NumPy array of `uint16` tokens.
    """
    eot = ENCODER._special_tokens["<|endoftext|>"]
    tokens = [eot]     # start token array with end-of-text (EOT) token
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
    print(f"total no. of tokens to process: {total_tokens:,} (~{total_tokens / len(dataset):,} tok/doc)")
    print(f"no. of shards: {total_shards:,} (with SHARD_SIZE={SHARD_SIZE * 1e-6:,}M)")
    return total_tokens, total_shards

# ------ MAIN PROCESS LOOP FUNCTION ------ #

def main() -> None:
    """
    Process `FineWeb-Edu Sample-10BT` by tokenzing documents and saving them in fixed-size shards.
    Each shard is saved as a NumPy array with `100M` tokens, of data type `uint16`.
    Uses parallel processing from the `multiprocessing` library to improve tokenisation speed.
    """

    # --- DOWNLOAD AND LOAD THE DATASET --- #
    try:
        fw = datasets.load_dataset(DATASET_NAME, name=REMOTE_NAME, split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)
    total_tokens, total_shards = print_stats(fw)    # get and print key dataset stats

    # --- MAIN PROCESS --- #
    t0 = time.time()    # capture starting time (for logging)
    n_proc = max(1, os.cpu_count() // 2)    # no. of worker processes - use half available CPU cores
    print(f"\nusing {n_proc} CPU cores\n")
    # create a process pool for parallel processing:
    # distribute tasks across multiple CPU cores to speed up workload
    with mp.Pool(processes=n_proc) as pool:               

        shard_idx = 0       # track current shard number   
        all_tokens = np.empty(SHARD_SIZE, dtype=np.uint16)   # pre-allocated array (buffer) to store current shard tokens
        token_count = 0     # track stored tokens in current shard
        tok_proc = 0        # total no. of ALL tokens processed

        # iterate through each document, tokenizing each one in parallel:
        for i, tokens in enumerate(pool.imap(func=tokenize, iterable=fw, chunksize=16)):      # input data is split into groups of chunksize
            
            if token_count + len(tokens) < SHARD_SIZE:      # if tokens fit in current shard

                all_tokens[token_count: token_count + len(tokens)] = tokens     # store all fittable tokens in array
                token_count += len(tokens)      # tokens processed in current shard
                tok_proc += len(tokens)         # all-time processed tokens (never reset)

                # --- PROGRESS LOGGING --- #
                if i % 50_000 == 0:                                 # log progress every <interval_value> documents                        
                    t1 = time.time()                                # current time
                    dt = t1 - t0                                    # total elapsed time
                    m, s = dt // 60, dt % 60                        # minutes and seconds of elapsed time (same as //60 and %60)
                    tps = (tok_proc / dt)                           # average tok/sec processed 
                    rem = total_tokens - tok_proc                   # all-time remaining tokens
                    t_rem = (rem / tps) if tps > 0 else 0           # estimated remaining time to full completion
                    m_rem, s_rem = t_rem // 60, t_rem % 60          # minutes and seconds of remaining time
                    shard_pct = (token_count / SHARD_SIZE) * 100    # percentage completion of current shard
                    full_pct = (tok_proc / total_tokens) * 100      # total overall completion
                    progress_str = (
                        f"\rprocessing shard {shard_idx + 1}/{total_shards}: "                      # current shard number
                        f"{token_count * 1e-6:.2f}M/{SHARD_SIZE * 1e-6:.0f}M tokens ({shard_pct:.0f}%) | "     # processed tokens for given shard (in millions)
                        f"total: {tok_proc * 1e-9:.2f}B/{total_tokens * 1e-9:.2f}B tok | "          # all-time processed tokens (in billions)
                        f"{tps * 1e-6:.1f}M tok/sec | "             # tokens per seconds processed
                        f"{full_pct:.1f}% complete | "              # percentage completion (full process)
                        f"{int(m):02d}:{int(s):02d}/{int(m_rem):02d}:{int(s_rem):02d}"      # elapsed time / estimated remaining time
                    )
                    print(progress_str, end="")
            else:
                # add tokens to any remaining leftover space
                remaining = SHARD_SIZE - token_count                                        # calculate how many more tokens can fit
                all_tokens[token_count: token_count + remaining] = tokens[:remaining]       # fill remaining space
                write_datafile(all_tokens, shard_idx)                                       # write full shard to disk
                shard_idx += 1                                                              # move to next shard
                # start by populating the next shard with the leftovers of the current document
                all_tokens[: len(tokens) - remaining] = tokens[remaining:]
                token_count = len(tokens) - remaining                           # reset token count to start with leftovers

        if token_count != 0:
            write_datafile(all_tokens, shard_idx)   # write any further remaining tokens as the last shard


if __name__ == "__main__":
    main()