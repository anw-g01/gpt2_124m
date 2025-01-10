import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import tiktoken
import requests
import json
from tqdm import tqdm
import os


DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'hellaswag_dataset')
DATASETS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}
ENC = tiktoken.get_encoding("gpt2")     # GPT2 tokenizer for encoding


def get_file(url: str, filename: str, chunk_size=1024):
    """Download a file from a given `url` and save it to a specified `filename`."""
    with requests.get(url, stream=True) as response:
        total = int(response.headers.get("content-length", 0))
        with open(filename, "wb") as file:                      # "write binary"
            pbar = tqdm(
                iterable=response.iter_content(chunk_size),     # iterate over response chunks
                total=total,                                    # total no. of bytes to download
                unit="iB",                                      # unit of progress ("iB" for binary bytes)
                unit_scale=True,                                # scale unit automatically
                unit_divisor=1024,                              # divisor for scaling the unit (1024 for binary scale)
                desc=f"downloading {filename}"
            )
            for chunk in pbar:
                size = file.write(chunk)                        # write file; returns no. of bytes written to file
                pbar.update(size)                               # update bar by bytes progress


def download(split: str):
    """Download the HellaSwag dataset for a given `split` and save it  to `DATA_CACHE_DIR`."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)                              # create cache directory if it doesn't exist already
    url = DATASETS[split]                                                   # get URL for the specified split
    filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")     # create filename for the split
    if not os.path.exists(filename):                                        # if file doesn't exist
        print(f"downloading {url} to {filename}...")
        get_file(url, filename)                                             # download the file into the directory path


def render_example(example: dict) -> tuple:
    """
    Render a given `example` dictionary into a suitable format for evaluation.
    
    Returns:
    - `data`: `dict()` with keys `label`, `context_tokens`, and `ending_tokens`
    - `tokens`: `torch.tensor` of shape `(4, N)` with concatenated context and ending tokens for each of the four candidates 
    -  `mask`: `torch.tensor` of shape `(4, N)` populated with `1`'s for ending tokens and `0`'s for context tokens
    - `label`: `int` (`0`, `1`, `2`, or `3`) representing the index of the correct ending candidate
    """

    context_tokens = ENC.encode(example["ctx"])     # tokenize context (event description)
    data = {                                
        "label": example["label"],                  # correct ending label
        "context_tokens": context_tokens,   
        "ending_tokens": []                         # list to tokenize endings for each of the 4 candidates
    }
    
    # populate [tokens + endings] and a [mask] for each candidate:
    token_rows, mask_rows = [], []                                      # arrays to populate each token and mask rows
    for ending in example["endings"]:                                   # for each ending candidate (four in total)
        ending_tokens = ENC.encode(" " + ending)                        # encode each ending - prepend space (" ") due to GPT-2 tokenizer
        t_row = context_tokens + ending_tokens                          # concatenate context and ending tokens
        token_rows.append(context_tokens + ending_tokens)               # populate as a given row
        m_row = [0] * len(context_tokens) + [1] * len(ending_tokens)    # create mask for ending tokens
        mask_rows.append(m_row)                                         # populate as a given row
        data["ending_tokens"].append(ending_tokens)                     # append ending tokens to data dictionary

    # create PyTorch tensors for tokens and mask:
    max_len = max(len(row) for row in token_rows)                       # find length of the longest row (from four options)
    tokens = torch.zeros(size=(4, max_len), dtype=torch.long)           # create tensor for tokens
    mask = torch.zeros(size=tokens.shape, dtype=torch.long)             # create tensor for masks (same shape)
    for i, (t_row, m_row) in enumerate(zip(token_rows, mask_rows)):     # populate tokens and masks
        tokens[i, :len(t_row)] = torch.tensor(t_row)                    # populate token tensor
        mask[i, :len(m_row)] = torch.tensor(m_row)                      # populate mask tensor

    return data, tokens, mask, data["label"]

if __name__ == "__main__":
    pass