import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import tiktoken
import requests
import json
from tqdm import tqdm
from tqdm_bars import tqdmHS
import os


DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'hellaswag_dataset')
DATASETS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}
ENC = tiktoken.get_encoding("gpt2")     # GPT2 tokenizer for encoding


def _get_file(url: str, filename: str, chunk_size=1024):
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


def _num_examples(filename: str, split: str):
    """Find the number of examples in the HellaSwag dataset for a given `split`."""
    with open(filename, "r") as file:
        length = len(file.readlines())
        DATASETS[split].append(length)      # add to the GLOBAL dictionary


def _download(split: str):
    """Download the HellaSwag dataset for a given `split` and save it  to `DATA_CACHE_DIR`."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)                              # create cache directory if it doesn't exist already
    url = DATASETS[split]                                                   # get URL for the specified split
    filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")     # create filename for the split
    if not os.path.exists(filename):                                        # if file doesn't exist
        print(f"downloading {url} to {filename}...")
        _get_file(url, filename)                                             # download the file into the directory path
        _num_examples(filename, split)                                       # find the number of examples in the file

def iterate_examples(split: str):
    """Iterate over examples in the HellaSwag dataset for a given `split`."""
    _download(split)                                        
    filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")     
    with open(filename, "r") as file:                       
        for line in file:               # each line is an example
            yield json.loads(line)      # yield the example as a JSON object (dictionary)


def render(example: dict) -> tuple:
    """
    Render a given `example` dictionary into a suitable format for evaluation.
    
    Returns:
    - `tokens`: `torch.tensor` of shape `(4, N)` with concatenated context and ending tokens for each of the four candidates 
    -  `mask`: `torch.tensor` of shape `(4, N)` populated with `1`'s for ending tokens and `0`'s for context tokens
    - `label`: `int` (`0`, `1`, `2`, or `3`) representing the index of the correct ending candidate
    """
    label = example["label"],                       # correct ending label (0, 1, 2, or 3)
    context_tokens = ENC.encode(example["ctx"])     # tokenize context (event description)
    
    # populate [tokens + endings] and a [mask] for each candidate:
    token_rows, mask_rows = [], []                                      # arrays to populate each token and mask rows
    for ending in example["endings"]:                                   # for each ending candidate (four in total)
        ending_tokens = ENC.encode(" " + ending)                        # encode each ending - prepend space (" ") due to GPT-2 tokenizer
        t_row = context_tokens + ending_tokens                          # concatenate context and ending tokens
        token_rows.append(context_tokens + ending_tokens)               # populate as a given row
        m_row = [0] * len(context_tokens) + [1] * len(ending_tokens)    # create mask for ending tokens
        mask_rows.append(m_row)                                         # populate as a given row

    # create PyTorch tensors for tokens and mask:
    max_len = max(len(row) for row in token_rows)                       # find length of the longest row (from four options)
    tokens = torch.zeros(size=(4, max_len), dtype=torch.long)           # create tensor for tokens
    mask = torch.zeros(size=tokens.shape, dtype=torch.long)             # create tensor for masks (same shape)
    for i, (t_row, m_row) in enumerate(zip(token_rows, mask_rows)):     # populate tokens and masks
        tokens[i, :len(t_row)] = torch.tensor(t_row)                    # populate token tensor
        mask[i, :len(m_row)] = torch.tensor(m_row)                      # populate mask tensor

    return tokens, mask, label                                       # return tokens, mask, and ground truth label


@torch.no_grad()
def evaluate(
        model_type: str = "gpt2",
        split: str = "train",
        compile: bool = False,
        verbose: bool = False,
        device: str = "cuda"
    ) -> tuple:
    """
    Evaluate a given `model_type` on the HellaSwag dataset for a given `split`.
    Loads examples iteratively and calculates the average cross entropy loss of
    the model's predictions for each set of ending candidates. The candidate with
    the lowest loss is chosen as the predicted ending. A tally of correct predictions
    against ground truth labels is kept and returned at the end of evaluation.
    """

    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type).to(device)
    model = torch.compile(model) if compile else model

    total, correct = 0, 0
    pbar = tqdmHS(
        iterable=iterate_examples(split),
        total=DATASETS[split][1],               # length of examples (from GLOBAL dictionary)
        desc=f"correct: 0/0",
        disable=not verbose
    )

    for i, example in enumerate(pbar):
        tokens, mask, label = render(example)
        T, M = tokens.to(device), mask.to(device)

        P = model(T).logits                     # get all logits from forward pass

        P = P[:, :-1, :].contiguous()           # remove last prediction (nothing to predict after ending)
        T = T[:, 1:].contiguous()               # remove first token (no previous token to predict it)

        L = F.cross_entropy(
            input=P.view(-1, P.shape[-1]),      # --> [N, vocab_size] (where N = batch_size * seq_len)
            target=T.view(-1),                  # --> [N,]
            reduction="none"                    # retain individual losses
        )                                       # --> [N,]
        L = L.view(T.shape[0], -1)              # reshape back to [batch_size * seq_len]

        M = M[:, 1:].contiguous()               # shift mask same as tokens tensor (T)
        L *= M                                  # apply mask element-wise (zero all context token positions)

        avg_L = L.sum(dim=1) / M.sum(dim=1)     # average along each sample candidate
        y_pred = avg_L.argmin().item()          # lower loss => more confident

        # accumulate stats
        total += 1
        correct += int(y_pred == label)

        if verbose and (i % 100 == 0):          # progress logging if verbose=True
            correct_pct = correct / (i + 1) * 100
            progress_str = (
                f"correct: {correct:,}/{i:,} ({correct_pct:.1f}%)"
            )
            pbar.set_description_str(progress_str)
        
    return correct, total


def main() -> None:
    """
    Evaluate on pretrained `GPT-2` (124M) on the HellaSwag training dataset:
    
    `[===============] 39,905/39,905 (100.0%) | correct: 12,356/39,900 (31.0%) [06:10<00:00, ? examples/s]`

    Final results score: `correct: 12,358/39,905 (31.0%)`

    Use `model_type="gpt2-xl"` for the `1.5B` parameter `GPT-2` model; default is `"gpt2"`.
    """

    correct, total = evaluate(verbose=True)     # evaluate GPT-2 (124M) on HellaSwag
    
    # display results:
    score = correct / total * 100
    print(f"total correct: {correct:,}/{total:,} ({score:.1f}%)")

    

if __name__ == "__main__":
    
    main()      # evaluate with a pretrained model (with progress bar logging)