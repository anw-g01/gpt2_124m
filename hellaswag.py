import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import tiktoken
import requests
import json
from tqdm_bars import tqdmHS
import os
from model import GPT2_124M, GPT2Config
from typing import Optional, Tuple


DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'hellaswag_dataset')
DATASETS = {
    "train": ["https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl", 39_905],
    "val": ["https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl", 10_042],
    "test": ["https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl", 10_003],
}
ENC = tiktoken.get_encoding("gpt2")     # GPT2 tokenizer for encoding


def _get_file(url: str, filename: str, chunk_size=1024):
    """Download a file from a given `url` and save it to a specified `filename`."""
    with requests.get(url, stream=True) as response:
        total = int(response.headers.get("content-length", 0))
        with open(filename, "wb") as file:                      # "write binary"
            for chunk in response.iter_content(chunk_size):     # iterate over chunks of data
                file.write(chunk)                               # write file (optional: can return the no. of bytes written to file)


def _download(split: str):
    """Download the HellaSwag dataset for a given `split` and save it  to `DATA_CACHE_DIR`."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)                              # create cache directory if it doesn't exist already
    url = DATASETS[split][0]                                                # get URL for the specified split
    filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")     # create filename for the split
    if not os.path.exists(filename):                                        # if file doesn't exist
        print(f"\ndownloading to {filename}...\n")
        _get_file(url, filename)                                             # download the file into the directory path


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
    label = example["label"]                        # correct ending label (0, 1, 2, or 3)
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

    return tokens, mask, label


@torch.no_grad()
def evaluate(
        model: Optional[GPT2_124M] = None,   # uses HuggingFace GPT2LMHeadModel() if None
        ddp_world_size: int = 1,
        ddp_local_rank: int = 0,
        model_type: str = "gpt2",
        split: str = "val",
        compile: bool = False,
        verbose: bool = False,
        device: str = "cuda"
    ) -> Tuple[int, int]:
    """
    Evaluate a specified `GPT2_124M` model on the HellaSwag dataset for a given `split`.
    If no `model` is specified (`model=None`), the function will use a pretrained `GPT-2` model
    from HuggingFace. Choose the model size with `model_type`; default is set to `"gpt2"` which
    is the 124M parameter size of GPT-2.

    The function loads examples iteratively and calculates the average cross entropy loss of
    the model's predictions for each set of ending candidates. The candidate with
    the lowest loss is chosen as the predicted ending. A tally of correct predictions
    against ground truth labels is kept and returned at the end of evaluation.
    """
    torch.set_float32_matmul_precision('high') # use tf32

    using_model = False     # flag for getting logits
    if model is None:       # use HuggingFace model if None
        model = GPT2LMHeadModel.from_pretrained(model_type).to(device)
        print(f"using HuggingFace model: {model_type}...\n")
    else:
        using_model = True
    model = torch.compile(model) if compile else model
    model.eval()    # set model to evaluation mode

    pbar = tqdmHS(
        iterable=iterate_examples(split),
        total=DATASETS[split][1],               # length of examples (39,905 for "train")
        desc=f"correct: 0/0",
        disable=(not verbose and ddp_world_size != 1)   # progress bar only for single processes
    )

    total, correct = 0, 0
    for i, example in enumerate(pbar):

        # a GPU rank will process every ddp_world_size'th example, where
        # each example is processed by exactly one GPU with no overlaps:
        if i % ddp_world_size != ddp_local_rank:     
            continue

        tokens, mask, label = render(example)
        T, M = tokens.to(device), mask.to(device)

        # get all logits from forward pass
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                if using_model:
                    logits, _ = model(T)                # get logits from GPT_124M() (see from model.py)
                else:
                    logits = model(T).logits             # logits from HuggingFace model
        
        P = logits[:, :-1, :].contiguous()      # remove last prediction (nothing to predict after ending)
        T = T[:, 1:].contiguous()               # remove first token (no previous token to predict it)
        M = M[:, 1:].contiguous()               # shift mask same as tokens tensor (T)

        L = F.cross_entropy(
            input=P.view(-1, P.shape[-1]),      # --> [N, vocab_size] (where N = batch_size * seq_len)
            target=T.view(-1),                  # --> [N,]
            reduction="none"                    # retain individual losses
        )                                       # --> [N,]
        L = L.view(T.shape[0], -1)              # reshape back to [batch_size * seq_len]

        L *= M                                  # apply mask element-wise (zero all context token positions)

        avg_L = L.sum(dim=1) / M.sum(dim=1)     # average along each sample candidate
        y_pred = avg_L.argmin().item()          # lower loss => more confident

        # accumulate stats
        total += 1
        correct += int(y_pred == label)

        if verbose and (i % 5 == 0 or i == DATASETS[split][1] - 1):            # progress logging if verbose=True
            correct_pct = correct / (i + 1) * 100
            progress_str = (
                f"correct: {correct:,}/{i + 1:,} ({correct_pct:.1f}%)"
            )
            pbar.set_description_str(progress_str)
        
    return correct, total


def eval1() -> None:
    """
    Pre-trained HuggingFace Model Example:
    ---
    Evaluate using a pretrained `GPT-2` (124M) model when `model=None` on the HellaSwag `"val"` dataset:
    N.B. use `model_type="gpt2-xl"` for the `1.5B` parameter `GPT-2` model; default is `"gpt2"`.
    
    `[===============] 10,042/10,042 (100.0%) | correct: 2,968/10,042 (29.6%) [01:34<00:00, ? examples/s]`

    Example final score output: `correct: 2,968/10,042 (29.6%)`
    N.B. Progress bar is disabled for multi-GPU evaluation due to shared examples.
    """

    correct, total = evaluate(verbose=True)     # evaluate GPT-2 (124M) on HellaSwag
    
    # display results:
    score = correct / total * 100
    print(f"total correct: {correct:,}/{total:,} ({score:.1f}%)")


def eval2() -> None:
    """
    Specified `GPT2_124M` model evaluation example:
    ---
    This example uses an un-trained newly instantiated `GPT2_124M()` model.
    Its accuracy should be ~25% (1/4) as a a result of randomly initialised
    weights, i.e. model "guessing". 
    
    Example final score output: 2,555/10,042 (25.4%)
    N.B. Progress bar is disabled for multi-GPU evaluation due to shared examples.
    """
    
    correct, total = evaluate(
        model=GPT2_124M(GPT2Config()),     
        verbose=True,
    )
    
    # display results:
    score = correct / total * 100
    print(f"total correct: {correct:,}/{total:,} ({score:.1f}%)")


if __name__ == "__main__":
    eval2()      