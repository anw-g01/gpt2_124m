import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import GPT2LMHeadModel
import tiktoken
import requests
import json
from tqdm_bars import tqdmHS
import os
from model import GPT2_124M, GPT2Config
from typing import Optional, Tuple


class HellaSwag(Dataset):
    """
    A PyTorch `torch.utils.data.Dataset` class for loading and processing the HellaSwag dataset.

    Args:
    --
        `split` (`str`): the dataset split to load. Options are `"train"`, `"val"`, and `"test"`. Default is `"val"`.
    
    Attributes:
    --
        `links` (`dict`): URLs for downloading the dataset splits.
        `split` (`str`): the dataset split to load.
        `dir` (`str`): cache directory to store the downloaded dataset files.
        `enc` (`tiktoken.Encoding`): a `GPT-2` tokenizer for encoding text.
        `examples` (`list`): list of examples loaded from the dataset file.
    
    Methods:
    --
        `__len__()`: returns the number of examples in the dataset.
        `__getitem__()`: returns the tokenized context and ending sequences, mask, and label for a given index.
        `_download()`: downloads the dataset file from the specified URL and saves it to a local directory.
        `_load()`: loads the dataset examples into a list from the downloaded file.
    """

    def __init__(self, split: str = "val"):
        self.links = {
            "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
            "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
            "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
        }
        self.split = split
        self.dir = os.path.join(os.path.dirname(__file__), 'hellaswag_dataset')
        self.enc = tiktoken.get_encoding("gpt2")    
        # download and load the dataset split:
        self.examples = self._load()

    def __len__(self):
        return len(self.examples) 

    def __getitem__(self, idx: int):

        example = self.examples[idx]
        label = example["label"]
        context_tokens = self.enc.encode(example["ctx"])

        token_rows, mask_rows = [], []
        # iterate over the 4 possible ending sequences:
        for ending in example["endings"]:
            ending_tokens = self.enc.encode(" " + ending)            # tokenize the ending tokens
            tok_row = context_tokens + ending_tokens    # concatenate with context tokens
            # create a mask row to 
            mask_row = [0] * len(context_tokens) + [1] * len(ending_tokens)
            # append to the list of token and mask rows:
            token_rows.append(tok_row)
            mask_rows.append(mask_row)
        
        # pre-allocate padded PyTorch tensors:
        max_len = max(len(row) for row in token_rows)
        tokens = torch.zeros(size=(4, max_len), dtype=torch.long)
        mask = torch.zeros(size=(4, max_len), dtype=torch.long)
        # fill the tensors with the token and mask rows:
        for i, (tok_row, mask_row) in enumerate(zip(token_rows, mask_rows)):
            tokens[i, :len(tok_row)] = torch.tensor(tok_row)
            mask[i, :len(mask_row)] = torch.tensor(mask_row)

        return tokens, mask, label

    def _download(self, chunk_size=1024) -> None:
        """
        Helper function: downloads a dataset file from a 
        specified URL and saves it to a local directory.
        """
        os.makedirs(self.dir, exist_ok=True)                # create cache directory to store the datasets
        url = self.links[self.split]                        # get URL for the specified split
        file_name = f"hellaswag_{self.split}.jsonl"     
        file_path = os.path.join(self.dir, file_name)       # path to save the file
        if not os.path.exists(file_path):                   # download if file doesn't exist
            print(f"downloading {file_name}...")
            with requests.get(url, stream=True) as response:
                with open(file_path, "wb") as file:                     # "write binary"
                    for chunk in response.iter_content(chunk_size):     # iterate over chunks of data
                        file.write(chunk)                               # write file (optional: can return the no. of bytes written to file)
        return file_path
    
    def _load(self) -> list:
        file_path = self._download()
        with open(file_path, "r") as file:
            examples = [json.loads(line) for line in file]
        return examples        


def evaluate(
        data_loader: DataLoader,
        model: Optional[GPT2_124M] = None,  
        device: str = "cuda",
        verbose: bool = False,
        model_type: str = "gpt2",
        compile: bool = False
    ) -> Tuple[int, int]:
    """
    Evaluate a specified `GPT2_124M` model on a HellaSwag dataset. If no `model` is specified, 
    the function will use a pretrained `GPT-2` model from HuggingFace.

    The function iterates through examples in a PyTorch `DataLoader` which must contain a 
    `HellaSwag` `Dataset` object. The `DataLoader` MUST be set with `batch_size=None` as 
    each example is inherently a batch of `4` candidate ending tokens.

    For each example, the function renders the context and 
    ending candidates and calculates the average cross entropy loss of the model's predictions 
    for each set of ending candidates. The candidate with the lowest loss is chosen as the 
    predicted ending. A tally of correct predictions against ground truth labels is kept and 
    returned at the end of evaluation.

    Args:
    --
        `data_loader` (`DataLoader`): a PyTorch data loader containing a specific split of the HellaSwag dataset.
        `model` (`Optional[GPT2_124M]`): the `GPT2_124M` model to evaluate. If `None`, a pretrained `GPT-2 model` from HuggingFace will be used.
        `device` (`str`): the device to run the evaluation on. Default is `"cuda"`.
        `verbose` (`bool`): if `True`, progress will be logged. Default is `False`.
        `model_type` (`str`): the type of `GPT-2` model to use if no `model` is specified. 
        e.g. use `"gpt2-xl"` for the `1.5B` parameter `GPT-2` model; default is `"gpt2"` (`124M`).
        `compile` (`bool`): if True, the model will be compiled with `torch.compile`. Default is `False`.
    
    Returns:
    --
    `tuple`: a tuple containing:
        `correct` (`int`): number of correct predictions
        `total` (`int`): number of examples processed.
    """
    torch.set_float32_matmul_precision('high') # use tf32

    using_model = False     # to flag method for getting logits
    if model is None:       # use HuggingFace model if None
        model = GPT2LMHeadModel.from_pretrained(model_type).to(device)
        print(f"using HuggingFace model: {model_type}...\n")
    else:
        using_model = True
    model = torch.compile(model) if compile else model
    model.eval()    # set model to evaluation mode

    pbar = tqdmHS(
        iterable=data_loader,
        desc=f"correct: 0/0",
        disable=(not verbose)   
    )

    total, correct = 0, 0
    for i, (tokens, mask, label) in enumerate(pbar):
        T, M = tokens.to(device), mask.to(device)
        # get all logits from forward pass
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            if using_model:
                logits, _ = model(T)            # get logits from GPT_124M() (see from model.py)
            else:
                logits = model(T).logits        # logits from HuggingFace model
        
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

        # accumulate stats:
        total += 1
        correct += int(y_pred == label)

        # progress logging if verbose=True:
        if verbose and (i % 5 == 0 or i == len(data_loader) - 1):            
            correct_pct = correct / (i + 1) * 100
            progress_str = (
                f"correct: {correct:,}/{i + 1:,} ({correct_pct:.1f}%)"
            )
            pbar.set_description_str(progress_str)
        
    return correct, total


def eval1() -> None:
    """
    Pre-trained HuggingFace Model Example:
    --
    Evaluate using a pretrained `GPT-2` (124M) model when `model=None` on the HellaSwag `"val"` dataset:
    
    `[===============] 10,042/10,042 (100.0%) | correct: 2,968/10,042 (29.6%) [01:34<00:00, ? examples/s]`

    Example final score output: `correct: 2,968/10,042 (29.6%)`
    """
    hs_loader = DataLoader(
        dataset=HellaSwag("val"),
        batch_size=None,    # examples must not be batched
        shuffle=False,      # no reason to shuffle for eval
    )

    # evaluate GPT-2 (124M) on HellaSwag:
    correct, total = evaluate(hs_loader, verbose=True)     
    
    # display results:
    score = correct / total * 100
    print(f"total correct: {correct:,}/{total:,} ({score:.1f}%)")


def eval2() -> None:
    """
    Specified `GPT2_124M` model evaluation example:
    ---
    This example uses an un-trained newly instantiated `GPT2_124M()` model.
    Its accuracy should be `~25%` (`1/4`) as a result of randomly initialised
    weights, i.e. model "guessing". 
    
    Example final score output: `correct: 2,555/10,042 (25.4%)`
    """

    correct, total = evaluate(
        data_loader=DataLoader(
            dataset=HellaSwag("val"),
            batch_size=None,    # examples must not be batched
            shuffle=False,      # no reason to shuffle for eval
        ), 
        model=GPT2_124M(GPT2Config()),
        verbose=True, 
        device="cpu"
    )     
    
    # display results:
    score = correct / total * 100
    print(f"total correct: {correct:,}/{total:,} ({score:.1f}%)")


if __name__ == "__main__":
    eval2()      