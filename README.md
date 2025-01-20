# Training GPT-2 (124M)

A PyTorch implementation of OpenAI's [GPT-2](https://github.com/openai/gpt-2) model with 124M parameters. 

Installed dependencies: [`PyTorch`](https://pytorch.org/), [`NumPy`](https://numpy.org/), [`datasets`](https://huggingface.co/docs/datasets/en/index), [`tiktoken`](https://github.com/openai/tiktoken), [`tqdm`](https://github.com/tqdm/tqdm), [`transformers`](https://huggingface.co/docs/transformers/en/index).

## Work in progress...
...

## Project Files

- `config.py`: Contains all customisable global hyperparameters, directory paths and settings for model training and optimisation.
- `model.py`: Defines the GPT-2 (124M parameter) model architecture using PyTorch's [`torch.nn`](https://pytorch.org/docs/stable/nn.html#module-torch.nn) modules.
- `data/`:
  - `load_fineweb.py`: Downloads the [`sample-10BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer/sample-10BT) subset of [`FineWeb-Edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) as shard files in a specified directory. 
  - `fineweb.py`: Defines a custom PyTorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) class to handle shard loading and indexing to return batched FineWeb-Edu tokens. 
  - `hellaswag.py`:  Defines a custom PyTorch `Dataset` class for rendering and loading examples from the [HellaSwag](https://github.com/rowanz/hellaswag/tree/master/data) dataset. Includes an evaluation function that computes accuracy scores for a specified model. 
  - `shakespeare.py`:  Defines a custom PyTorch `Dataset` class for loading, tokenizing and batching data from an option of two available Shakespeare text file datasets.
- `train.py`: Implements the core model training and optimisation loop with loaded datasets. Further supports distributed training across multiple GPUs in parallel using PyTorch's [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP).
- `main.py`: Initiates the GPT-2 model training and optimisation process.
- `tqdm_bars.py`: Defines custom [`tqdm`](https://github.com/tqdm/tqdm) progress bar classes, with tailored metrics for monitoring during model training in `train.py`, dataset shard downloading in `data/load_fineweb.py`, and optional model evaluation runs on the HellaSwag dataset.
