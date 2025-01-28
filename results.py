import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams["font.size"] = 9
plt.rcParams["lines.linewidth"] = 1
from config import LOG_DIR
from train import _get_checkpoint_filename
from model import GPT2_124M, GPT2Config


def display_graphs(
        used_gpus: int,
        epoch_num: int,
        iter_num: int,
        truncate: bool = True,
        log_dir: str = LOG_DIR,
        checkpoint_type: str = "end",
        plot_lr: bool = True,
        save: bool = True,
        img_format: str = "png"
    ) -> str:
    """
    Plot training results given a model checkpoint directory.

    Loads training and validation losses, HellaSwag evaluation scores, and learning rates
    from the specified checkpoint directory. The figure consists of two subplots: one for
    training and validation losses, and one for HellaSwag evaluation scores. Optionally, 
    it can also plot the learning rate on the same subplot as the losses. The figure can
    be saved as an image file in the specified `format` type.

    Args:
    --
        `used_gpus` (`int`): number of GPUs used for training.
        `epoch_num` (`int`): epoch number (starting from `1`) for the checkpoint.
        `iter_num` (`int`): iteration number (starting from `1`) for the checkpoint.
        `log_dir` (`str`): directory where the logs and checkpoints are stored.
        `checkpoint_type` (`str`): prefix of the checkpoint file name (must be `"end"` or `"val"`).  Default: `"end"`.
        `plot_lr` (`bool`): whether to plot the learning rate on the same plot as the losses.
        `save` (`bool`): whether to save the plot as an image file.
        `img_format` (`str`): format to save the plot image (e.g., `"png"`, `"jpg"`).   
    """
    assert checkpoint_type in ["end", "val"], "checkpoint_type must be 'end' or 'val'"

    # get the checkpoint directory from the filename (for the specified epoch and iteration):
    filename = _get_checkpoint_filename(checkpoint_type, epoch_num, iter_num, used_gpus)
    checkpoint_dir = os.path.join(log_dir, filename)
    assert os.path.exists(checkpoint_dir), f"checkpoint directory does not exist: {checkpoint_dir}"     # check if the directory exists
    print(f"checkpoint file: {filename}")
    print(f"\nloading training plots...\n")
    # load numpy arrays from the checkpoint directory:
    train_losses = np.load(os.path.join(checkpoint_dir, "train_losses.npy"))
    val_losses = np.load(os.path.join(checkpoint_dir, "val_losses.npy"))
    hellaswag_scores = np.load(os.path.join(checkpoint_dir, "hellaswag_scores.npy"))
    learning_rates = np.load(os.path.join(checkpoint_dir, "learning_rates.npy"))      
      
    # truncate arrays up to iter_idx (default):
    if truncate:
        idx = min(iter_num - 1, len(train_losses))    # in case iter_idx > len(train_losses) 
        train_losses = train_losses[:idx]
        val_losses = val_losses[:idx]
        hellaswag_scores = hellaswag_scores[:idx]
        learning_rates = learning_rates[:idx]

    # ----- MAIN FIGURE ----- #
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(f"GPT2 Training Results: {filename}")  # title for the entire figure
    x = np.arange(1, len(train_losses) + 1)             # x-values for all plots
    
    # --- LEFT SUBPLOT (training and validation losses) --- #
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("iteration (per GPU)")
    axs[0].set_ylabel("loss")
    axs[0].grid(True)
    axs[0].plot(
        x, train_losses,
        label="train_loss", color="tab:olive"
    )
    # for val_loss, only plot non-NaN values due to interval storage:
    val_idx = np.isfinite(val_losses)           # cell turns False if cell was NaN
    axs[0].plot(
        x[val_idx], val_losses[val_idx],        # only select elements where val_idx is True (i.e. non-NaN)  
        label="val_loss", color="tab:purple"
    )
    axs[0].axhline(y=3.292, color="tab:red", linestyle="--", label="OpenAI GPT-2 (124M) baseline")
    axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))    # comma separator for x-axis tickers
    # optional: plot learning rates in a twin axis (with losses figure):
    if plot_lr:
        lr_color = "tab:cyan"    
        ax_lr = axs[0].twinx()                              # create a second y-axis for learning rates
        ax_lr.set_ylabel("learning rate", color=lr_color)
        ax_lr.plot(
            x, learning_rates, label="learning_rate",
            color=lr_color, alpha=0.5                       # set alpha for transparency    
        )
        ax_lr.tick_params(axis="y", labelcolor=lr_color)    # set axis tick colour
        ax_lr.legend()                                      # set legend location
        # combine legends for twin axes:
        lines, labels = axs[0].get_legend_handles_labels()
        lines_lr, labels_lr = ax_lr.get_legend_handles_labels()
        axs[0].legend(lines + lines_lr, labels + labels_lr, loc="upper right")
    else:
        axs[0].legend()                                     # legend for losses
    
    # --- RIGHT SUBPLOT (HellaSwag evaluation scores) --- #
    axs[1].set_title("HellaSwag Evaluation")
    axs[1].set_xlabel("iteration (per GPU)")
    axs[1].set_ylabel("accuracy (%)")
    axs[1].grid(True)
    axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))    # comma separator for x-axis tickers
    hs_idx = np.isfinite(hellaswag_scores)    # bool: False if cell is NaN
    axs[1].plot(x[hs_idx], hellaswag_scores[hs_idx])
    # plot baseline scores from other models:
    axs[1].axhline(y=29.6, color="tab:red", linestyle="--", label="OpenAI GPT-2 (124M) baseline")
    axs[1].axhline(y=48.9, color="tab:blue", linestyle="--", label="OpenAI GPT-2 (1.56B) baseline")
    axs[1].axhline(y=54.7, color="tab:green", linestyle="--", label="OpenAI GPT-3 (1.56B) baseline")
    axs[1].legend()
    plt.tight_layout()  
    if save:
        plt.savefig(
            f"figure_{filename}", 
            dpi=300, bbox_inches="tight",
            format=img_format
        )
    if not save:
        plt.show()      # code will pause until the plot window is closed

    # get directory path to the .pt checkpoint file:
    print(f"\nloading model weights...\n")
    model_checkpoint_dir = os.path.join(checkpoint_dir, "model_checkpoint.pt")      # path to the .pt file holding dictionary of checkpoints file
    checkpoint = torch.load(model_checkpoint_dir)                                   # load the .pt file for dictionary of checkpoints
    # create a new model instance and load the state_dict from the dictionary:
    model = GPT2_124M(GPT2Config(vocab_size=50304))                                 # create new model instance, must be same config as trained model
    model.load_state_dict(checkpoint["model_state_dict"])                           # load the model state_dict from the dictionary
    # generate text samples from the model:
    model.sample("Hello, I'm a")     


def load_model(
        used_gpus: int,
        epoch_num: int,
        iter_num: int,
        log_dir: str = LOG_DIR,
        checkpoint_type: str = "end",
    ) -> GPT2_124M:
    """
    Load a trained model from a specified checkpoint directory.

    Args:
    --
        `used_gpus` (`int`): number of GPUs used for training.
        `epoch_num` (`int`): epoch number (starting from `1`) for the checkpoint.
        `iter_num` (`int`): iteration number (starting from `1`) for the checkpoint.
        `log_dir` (`str`): directory where the logs and checkpoints are stored.
        `checkpoint_type` (`str`): prefix of the checkpoint file name (must be `"end"` or `"val"`). Default: `"end"`.

    Returns:
    --
        `model` (`GPT2_124M`): GPT-2 model instance with loaded weights from a checkpoint.
    """
    # get the checkpoint directory from the filename (for the specified epoch and iteration):
    filename = _get_checkpoint_filename(checkpoint_type, epoch_num, iter_num, used_gpus)
    checkpoint_dir = os.path.join(log_dir, filename)
    assert os.path.exists(checkpoint_dir), f"checkpoint directory does not exist: {checkpoint_dir}"     # check if the directory exists
    print(f"loading checkpoint: {filename}...")
    
    # get full path to the .pt file holding dictionary of checkpoints:
    full_path = os.path.join(checkpoint_dir, "model_checkpoint.pt")  
    checkpoint = torch.load(full_path)      # load the .pt file for dictionary of checkpoints

    # create a new model instance and load the state_dict from the dictionary:
    print(f"loading model weights...\n")
    model = GPT2_124M(GPT2Config(vocab_size=50304))         # create new model instance, must be same config as trained model
    model.load_state_dict(checkpoint["model_state_dict"])   # load the model state_dict from the checkpoint dictionary

    return model    # return the trained model

if __name__ == "__main__":

    # --- DISPLAY TRAINING CURVES --- #

    # select a checkpoint file to display after training:
    display_graphs(
        used_gpus=8,
        epoch_num=1,
        iter_num=18850
    )

    # --- LOAD MODEL + GENERATE SAMPLES --- #

    # separately load a trained model from a checkpoint:
    model = load_model(
        used_gpus=8,
        epoch_num=1,
        iter_num=18850
    )

    # generate text samples from the model:
    model.sample(
        text="From the depths of",      # starting input text to feed the model
        n_seqs=3,                       # number of (sequences) samples to generate
        max_length=150,                 # maximum length of each generated sequence
        top_k=1000                      # higher k-value for more diverse samples
    )