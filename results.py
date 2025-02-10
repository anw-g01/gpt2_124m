import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams["font.size"] = 9
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["font.family"] = "monospace"
from config import LOG_DIR
from train import _get_checkpoint_filename
from model import GPT2_124M, GPT2Config


def display_graphs(
        used_gpus: int,
        epoch_num: int,
        step_num: int,
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
        `step_num` (`int`): step number (starting from `1`) for the checkpoint.
        `log_dir` (`str`): directory where the logs and checkpoints are stored. Default: `"LOG_DIR"` from `config.py`.
        `checkpoint_type` (`str`): prefix of the checkpoint file name (must be `"end"` or `"val"`).  Default: `"end"`.
        `plot_lr` (`bool`): whether to plot the learning rate on the same plot as the losses. Default: `"True"`.
        `save` (`bool`): whether to save the plot as an image file. Default: `"True"`.
        `img_format` (`str`): format to save the plot image (e.g., `"png"`, `"jpg"`). Default: `"png"`.
    """
    assert checkpoint_type in ["end", "val"], "checkpoint_type must be 'end' or 'val'"

    # get the checkpoint directory from the filename (for the specified epoch and step number):
    filename = _get_checkpoint_filename(checkpoint_type, epoch_num, step_num, used_gpus)
    checkpoint_dir = os.path.join(log_dir, filename)
    assert os.path.exists(checkpoint_dir), f"checkpoint directory does not exist: {checkpoint_dir}"     # check if the directory exists
    print(f"\nusing checkpoint directory: {filename}")
    print(f"\nloading training plots...")
    # load numpy arrays from the checkpoint directory:
    train_losses = np.load(os.path.join(checkpoint_dir, "train_losses.npy"))
    val_losses = np.load(os.path.join(checkpoint_dir, "val_losses.npy"))
    hellaswag_scores = np.load(os.path.join(checkpoint_dir, "hellaswag_scores.npy"))
    learning_rates = np.load(os.path.join(checkpoint_dir, "learning_rates.npy"))      
      
    # truncate arrays up to step_idx (default):
    if truncate:
        idx = min(step_num, len(train_losses))      # in case step_num > len(train_losses) 
        train_losses = train_losses[:idx]           # valid because steps were saved as i + 1 
        val_losses = val_losses[:idx]
        hellaswag_scores = hellaswag_scores[:idx]
        learning_rates = learning_rates[:idx]

    # ---------- MAIN FIGURE ---------- #
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    fig.suptitle(f"GPT-2 (124M) Training Results", fontsize=14, fontweight="bold")      # title for the entire figure
    x = np.arange(1, len(train_losses) + 1)             # x-values for all plots
    
    # ------ LEFT SUBPLOT ------ #
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("step")
    axs[0].set_ylabel("loss")
    axs[0].grid(True)
    axs[0].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):,}'))    # comma separator for x-axis tickers
    line_colours = ["tab:cyan", "tab:olive", "tab:purple", "tab:red"]     # colours for lr, train, val, and baseline losses
    # optional: plot learning rates in a twin axis (with losses figure):
    if plot_lr:
        # create a second y-axis for learning rates:   
        ax_lr = axs[0].twinx()  
        ax_lr.set_ylabel("learning rate", color=line_colours[0])
        ax_lr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))     # use scientific notation
        ax_lr.tick_params(axis="y", labelcolor=line_colours[0])             # set axis tick colour
        alpha_val = 0.3
        ax_lr.plot(
            x, learning_rates, label="learning rate",
            color=line_colours[0], alpha=alpha_val,      # set alpha for transparency    
        )
        # adjust the transparency of the right y-axis tick labels:
        # for label in ax_lr.get_yticklabels():
        #     label.set_alpha(alpha_val)
        # ax_lr.yaxis.label.set_alpha(alpha_val)  # set transparency for the y-axis label
    # plot training and validation losses:
    axs[0].plot(
        x, train_losses,
        label="train loss", color=line_colours[1],
        marker=".", markevery=[0, -1]           # mark first and last data point
    )
    # for val_loss, only plot non-NaN values due to interval storage:
    val_idx = np.isfinite(val_losses)           # cell turns False if cell was NaN
    axs[0].plot(
        x[val_idx], val_losses[val_idx],        # only select elements where val_idx is True (i.e. non-NaN)  
        label="val loss", color=line_colours[2],
        marker=".", markevery=[0, -1]           # mark first and last data point
    )
    # text labels for the first and last data points:
    offset = 0.1
    axs[0].text(0, train_losses[0] + offset, f"{train_losses[0]:.2f}", 
        color=line_colours[1], ha='left', va='bottom')#, weight="bold")
    axs[0].text(x[val_idx][0], val_losses[val_idx][0] + offset, f"{val_losses[val_idx][0]:.2f}", 
        color=line_colours[2], ha='center', va='bottom')#, weight="bold")
    axs[0].text(x[-1], train_losses[-1] - offset, f"{train_losses[-1]:.2f}", 
        color=line_colours[1], ha='center', va='top')#, weight="bold")
    axs[0].text(x[val_idx][-1], val_losses[val_idx][-1] - offset, f"{val_losses[val_idx][-1]:.2f}", 
        color=line_colours[2], ha='center', va='top')#, weight="bold")
    # configure legends:
    if plot_lr:
        # combine legends for twin axes:
        lines_lr, labels_lr = ax_lr.get_legend_handles_labels()
        lines, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(lines_lr + lines, labels_lr + labels, loc="upper right")
    else:
        axs[0].legend()                                     # legend for losses
    # plot baseline losses from other models:
    axs[0].axhline(y=3.292, color=line_colours[3], linestyle="--")      # plot baseline loss last (for legend ordering)
    axs[0].text(
        0.95, 3.292 + 0.1, "OpenAI GPT-2 (124M) val loss: 3.29",        # x in axis coordinates, y in data coordinates
        color="tab:red", ha='right', va='bottom',                       # horizontal and vertical alignment
        transform=axs[0].get_yaxis_transform(),                          # transform to y-axis coordinates
    )

    # ------ RIGHT SUBPLOT  ------ #
    axs[1].set_title("HellaSwag Evaluation")
    axs[1].set_xlabel("step")
    axs[1].set_ylabel("accuracy (%)")
    axs[1].grid(True)
    axs[1].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):,}'))    # comma separator for x-axis tickers
    
    # plot HellaSwag scores:
    hs_idx = np.isfinite(hellaswag_scores)      # bool: False if cell is NaN
    axs[1].plot(
        x[hs_idx], hellaswag_scores[hs_idx],
        label="HellaSwag score", color="tab:orange",
        marker=".", markevery=[0, -1]           # mark first and data point
    )
    # text labels for the first and last data points:
    offset = 0.2
    axs[1].text(0, hellaswag_scores[0] - offset, f"{hellaswag_scores[0]:.1f}%", color="tab:orange", ha='left', va='top')#, weight="bold")
    axs[1].text(x[hs_idx][-1], hellaswag_scores[hs_idx][-1] + offset, f"{hellaswag_scores[hs_idx][-1]:.1f}%",
                 color="tab:orange", ha='center', va='bottom')#, weight="bold")
    axs[1].legend(loc="center right")     # legend for right subplot

    # plot baseline scores from other models:
    hs_colours = ["tab:green", "tab:blue", "tab:red", "tab:brown"]
    hs_scores = [37.5, 33.7, 29.6, 25.0]     # GPT-3 (124M), GPT-2 (124M), and naive baseline
    for score, colour in zip(hs_scores, hs_colours):
        axs[1].axhline(y=score, color=colour, linestyle="--")     # plot horizontal lines as model scores
    # add text just above axhline lines:
    offset = 0.25
    axs[1].text(0.95, hs_scores[0] - offset, f"OpenAI GPT-2 (350M) score: {hs_scores[0]}%", color=hs_colours[0],
        ha='right', va='top', transform=axs[1].get_yaxis_transform())
    axs[1].text(0.95, hs_scores[1] + offset, f"OpenAI GPT-3 (124M) score: {hs_scores[1]}%", color=hs_colours[1],
        ha='right', va='bottom', transform=axs[1].get_yaxis_transform())
    axs[1].text(0.95, hs_scores[2] + offset, f"OpenAI GPT-2 (124M) score: {hs_scores[2]}%", color=hs_colours[2],
        ha='right', va='bottom', transform=axs[1].get_yaxis_transform())
    axs[1].text(0.95, hs_scores[3] + offset, f"naive baseline: {hs_scores[3]}%", color=hs_colours[3],
        ha='right', va='bottom', transform=axs[1].get_yaxis_transform())

    plt.tight_layout()  
    if save:
        print(f"\nfigure saved as 'figure_{filename}.{img_format}'")
        plt.savefig(
            f"figure_{filename}.{img_format}", 
            dpi=300, bbox_inches="tight",
        )
    if not save:
        plt.show()      # code will pause until the plot window is closed

    # get directory path to the .pt checkpoint file:
    print(f"\nloading model weights...\n")
    model_checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pt")     # path to the .pt file holding dictionary of checkpoints file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           # set device to GPU if available
    # load the .pt file for dictionary of checkpoints
    checkpoint_dict = torch.load(
        model_checkpoint_path,
        map_location=device,        # ensures loading onto the CPU (if no GPU available), if originally saved on a GPU
        weights_only=True
    )               
    
    # create a new model instance and load the state_dict from the dictionary:
    model = GPT2_124M(GPT2Config(vocab_size=50304))                         # create new model instance, must be same config as trained model
    state_dict = _filter_state_dict(checkpoint_dict["model_state_dict"])    # filter the state_dict keys to handle DDP model loading
    
    model.load_state_dict(state_dict)                           
    model = model.to(device)

    # generate text samples from the model:
    model.sample("Hello, I'm a")     


def load_model(
        used_gpus: int,
        epoch_num: int,
        step_num: int,
        log_dir: str = LOG_DIR,
        checkpoint_type: str = "end",
    ) -> GPT2_124M:
    """
    Load a trained model from a specified checkpoint directory.

    Args:
    --
        `used_gpus` (`int`): number of GPUs used for training.
        `epoch_num` (`int`): epoch number (starting from `1`) for the checkpoint.
        `step_num` (`int`): step number (starting from `1`) for the checkpoint.
        `log_dir` (`str`): directory where the logs and checkpoints are stored.
        `checkpoint_type` (`str`): prefix of the checkpoint file name (must be `"end"` or `"val"`). Default: `"end"`.

    Returns:
    --
        `model` (`GPT2_124M`): GPT-2 model instance with loaded weights from a checkpoint.
    """
    # get the checkpoint directory from the filename (for the specified epoch and step):
    filename = _get_checkpoint_filename(checkpoint_type, epoch_num, step_num, used_gpus)
    checkpoint_dir = os.path.join(log_dir, filename)
    assert os.path.exists(checkpoint_dir), f"checkpoint directory does not exist: {checkpoint_dir}"     # check if the directory exists
    print(f"\nloading model from: {filename}...")
    
    # get full path to the .pt file holding dictionary of checkpoints:
    model_checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pt")  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # set device to GPU (single default 0) if available
    # load the .pt file for dictionary of checkpoints
    checkpoint_dict = torch.load(
        model_checkpoint_path,
        map_location=device,        # ensures loading onto the CPU (if no GPU available), if originally saved on a GPU
        weights_only=True
    )        
    state_dict = _filter_state_dict(checkpoint_dict["model_state_dict"])     # filter the state_dict keys to handle DDP model loading

    # create a new model instance and load the state_dict from the dictionary:
    print(f"loading model weights...\n")
    model = GPT2_124M(GPT2Config(vocab_size=50304))     # create new model instance, must be same config as trained model
    model.load_state_dict(state_dict)                   
    model = model.to(device)

    return model    # return the trained model


def _filter_state_dict(state_dict: dict) -> dict:
    """
    Helper function to filters the keys in the state dictionary by removing the `"module."` prefix (due to 
    `DDP`) or `"_orig_mod."` (due to `torch.compile`) if either are present.

    N.B. `"module."` won't be present if the raw model (`model.module`) was used for saving the checkpoint.

    Args:
    --
        `state_dict` (`dict`): The state dictionary to filter.

    Returns:
    --
        `dict`: A new state dictionary with the `"module."` prefix removed from the keys.
    """
    # remove "module." prefix if present (due to DDP)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    # remove "_orig_mod." prefix if present (due to torch.compile)
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

    return state_dict    # return the filtered state dictionary


if __name__ == "__main__":

    # --- DISPLAY TRAINING CURVES --- #

    # select a checkpoint file to display after training:
    display_graphs(
        used_gpus=1,
        epoch_num=8,
        step_num=18850,
    )

    # # --- LOAD MODEL + GENERATE SAMPLES --- #

    # # separately load a trained model from a checkpoint:
    model = load_model(
        used_gpus=1,
        epoch_num=8,
        step_num=18850,
    )

    # # generate text samples from the model:
    model.sample(
        input="\n",         # starting input text to feed the model
        n_seqs=3,           # number of (sequences) samples to generate
        max_length=120,     # maximum length of each generated sequence
        k=1000              # higher k-value for more diverse samples
    )