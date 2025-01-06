import torch
from model import GPT2_124M, GPT2Config
from train import train, plot_losses, plot_lr


def load_weights(filename: str) -> GPT2_124M:
    """Load pre-trained model weights from a file and return the model instance."""
    model = GPT2_124M(GPT2Config())         # create a new model instance
    state_dict = torch.load(filename)       # load pre-trained model weights state_dict()
    # strip "_orig_mod." prefix if it exists, due to torch.compile() from training:
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)      # load the cleaned state_dict into the model
    return model


def train_model(plot=True) -> GPT2_124M:
    """Train a GPT-2 (124M) model and return the trained model."""
    model, train_losses, val_losses, learning_rates = train()      # train model
    if plot:
        plot_losses(train_losses, val_losses)
        plot_lr(learning_rates)
    return model


def main() -> None:

    model = train_model()           # train a GPT-2 (124M) model

    outputs = model.sample(
        text="There was once a",    # input starting text
        n_seqs=10,                  # number of sequences to generate
        max_length=50               # maximum length of an output
    )
    print(outputs)                  # dictionary of outputs


if __name__ == "__main__":
    main()