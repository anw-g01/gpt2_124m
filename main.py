import torch
from model import GPT2_124M, GPT2Config
from train import train, plot_losses, plot_lr

if __name__ == "__main__":

    # ----- CREATE MODEL INSTANCE AND TRAIN ----- #

    model = GPT2_124M(GPT2Config(vocab_size=50304))     # increase vocab size to (2^7 * 3 * 131)

    trained_model, train_losses, val_losses, learning_rates = train(model)      # train model

    # ----- LOAD MODEL WEIGHTS (OPTIONAL) ----- #
    # state_dict = torch.load(f"<file_name>")     # load pre-trained model weights state_dict()
    # strip "_orig_mod." prefix if it exists, due to torch.compile() from training:
    # clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    # model.load_state_dict(clean_state_dict)      # load the cleaned state_dict into the model

    # ----- PLOT TRAINING/VALIDATION CURVES ----- #

    plot_losses(train_losses, val_losses)
    plot_lr(learning_rates)

    # ----- SAMPLE RESULTS ----- #
    
    outputs = model.sample(
        text="There was once a",    # input starting text
        n_seqs=10,                  # number of sequences to generate
        max_length=50               # maximum length of an output
    )
    print(outputs)      # dictionary of outputs

    # ----- SAVE MODEL WEIGHTS ----- #

    # model.save()     # optional file_name can be passed in