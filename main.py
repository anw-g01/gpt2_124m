from train import train_gpt2, display_graphs


def main() -> None:
    model = train_gpt2()           


if __name__ == "__main__":

    main()

    # select a checkpoint file to display after training:
    display_graphs(
        ddp_world_size=8,
        num_epochs=1,
        iter_idx=18850
    )