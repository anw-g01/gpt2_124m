from train import train_gpt2


def main() -> None:
    train_gpt2(
        compile=True,
        # eval=False        # uncomment: don't evaulate on HellaSwag IF issues arise with torch.compile()
    )           


if __name__ == "__main__":

    main()
