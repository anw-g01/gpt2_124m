from train import train_gpt2


def main() -> None:
    model = train_gpt2(compile=False)           


if __name__ == "__main__":
    main()