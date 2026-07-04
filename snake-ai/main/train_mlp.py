import sys

from train import main


if __name__ == "__main__":
    main(["--agent", "mlp", *sys.argv[1:]])
