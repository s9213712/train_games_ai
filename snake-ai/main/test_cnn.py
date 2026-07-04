import sys

from evaluate import main


if __name__ == "__main__":
    main(["--agent", "cnn", *sys.argv[1:]])
