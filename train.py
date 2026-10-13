"""Training program: reads the dataset, trains the model and saves θ0 and θ1."""

import sys

from linear_regression.cli import parse_args, print_error, print_success
from linear_regression.dataset import load_dataset
from linear_regression.model import save_thetas
from linear_regression.training import train_model


def main(args: list[str]) -> None:
    data_file = parse_args(args)

    print_success(f"Loading data from: \033[1m{data_file}\033[0m")
    try:
        data = load_dataset(data_file)
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)

    theta0, theta1 = train_model(data)
    try:
        save_thetas(theta0, theta1)
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)
    print_success("Model parameters saved.")


if __name__ == "__main__":
    main(sys.argv[1:])
