"""Bonus program: measures the precision of the trained model on a dataset."""

import sys

from linear_regression.cli import parse_args, print_error
from linear_regression.dataset import Data, load_dataset
from linear_regression.maths import mean_absolute_error, r_squared, root_mean_squared_error
from linear_regression.model import estimate_price, load_thetas


def main(args: list[str]) -> None:
    """Compare the model's predictions with the real prices of the dataset given as argument.

    Args:
        args (list[str]): Command line arguments.
    """
    data_file = parse_args(args)

    try:
        data = load_dataset(data_file)
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)

    try:
        theta0, theta1 = load_thetas()
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)

    if theta0 == 0 and theta1 == 0:
        print("Warning: Model parameters are not trained yet. Please run the training program first.")

    _print_precision(data, theta0, theta1)


def _print_precision(data: tuple[Data, ...], theta0: float, theta1: float) -> None:
    """Print MAE, RMSE and R² of the model on the given data."""
    actual = tuple(d.price for d in data)
    predicted = tuple(estimate_price(d.kilometers, theta0, theta1) for d in data)

    print(f"MAE  : {mean_absolute_error(actual, predicted):.2f}  (mean absolute error)")
    print(f"RMSE : {root_mean_squared_error(actual, predicted):.2f}  (root mean squared error)")
    try:
        print(f"R²   : {r_squared(actual, predicted):.4f}  (coefficient of determination, 1 = perfect)")
    except ValueError as e:
        print(f"R²   : undefined ({e})")


if __name__ == "__main__":
    main(sys.argv[1:])
