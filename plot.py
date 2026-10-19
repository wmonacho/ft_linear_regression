"""Bonus program: plots the dataset and the regression line."""

import sys

import matplotlib.pyplot as plt

from linear_regression.cli import parse_args, print_error
from linear_regression.dataset import load_dataset, Data
from linear_regression.model import estimate_price, load_thetas


def main(args: list[str]) -> None:
    """Plot the dataset given as argument and the trained regression line.

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

    _plot(data, theta0, theta1)


def _plot(data: tuple[Data, ...], theta0: float, theta1: float) -> None:
    """Show the dataset as points and the regression line."""
    kilometers = [d.kilometers for d in data]
    prices = [d.price for d in data]

    plt.figure(figsize=(10, 6))
    plt.scatter(kilometers, prices, label="Dataset")

    km_min = min(kilometers)
    km_max = max(kilometers)
    price_at_min = estimate_price(km_min, theta0, theta1)
    price_at_max = estimate_price(km_max, theta0, theta1)
    plt.plot([km_min, km_max], [price_at_min, price_at_max], color="red", label="Regression line")

    plt.title("Car price by mileage")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
