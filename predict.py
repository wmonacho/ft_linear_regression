"""Prediction program: prompts for a mileage and prints the estimated price."""

import math
import sys

from linear_regression.cli import print_error
from linear_regression.model import estimate_price, load_thetas


def main() -> None:
    try:
        theta0, theta1 = load_thetas()
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)

    if theta0 == 0 and theta1 == 0:
        print("Warning: Model parameters are not trained yet. Please run the training program first.")

    try:
        mileage = _ask_mileage()
    except EOFError:
        print_error("No input received (Ctrl-D).")
        sys.exit(1)
    except KeyboardInterrupt:
        print_error("Interrupted (Ctrl-C).")
        sys.exit(130)

    price = estimate_price(mileage, theta0, theta1)
    _print_price(mileage, price)


def _ask_mileage() -> float:
    """
    Prompt the user for a mileage until a valid one is entered.

    Raises:
        EOFError: If the input is closed (Ctrl-D).
        KeyboardInterrupt: If the user presses Ctrl-C.
    """
    while True:
        text = input("Enter mileage: ")
        try:
            return _parse_mileage(text)
        except ValueError as e:
            print_error(f"Invalid mileage: {e}")


def _parse_mileage(text: str) -> float:
    """
    Convert the user input into a mileage.

    Raises:
        ValueError: If the input is empty, not a number, not finite or negative.
    """
    text = text.strip()
    if text == "":
        raise ValueError("Input cannot be empty.")
    try:
        mileage = float(text)
    except ValueError:
        raise ValueError(f"'{text}' must be a number.") from None
    if not math.isfinite(mileage):
        raise ValueError(f"'{text}' must be a finite number.")
    if mileage < 0:
        raise ValueError(f"'{text}' must be non-negative.")
    return mileage


def _print_price(mileage: float, price: float) -> None:
    """Display the estimated price for the given mileage."""
    if price < 0:
        print(f"Warning: Estimated price is negative ({price:.2f}). This may indicate that the mileage is too high: {mileage}")
    print(f"Estimated price for mileage {mileage} is: {price:.2f}.")


if __name__ == "__main__":
    main()
