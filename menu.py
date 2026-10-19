"""Interactive launcher: pick a program to run from a numbered menu."""

import subprocess
import sys

from linear_regression.cli import print_error, print_success
from linear_regression.model import THETAS_FILE, load_thetas

DEFAULT_DATASET = "data/data.csv"


def main() -> None:
    try:
        while True:
            _print_menu()
            choice = input("Choice: ").strip()
            if choice == "7":
                break
            if choice not in ACTIONS:
                print_error("Please enter a number between 1 and 7.")
                continue
            try:
                ACTIONS[choice]()
            except KeyboardInterrupt:
                print("\nInterrupted.")
    except (KeyboardInterrupt, EOFError):
        print()
    print("Bye!")


def _print_menu() -> None:
    print("\n=== ft_linear_regression ===")
    print("1. Train the model")
    print("2. Predict a price")
    print("3. Plot dataset and regression line")
    print("4. Show model precision")
    print("5. Show current parameters")
    print("6. Reset training (delete thetas.json)")
    print("7. Quit")


def _run(script: str, *args: str) -> None:
    """Run one of the project's programs with the same Python as the menu."""
    subprocess.run([sys.executable, script, *args])


def _ask_dataset() -> str:
    return input(f"Dataset [{DEFAULT_DATASET}]: ").strip() or DEFAULT_DATASET


def _train() -> None:
    _run("train.py", _ask_dataset())


def _predict() -> None:
    _run("predict.py")


def _plot() -> None:
    _run("plot.py", _ask_dataset())


def _precision() -> None:
    _run("precision.py", _ask_dataset())


def _status() -> None:
    if not THETAS_FILE.exists():
        print("Model not trained: θ0 = θ1 = 0.")
        return
    try:
        theta0, theta1 = load_thetas()
    except ValueError as e:
        print_error(str(e))
        return
    print(f"θ0 = {theta0:.4f}")
    print(f"θ1 = {theta1:.6f}")
    print(f"price = {theta0:.2f} + ({theta1:.6f} × km)")


def _reset() -> None:
    if not THETAS_FILE.exists():
        print("Nothing to reset: the model is not trained.")
        return
    answer = input(f"Delete {THETAS_FILE}? [y/N] ").strip().lower()
    if answer != "y":
        print("Cancelled.")
        return
    THETAS_FILE.unlink()
    print_success("Training reset: θ0 = θ1 = 0.")


ACTIONS = {
    "1": _train,
    "2": _predict,
    "3": _plot,
    "4": _precision,
    "5": _status,
    "6": _reset,
}


if __name__ == "__main__":
    main()
