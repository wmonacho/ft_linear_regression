"""Terminal output and argument parsing."""

import sys


def print_error(message: str) -> None:
    print(f"\n\033[91m❌ Error:\033[0m {message}\n", file=sys.stderr)


def print_success(message: str) -> None:
    print(f"\n\033[92m✓\033[0m {message}\n")


def print_usage() -> None:
    print("\n\033[94m═══════════════════════════════════════════\033[0m")
    print("\033[1m  Linear Regression - Training Program\033[0m")
    print("\033[94m═══════════════════════════════════════════\033[0m\n")
    print("\033[93mUsage:\033[0m")
    print("  python train.py <data_file.csv>\n")
    print("\033[93mExample:\033[0m")
    print("  python train.py data/data.csv\n")
    print("\033[94m═══════════════════════════════════════════\033[0m\n")


def parse_args(args: list[str]) -> str:
    """Return the data file path, or exit with the usage if arguments are invalid."""
    if len(args) != 1:
        if not args:
            print_error("No data file provided!")
        else:
            print_error(f"Too many arguments! Expected 1, got {len(args)}")
        print_usage()
        sys.exit(1)
    return args[0]
