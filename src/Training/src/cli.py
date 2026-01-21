"""
Command-line interface utilities for the training program.
Handles argument parsing and user-facing messages.
"""

import sys


def print_error(message: str):
    """Print styled error message"""
    print(f"\n\033[91m❌ Error:\033[0m {message}\n")


def print_usage():
    """Print styled usage information"""
    print("\n\033[94m═══════════════════════════════════════════\033[0m")
    print("\033[1m  Linear Regression - Training Program\033[0m")
    print("\033[94m═══════════════════════════════════════════\033[0m\n")
    print("\033[93mUsage:\033[0m")
    print("  python train.py <data_file.csv>\n")
    print("\033[93mExample:\033[0m")
    print("  python train.py data.csv\n")
    print("\033[94m═══════════════════════════════════════════\033[0m\n")


def print_success(message: str):
    """Print styled success message"""
    print(f"\n\033[92m✓\033[0m {message}\n")


def parse_args(*args) -> str:
    """
    Parse and validate command-line arguments.
    
    Args:
        *args: Command-line arguments (excluding program name)
        
    Returns:
        str: The validated data file path
        
    Raises:
        SystemExit: If arguments are invalid
    """
    if len(args) == 0:
        print_error("No data file provided!")
        print_usage()
        sys.exit(1)
    
    if len(args) > 1:
        print_error(f"Too many arguments! Expected 1, got {len(args)}")
        print_usage()
        sys.exit(1)
    
    return args[0]
