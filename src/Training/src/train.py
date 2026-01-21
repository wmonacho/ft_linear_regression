"""
file for parse and train the LR model and read with panda csv

with archi modulaire
"""

import sys
from Training.src.module.data import Data
from Training.src.parsing import ft_parsing
from Training.src.cli import parse_args, print_success

def ft_train(data_tuple: tuple[Data, ...]) -> None:
    """
    Trains a linear regression model using the provided data.

    Args:
        data_tuple (tuple): A tuple of Data objects containing training data.
    """
    pass

def main(*args):
    data_file = parse_args(*args)
    
    print_success(f"Loading data from: \033[1m{data_file}\033[0m")
    data_tuple: tuple[Data, ...] = ft_parsing(data_file)

    ft_train(data_tuple)

    


if __name__ == "__main__":
    main(*sys.argv[1:])