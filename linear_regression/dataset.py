"""Loading of the training dataset from a CSV file."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Data:
    kilometers: float
    price: float


def load_dataset(filepath: str) -> tuple[Data, ...]:
    """
    Load a CSV file with 'km' and 'price' columns.

    Raises:
        ValueError: If the file is invalid or cannot be read.
    """
    _validate_path(filepath)
    df = pd.read_csv(filepath)
    # TODO: valider le contenu (colonnes 'km'/'price' présentes, fichier vide,
    #       valeurs non numériques ou manquantes) et lever ValueError sinon.
    return tuple(
        Data(kilometers=row["km"], price=row["price"])
        for _, row in df.iterrows()
    )


def _validate_path(filepath: str) -> None:
    if not filepath or filepath.isspace():
        raise ValueError("Filename cannot be empty or whitespace.")
    if not filepath.lower().endswith(".csv"):
        raise ValueError("Invalid file format. Please provide a .csv file.")
    if not Path(filepath).is_file():
        raise ValueError(f"File does not exist: {filepath}")
