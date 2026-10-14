"""Loading of the training dataset from a CSV file."""

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("km", "price")
MIN_ROWS = 2


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
    df = _read_csv(filepath)

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Dataset must contain a '{column}' column: {filepath}")
    if len(df) < MIN_ROWS:
        raise ValueError(f"Dataset must contain at least {MIN_ROWS} rows: {filepath}")
    for column in REQUIRED_COLUMNS:
        _validate_column(df[column], column, filepath)

    return tuple(
        Data(kilometers=float(km), price=float(price))
        for km, price in zip(df["km"], df["price"])
    )


def _validate_path(filepath: str) -> None:
    if not filepath or filepath.isspace():
        raise ValueError("Filename cannot be empty or whitespace.")
    if not filepath.lower().endswith(".csv"):
        raise ValueError("Invalid file format. Please provide a .csv file.")
    if not Path(filepath).is_file():
        raise ValueError(f"File does not exist: {filepath}")


def _read_csv(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset is empty: {filepath}") from None
    except pd.errors.ParserError:
        raise ValueError(f"Malformed CSV (inconsistent number of fields): {filepath}") from None
    except UnicodeDecodeError:
        raise ValueError(f"Not a valid text CSV file: {filepath}") from None
    except OSError as e:
        raise ValueError(f"Cannot read {filepath}: {e.strerror}") from None


def _validate_column(values: pd.Series, column: str, filepath: str) -> None:
    """Check that every value is a finite, non-negative number."""
    # Line 1 is the header, so the first data row is line 2.
    for line, raw in enumerate(values, start=2):
        where = f"{filepath}, line {line}"
        if pd.isna(raw):
            raise ValueError(f"{where}: missing '{column}' value.")
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{where}: '{column}' is not a number ({raw!r}).") from None
        if not math.isfinite(value):
            raise ValueError(f"{where}: '{column}' must be a finite number ({raw!r}).")
        if value < 0:
            raise ValueError(f"{where}: '{column}' must be non-negative ({raw!r}).")
