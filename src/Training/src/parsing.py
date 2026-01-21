import pandas as pd
import pathlib as pl
from Training.src.module.data import Data

def ft_verify_filename(FileName: str) -> None:
    """
    Verifies that the provided filename has a .csv extension.

    Args:
        FileName (str): The name of the file to verify.

    Raises:
        ValueError: If the file does not have a .csv extension.
    """
    if not FileName.lower().endswith('.csv'):
        raise ValueError("Invalid file format. Please provide a .csv file.")
    
    if not FileName or FileName.isspace():
        raise ValueError("Filename cannot be empty or whitespace.")

    file_path = pl.Path(FileName).resolve()
    project_root = pl.Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"
    try:
        file_path.relative_to(data_dir)
    except ValueError:
        raise ValueError(f"File must be in the 'data/' directory at the project root. Got: {file_path}")
    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path}")
    
def ft_parsing(FileName: str) -> tuple[Data, ...]:
    """
    Parses a CSV file and returns its contents as a tuple of Data objects.

    Args:
        FileName (str): The path to the CSV file to be parsed.
    """
    ft_verify_filename(FileName)
    data = pd.read_csv(FileName)
    data_tuple = tuple(Data(kilometers=row['km'], price=row['price']) for index, row in data.iterrows())

    return data_tuple