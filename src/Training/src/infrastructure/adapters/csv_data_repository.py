import pandas as pd
import pathlib as pl
from Training.src.domain.data import Data
from Training.src.application.ports.data_repository import DataRepository

class CsvDataRepository(DataRepository):
    def load_data(self, filepath: str) -> tuple[Data, ...]:
        """
        Loads data from a CSV file and returns it as a tuple of Data objects.

        Args:
            filepath (str): The path to the CSV file to be loaded.
        """
        self._validate(filepath)
        df = pd.read_csv(filepath)
        return tuple(
            Data(kilometers=row['km'], price=row['price'])
            for _, row in df.iterrows()
        )

    def _validate(self, filepath: str) -> None:
        """
        Validates the provided file path to ensure it points to a valid CSV file.

        Args:
            filepath (str): The path to the file to validate.

        Raises:
            ValueError: If the file does not have a .csv extension or if the file does not exist.
        """
        if not filepath.lower().endswith('.csv'):
            raise ValueError("Invalid file format. Please provide a .csv file.")
        
        if not filepath or filepath.isspace():
            raise ValueError("Filename cannot be empty or whitespace.")

        file_path = pl.Path(filepath).resolve()
        project_root = pl.Path(__file__).resolve().parents[3]
        data_dir = project_root / "data"
        try:
            file_path.relative_to(data_dir)
        except ValueError:
            raise ValueError(f"File must be in the 'data/' directory at the project root. Got: {file_path}")
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
