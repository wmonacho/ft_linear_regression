from abc import ABC, abstractmethod
from typing import Tuple
from Training.src.domain.data import Data

class DataRepository(ABC):

    @abstractmethod
    def load_data(self, filepath: str) -> Tuple[Data, ...]:
        """Loads data from the repository based on a file path."""
        pass