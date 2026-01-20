import pandas as pd



def parsing(FileName: str) -> pd.DataFrame:
    """
    Parses a CSV file and returns its contents as a pandas DataFrame.

    Args:
        FileName (str): The path to the CSV file to be parsed.
        """
    
    data = pd.read_csv(FileName)
    

    