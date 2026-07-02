import pandas as pd
import os
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the data: {e}")