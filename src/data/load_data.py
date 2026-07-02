import os
from pathlib import Path
import pandas as pd


def _resolve_file_path(file_path: str) -> Path:
    """Resolve a data path from the current working directory or the project root."""
    if not file_path:
        raise FileNotFoundError("No file path provided.")

    path = Path(file_path)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        Path(__file__).resolve().parents[2] / path,
        Path(__file__).resolve().parents[2] / "data" / path.name,
    ]

    if path.name != str(path):
        candidates.append(Path(__file__).resolve().parents[2] / "data" / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path.cwd() / path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    resolved_path = _resolve_file_path(file_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        data = pd.read_csv(resolved_path)
        return data
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the data: {e}")