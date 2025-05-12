from typing import Tuple
import pandas as pd
import numpy as np
import warnings


def load_plant_data(file_path: str = "data/plant_knowledge.csv") -> np.ndarray:
    """Load and preprocess plant knowledge data."""
    try:
        df = pd.read_csv(file_path)
       
        # Remove the 'Informant' column explicitly - This removal was created by AI
        if 'Informant' in df.columns:
            print("Removing 'Informant' ID column")
            df = df.drop(columns='Informant')
            data = df.to_numpy().astype(int)
        else:
            raise ValueError("CSV must contain an 'Informant' column as the first column")
       
        validate_data(data)
        print(f"Successfully loaded data with shape {data.shape}")
        return data


    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {file_path}") from None


def validate_data(data: np.ndarray) -> None:
    """Ensure data contains only binary values. This part was suggested by AI, implemented by me and improved by AI"""
    if not np.all(np.isin(data, [0, 1])):
        bad_rows, bad_cols = np.where(~np.isin(data, [0, 1]))
        error_locations = [(r+1, c+1) for r, c in zip(bad_rows[:5], bad_cols[:5])]  # Show first 5 errors
        raise ValueError(
            "Data validation failed:\n"
            f"- Found {len(bad_rows)} non-binary values\n"
            f"- First errors at (row, column): {error_locations}\n"
            "Please ensure all responses are 0 or 1"
        )
