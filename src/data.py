from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Загружает CSV-датасет по указанному пути."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)
