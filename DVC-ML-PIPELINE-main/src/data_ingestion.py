"""
Stage 1: Data Ingestion
------------------------
Loads the Boston Housing dataset (regression: predicts median home value
'MEDV' in $1000s from 13 neighborhood/structural features) and dumps it
as a raw CSV file.

Note: sklearn removed `load_boston()` in 1.2 due to ethical concerns
documented in its dataset description, so we pull the original data
directly from the source CMU StatLib file instead.

Output:
    data/raw/data.csv
"""

import os
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]
DATA_URL = "http://lib.stat.cmu.edu/datasets/boston"


def load_data() -> pd.DataFrame:
    """Load the Boston Housing dataset into a DataFrame."""
    raw_df = pd.read_csv(DATA_URL, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    df = pd.DataFrame(data, columns=FEATURE_NAMES)
    df["target"] = target  # MEDV: median value of owner-occupied homes ($1000s)
    return df


def save_raw_data(df: pd.DataFrame, out_dir: str = "data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.csv")
    df.to_csv(out_path, index=False)
    print(f"[data_ingestion] Saved raw data -> {out_path} (shape={df.shape})")


def main():
    df = load_data()
    save_raw_data(df)


if __name__ == "__main__":
    main()