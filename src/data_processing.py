from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class DataProcessor:
    path: str

    def load(self) -> pd.DataFrame:
        """Load dataset from CSV."""
        return pd.read_csv(self.path)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: remove duplicates + fill missing values."""
        df = df.drop_duplicates()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])

        return df

    def split_xy(self, df: pd.DataFrame, target_col: str):
        """Split dataframe into features X and target y."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y
