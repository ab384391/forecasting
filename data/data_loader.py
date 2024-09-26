import pandas as pd
import numpy as np
import dask.dataframe as dd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

class FastDataLoader:
    def __init__(self, file_path: str, date_column: str, target_column: str):
        self.file_path = file_path
        self.date_column = date_column
        self.target_column = target_column

    def load_data(self, use_dask: bool = False, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load data efficiently using either pandas or dask.
        
        Args:
            use_dask (bool): Whether to use dask for out-of-memory computations.
            nrows (int, optional): Number of rows to read. None means read all.
        
        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        if use_dask:
            df = dd.read_csv(self.file_path)
            if nrows:
                df = df.head(nrows)
            return df.compute()
        else:
            dtypes = self._infer_dtypes()
            return pd.read_csv(self.file_path, nrows=nrows, dtype=dtypes, parse_dates=[self.date_column])

    def _infer_dtypes(self) -> dict:
        """Infer column dtypes for optimized loading."""
        dtypes = {}
        df_sample = pd.read_csv(self.file_path, nrows=1000)
        
        for col in df_sample.columns:
            if col == self.date_column:
                dtypes[col] = 'datetime64[ns]'
            elif df_sample[col].dtype == 'object':
                dtypes[col] = 'category'
            else:
                dtypes[col] = df_sample[col].dtype
        
        return dtypes

    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, max_lag: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform a time-based train-test split.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            test_size (float): Proportion of the dataset to include in the test split.
            max_lag (int): Maximum lag used in feature engineering to ensure no data leakage.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
        """
        df = df.sort_values(self.date_column)
        split_index = int(len(df) * (1 - test_size))
        
        train = df.iloc[:split_index - max_lag]
        test = df.iloc[split_index:]
        
        return train, test

    def load_and_split(self, test_size: float = 0.2, max_lag: int = 0, use_dask: bool = False, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data and perform train-test split in one step.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            max_lag (int): Maximum lag used in feature engineering to ensure no data leakage.
            use_dask (bool): Whether to use dask for out-of-memory computations.
            nrows (int, optional): Number of rows to read. None means read all.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
        """
        df = self.load_data(use_dask, nrows)
        return self.train_test_split(df, test_size, max_lag)

# Usage example:
# loader = FastDataLoader('path/to/data.csv', 'date_column', 'target_column')
# train_df, test_df = loader.load_and_split(test_size=0.2, max_lag=30, use_dask=True)