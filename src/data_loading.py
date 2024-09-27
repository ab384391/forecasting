import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import dask.dataframe as dd
from src.config import DATA_CONFIG

def load_data(file_path: str, use_dask: bool = False) -> pd.DataFrame:
    """
    Load data from a CSV file using either pandas or dask for larger datasets.
    
    Args:
        file_path (str): Path to the CSV file.
        use_dask (bool): Whether to use dask for loading large datasets.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    if use_dask:
        ddf = dd.read_csv(file_path, 
                          usecols=DATA_CONFIG['date_column'] + 
                                  DATA_CONFIG['target_column'] + 
                                  DATA_CONFIG['categorical_columns'] + 
                                  DATA_CONFIG['numerical_columns'])
        df = ddf.compute()
    else:
        df = pd.read_csv(file_path, 
                         usecols=DATA_CONFIG['date_column'] + 
                                 DATA_CONFIG['target_column'] + 
                                 DATA_CONFIG['categorical_columns'] + 
                                 DATA_CONFIG['numerical_columns'],
                         parse_dates=[DATA_CONFIG['date_column']])
    
    return df

def time_based_split(df: pd.DataFrame, 
                     date_column: str, 
                     test_size: float = 0.2,
                     max_lag: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a time-based train-test split.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        date_column (str): Name of the date column.
        test_size (float): Proportion of the dataset to include in the test split.
        max_lag (int, optional): Maximum lag used in feature engineering, to ensure no data leakage.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
    """
    df = df.sort_values(date_column)
    
    split_index = int(len(df) * (1 - test_size))
    if max_lag:
        split_index -= max_lag  # Adjust split to account for lagged features
    
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    
    return train, test

def load_and_split_data(file_path: str, 
                        test_size: float = 0.2, 
                        use_dask: bool = False,
                        max_lag: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data and perform time-based train-test split.
    
    Args:
        file_path (str): Path to the CSV file.
        test_size (float): Proportion of the dataset to include in the test split.
        use_dask (bool): Whether to use dask for loading large datasets.
        max_lag (int, optional): Maximum lag used in feature engineering.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
    """
    df = load_data(file_path, use_dask)
    train, test = time_based_split(df, DATA_CONFIG['date_column'], test_size, max_lag)
    
    return train, test

if __name__ == "__main__":
    # Example usage
    train_data, test_data = load_and_split_data(DATA_CONFIG['raw_data_path'], 
                                                test_size=0.2, 
                                                use_dask=True,
                                                max_lag=30)
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
