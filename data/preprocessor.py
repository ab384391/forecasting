import pandas as pd
import numpy as np
from typing import List, Dict, Union
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf
from joblib import Parallel, delayed

class FastPreprocessor:
    def __init__(self, numerical_columns: List[str], categorical_columns: List[str], target_column: str):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, tf.keras.layers.StringLookup] = {}

    def preprocess(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Preprocess the data efficiently.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            is_train (bool): Whether this is training data.
        
        Returns:
            pd.DataFrame: Preprocessed dataframe.
        """
        df = self._handle_missing_values(df)
        df = self._handle_infinite_values(df)
        df = self._encode_categorical_variables(df, is_train)
        df = self._normalize_numerical_variables(df, is_train)
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values efficiently."""
        for col in self.numerical_columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        for col in self.categorical_columns:
            df[col] = df[col].astype('category').cat.add_categories(['Unknown']).fillna('Unknown')
        
        return df

    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinity values efficiently."""
        return df.replace([np.inf, -np.inf], np.nan)

    def _encode_categorical_variables(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Encode categorical variables using TensorFlow's StringLookup layer."""
        for col in self.categorical_columns:
            if is_train or col not in self.encoders:
                encoder = tf.keras.layers.StringLookup(output_mode='multi_hot')
                encoder.adapt(df[col].astype(str))
                self.encoders[col] = encoder
            
            encoded = self.encoders[col](df[col].astype(str))
            encoded_df = pd.DataFrame(encoded.numpy(), columns=[f"{col}_{i}" for i in range(encoded.shape[1])])
            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
        
        return df

    def _normalize_numerical_variables(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Normalize numerical variables using StandardScaler."""
        for col in self.numerical_columns:
            if is_train or col not in self.scalers:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                df[col] = self.scalers[col].transform(df[[col]])
        
        return df

    def create_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create time-based features efficiently."""
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        
        return df

    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lag features efficiently using parallel processing."""
        def create_lag(df: pd.DataFrame, column: str, lag: int) -> pd.Series:
            return df[column].shift(lag).rename(f"{column}_lag_{lag}")

        lagged_features = Parallel(n_jobs=-1)(
            delayed(create_lag)(df, col, lag)
            for col in columns
            for lag in lags
        )

        return pd.concat([df] + lagged_features, axis=1)

    def prune_features(self, df: pd.DataFrame, variance_threshold: float = 0.0, correlation_threshold: float = 0.95) -> pd.DataFrame:
        """Prune features based on variance and correlation."""
        # Drop zero variance features
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(df)
        df = df.loc[:, selector.get_support()]

        # Drop highly correlated features
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        df = df.drop(columns=to_drop)

        return df

# Usage example:
# preprocessor = FastPreprocessor(numerical_columns=['num1', 'num2'], 
#                                 categorical_columns=['cat1', 'cat2'], 
#                                 target_column='target')
# processed_df = preprocessor.preprocess(df, is_train=True)
# processed_df = preprocessor.create_time_features(processed_df, 'date_column')
# processed_df = preprocessor.create_lag_features(processed_df, ['target'], [1, 7, 30])
# processed_df = preprocessor.prune_features(processed_df)