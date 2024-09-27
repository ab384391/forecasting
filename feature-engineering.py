import pandas as pd
import numpy as np
from typing import List, Dict, Union
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import boxcox
import joblib
from src.config import FEATURE_ENGINEERING_CONFIG, DATA_CONFIG, SAVE_LOAD_CONFIG

class AutoFeatureEngineer:
    def __init__(self, config: Dict[str, bool] = FEATURE_ENGINEERING_CONFIG):
        self.config = config
        self.date_column = DATA_CONFIG['date_column']
        self.target_column = DATA_CONFIG['target_column']
        self.numeric_columns = DATA_CONFIG['numerical_columns']
        self.categorical_columns = DATA_CONFIG['categorical_columns']
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.feature_importances = {}

    def fit(self, df: pd.DataFrame) -> 'AutoFeatureEngineer':
        """
        Fit the feature engineer to the data.
        
        Args:
            df (pd.DataFrame): Input dataframe.
        
        Returns:
            AutoFeatureEngineer: Fitted feature engineer.
        """
        if self.config['time_based_features']:
            self._fit_time_features(df)
        
        if self.config['lag_features']:
            self._fit_lag_features(df)
        
        if self.config['interaction_features']:
            self._fit_interaction_features(df)
        
        if self.config['transformation_features']:
            self._fit_transformation_features(df)
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted feature engineer.
        
        Args:
            df (pd.DataFrame): Input dataframe.
        
        Returns:
            pd.DataFrame: Transformed dataframe with engineered features.
        """
        transformed_df = df.copy()
        
        if self.config['time_based_features']:
            transformed_df = self._transform_time_features(transformed_df)
        
        if self.config['lag_features']:
            transformed_df = self._transform_lag_features(transformed_df)
        
        if self.config['interaction_features']:
            transformed_df = self._transform_interaction_features(transformed_df)
        
        if self.config['transformation_features']:
            transformed_df = self._transform_transformation_features(transformed_df)
        
        if self.config['differencing']:
            transformed_df = self._apply_differencing(transformed_df)
        
        if self.config['fourier_features']:
            transformed_df = self._create_fourier_features(transformed_df)
        
        return transformed_df

    def _fit_time_features(self, df: pd.DataFrame):
        # No fitting required for time-based features
        pass

    def _transform_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['year'] = df[self.date_column].dt.year
        df['month'] = df[self.date_column].dt.month
        df['day'] = df[self.date_column].dt.day
        df['dayofweek'] = df[self.date_column].dt.dayofweek
        df['quarter'] = df[self.date_column].dt.quarter
        return df

    def _fit_lag_features(self, df: pd.DataFrame):
        # Determine optimal lags using autocorrelation
        target_autocorr = df[self.target_column].autocorr(lag=1)
        self.lags = [1, 7, 30]  # Example lags, can be adjusted based on autocorrelation

    def _transform_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for lag in self.lags:
            df[f'target_lag_{lag}'] = df[self.target_column].shift(lag)
        return df

    def _fit_interaction_features(self, df: pd.DataFrame):
        self.poly_features.fit(df[self.numeric_columns])

    def _transform_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        poly_features = self.poly_features.transform(df[self.numeric_columns])
        poly_feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
        df[poly_feature_names] = poly_features
        return df

    def _fit_transformation_features(self, df: pd.DataFrame):
        # Determine which features benefit from transformation
        self.transform_features = []
        for col in self.numeric_columns:
            if np.abs(df[col].skew()) > 0.5:  # Arbitrary threshold
                self.transform_features.append(col)

    def _transform_transformation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.transform_features:
            df[f'{col}_log'] = np.log1p(df[col])
            df[f'{col}_sqrt'] = np.sqrt(df[col])
            try:
                df[f'{col}_boxcox'], _ = boxcox(df[col] + 1)  # Adding 1 to handle zero values
            except:
                pass  # Skip if BoxCox transformation fails
        return df

    def _apply_differencing(self, df: pd.DataFrame) -> pd.DataFrame:
        df['target_diff'] = df[self.target_column].diff()
        return df

    def _create_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Assuming daily data, adjust as needed
        for period in [365.25, 7]:  # Annual and weekly cycles
            for order in [1, 2, 3]:
                df[f'fourier_sin_{period}_{order}'] = np.sin(2 * np.pi * order * df.index.dayofyear / period)
                df[f'fourier_cos_{period}_{order}'] = np.cos(2 * np.pi * order * df.index.dayofyear / period)
        return df

    def calculate_feature_importance(self, df: pd.DataFrame):
        """
        Calculate feature importance using mutual information and F-test.
        
        Args:
            df (pd.DataFrame): Input dataframe with target variable.
        """
        X = df.drop(columns=[self.target_column, self.date_column])
        y = df[self.target_column]
        
        mi_scores = mutual_info_regression(X, y)
        f_scores, _ = f_regression(X, y)
        
        self.feature_importances = {
            'mutual_info': dict(zip(X.columns, mi_scores)),
            'f_test': dict(zip(X.columns, f_scores))
        }

    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Generate a feature importance report.
        
        Returns:
            pd.DataFrame: Feature importance report.
        """
        mi_df = pd.DataFrame.from_dict(self.feature_importances['mutual_info'], orient='index', columns=['MI_score'])
        f_df = pd.DataFrame.from_dict(self.feature_importances['f_test'], orient='index', columns=['F_score'])
        
        importance_df = mi_df.join(f_df)
        importance_df['MI_rank'] = importance_df['MI_score'].rank(ascending=False)
        importance_df['F_rank'] = importance_df['F_score'].rank(ascending=False)
        importance_df['Avg_rank'] = (importance_df['MI_rank'] + importance_df['F_rank']) / 2
        
        return importance_df.sort_values('Avg_rank')

    def save(self, path: str = SAVE_LOAD_CONFIG['feature_engineer_save_path']):
        """
        Save the fitted feature engineer to disk.
        
        Args:
            path (str): Path to save the feature engineer.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = SAVE_LOAD_CONFIG['feature_engineer_save_path']) -> 'AutoFeatureEngineer':
        """
        Load a fitted feature engineer from disk.
        
        Args:
            path (str): Path to load the feature engineer from.
        
        Returns:
            AutoFeatureEngineer: Loaded feature engineer.
        """
        return joblib.load(path)

def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, AutoFeatureEngineer]:
    """
    Engineer features for train and test data using AutoFeatureEngineer.
    
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, AutoFeatureEngineer]: Engineered train and test data, and the fitted feature engineer.
    """
    feature_engineer = AutoFeatureEngineer()
    train_engineered = feature_engineer.fit(train_df).transform(train_df)
    test_engineered = feature_engineer.transform(test_df)
    
    feature_engineer.calculate_feature_importance(train_engineered)
    
    return train_engineered, test_engineered, feature_engineer

if __name__ == "__main__":
    # Example usage
    from src.data_loading import load_and_split_data
    from src.preprocessing import preprocess_data
    
    train_data, test_data = load_and_split_data(DATA_CONFIG['raw_data_path'])
    train_preprocessed, test_preprocessed, _ = preprocess_data(train_data, test_data)
    
    train_engineered, test_engineered, feature_engineer = engineer_features(train_preprocessed, test_preprocessed)
    
    print(f"Engineered train data shape: {train_engineered.shape}")
    print(f"Engineered test data shape: {test_engineered.shape}")
    
    # Print feature importance report
    importance_report = feature_engineer.get_feature_importance_report()
    print("\nTop 10 most important features:")
    print(importance_report.head(10))
    
    # Save the feature engineer
    feature_engineer.save()
    print(f"Feature engineer saved to {SAVE_LOAD_CONFIG['feature_engineer_save_path']}")
