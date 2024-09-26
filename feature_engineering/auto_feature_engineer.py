import pandas as pd
import numpy as np
from typing import List, Dict, Any
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import acf
import holidays
from joblib import Parallel, delayed

class AutoFeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.holidays = holidays.US()  # Change this for different countries

    def engineer_features(self, df: pd.DataFrame, date_column: str, target_column: str) -> pd.DataFrame:
        """
        Perform automated feature engineering based on the configuration.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            date_column (str): Name of the date column.
            target_column (str): Name of the target column.
        
        Returns:
            pd.DataFrame: Dataframe with engineered features.
        """
        if self.config.get('time_features', False):
            df = self._create_time_features(df, date_column)
        
        if self.config.get('lag_features', False):
            df = self._create_lag_features(df, target_column)
        
        if self.config.get('interaction_features', False):
            df = self._create_interaction_features(df)
        
        if self.config.get('transformation_features', False):
            df = self._create_transformation_features(df)
        
        if self.config.get('differencing_features', False):
            df = self._create_differencing_features(df, target_column)
        
        if self.config.get('fourier_features', False):
            df = self._create_fourier_features(df, date_column)
        
        if self.config.get('target_encoding', False):
            df = self._create_target_encoding(df, target_column)
        
        return df

    def _create_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        df['is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        df['is_holiday'] = df[date_column].isin(self.holidays).astype(int)
        
        # Cyclical encoding for time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df

    def _create_lag_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        # Determine optimal lags using autocorrelation
        series = df[target_column].dropna()
        lag_acf = acf(series, nlags=30)
        significant_lags = [i for i, corr in enumerate(lag_acf) if abs(corr) > 1.96 / np.sqrt(len(series))]
        
        for lag in significant_lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Create rolling statistics
        for window in [7, 14, 30]:
            df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
            df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
        
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create polynomial features
        for col in numeric_cols:
            df[f'{col}_squared'] = df[col] ** 2
        
        # Create interactions between top correlated features
        correlations = df[numeric_cols].corr().abs()
        top_corr = correlations.unstack().sort_values(kind="quicksort").drop_duplicates().tail(5)
        for (col1, col2), _ in top_corr.items():
            df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        
        return df

    def _create_transformation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df[f'{col}_log'] = np.log1p(df[col] - df[col].min() + 1)
            df[f'{col}_sqrt'] = np.sqrt(df[col] - df[col].min())
            
            # Box-Cox transformation
            df[f'{col}_boxcox'], _ = stats.boxcox(df[col] - df[col].min() + 1)
            
            # Yeo-Johnson transformation
            df[f'{col}_yeojohnson'], _ = stats.yeojohnson(df[col])
        
        return df

    def _create_differencing_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        df[f'{target_column}_diff'] = df[target_column].diff()
        df[f'{target_column}_diff_7'] = df[target_column].diff(7)  # Weekly differencing
        df[f'{target_column}_diff_30'] = df[target_column].diff(30)  # Monthly differencing
        
        return df

    def _create_fourier_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        df['day_of_year'] = df[date_column].dt.dayofyear
        for period in [365.25, 365.25/2, 365.25/4]:  # Yearly, half-yearly, quarterly
            df[f'fourier_sin_{period:.0f}'] = np.sin(2 * np.pi * df['day_of_year'] / period)
            df[f'fourier_cos_{period:.0f}'] = np.cos(2 * np.pi * df['day_of_year'] / period)
        
        return df

    def _create_target_encoding(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            global_mean = df[target_column].mean()
            agg = df.groupby(col)[target_column].agg(['mean', 'count'])
            smoothing = 1 / (1 + np.exp(-(agg['count'] - 10) / 10))
            agg['enc'] = smoothing * agg['mean'] + (1 - smoothing) * global_mean
            df[f'{col}_target_enc'] = df[col].map(agg['enc'])
        
        return df

    def select_features(self, df: pd.DataFrame, target_column: str, n_features: int = 50) -> List[str]:
        """
        Select top features based on mutual information and recursive feature elimination.
        
        Args:
            df (pd.DataFrame): Input dataframe with engineered features.
            target_column (str): Name of the target column.
            n_features (int): Number of features to select.
        
        Returns:
            List[str]: List of selected feature names.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Mutual Information
        mi_scores = mutual_info_regression(X, y)
        mi_features = X.columns[np.argsort(mi_scores)[-n_features:]]
        
        # Recursive Feature Elimination
        rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=n_features)
        rfe.fit(X, y)
        rfe_features = X.columns[rfe.support_]
        
        # Combine and deduplicate
        selected_features = list(set(mi_features) | set(rfe_features))
        
        return selected_features

    def generate_feature_importance_report(self, df: pd.DataFrame, target_column: str, selected_features: List[str]) -> pd.DataFrame:
        """
        Generate a feature importance report.
        
        Args:
            df (pd.DataFrame): Input dataframe with engineered features.
            target_column (str): Name of the target column.
            selected_features (List[str]): List of selected feature names.
        
        Returns:
            pd.DataFrame: DataFrame with feature importances.
        """
        X = df[selected_features]
        y = df[target_column]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importances = pd.DataFrame({
            'feature': selected_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importances

# Usage example:
# config = {
#     'time_features': True,
#     'lag_features': True,
#     'interaction_features': True,
#     'transformation_features': True,
#     'differencing_features': True,
#     'fourier_features': True,
#     'target_encoding': True
# }
# auto_fe = AutoFeatureEngineer(config)
# df_engineered = auto_fe.engineer_features(df, 'date_column', 'target_column')
# selected_features = auto_fe.select_features(df_engineered, 'target_column')
# importance_report = auto_fe.generate_feature_importance_report(df_engineered, 'target_column', selected_features)