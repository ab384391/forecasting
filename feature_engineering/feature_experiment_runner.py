import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import Parallel, delayed
from auto_feature_engineer import AutoFeatureEngineer
from ..models.base_forecaster import BaseForecaster
from ..evaluation.metrics import calculate_mape, calculate_smape

class FeatureExperimentRunner:
    def __init__(self, base_model: BaseForecaster, configs: List[Dict[str, Any]], n_splits: int = 5):
        """
        Initialize the FeatureExperimentRunner.

        Args:
            base_model (BaseForecaster): The base forecasting model to use for experiments.
            configs (List[Dict[str, Any]]): List of feature engineering configurations to test.
            n_splits (int): Number of splits for time series cross-validation.
        """
        self.base_model = base_model
        self.configs = configs
        self.n_splits = n_splits

    def run_experiments(self, df: pd.DataFrame, date_column: str, target_column: str) -> pd.DataFrame:
        """
        Run feature engineering experiments.

        Args:
            df (pd.DataFrame): Input dataframe.
            date_column (str): Name of the date column.
            target_column (str): Name of the target column.

        Returns:
            pd.DataFrame: DataFrame with experiment results.
        """
        results = Parallel(n_jobs=-1)(
            delayed(self._run_single_experiment)(df, date_column, target_column, config)
            for config in self.configs
        )
        return pd.DataFrame(results)

    def _run_single_experiment(self, df: pd.DataFrame, date_column: str, target_column: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single feature engineering experiment.

        Args:
            df (pd.DataFrame): Input dataframe.
            date_column (str): Name of the date column.
            target_column (str): Name of the target column.
            config (Dict[str, Any]): Feature engineering configuration.

        Returns:
            Dict[str, Any]: Experiment results.
        """
        auto_fe = AutoFeatureEngineer(config)
        df_engineered = auto_fe.engineer_features(df, date_column, target_column)
        selected_features = auto_fe.select_features(df_engineered, target_column)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = []

        for train_index, test_index in tscv.split(df_engineered):
            X_train, X_test = df_engineered.iloc[train_index][selected_features], df_engineered.iloc[test_index][selected_features]
            y_train, y_test = df_engineered.iloc[train_index][target_column], df_engineered.iloc[test_index][target_column]

            model = self.base_model.clone()  # Create a fresh instance of the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            scores = self._calculate_metrics(y_test, y_pred)
            cv_scores.append(scores)

        avg_scores = {metric: np.mean([score[metric] for score in cv_scores]) for metric in cv_scores[0]}
        feature_importance = auto_fe.generate_feature_importance_report(df_engineered, target_column, selected_features)

        return {
            'config': config,
            'n_features': len(selected_features),
            'top_features': feature_importance['feature'].tolist()[:10],  # Top 10 features
            **avg_scores
        }

    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Calculate various performance metrics.

        Args:
            y_true (pd.Series): True values.
            y_pred (pd.Series): Predicted values.

        Returns:
            Dict[str, float]: Dictionary of metric names and values.
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': calculate_mape(y_true, y_pred),
            'smape': calculate_smape(y_true, y_pred)
        }

    def visualize_results(self, results: pd.DataFrame) -> None:
        """
        Visualize experiment results.

        Args:
            results (pd.DataFrame): DataFrame with experiment results.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=results, x='n_features', y='rmse', hue='config', palette='deep')
        plt.title('Number of Features vs RMSE')
        plt.xlabel('Number of Features')
        plt.ylabel('RMSE')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results.melt(id_vars=['config'], value_vars=['mse', 'rmse', 'mae', 'mape', 'smape']), 
                    x='variable', y='value', hue='config')
        plt.title('Performance Metrics Across Configurations')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.yscale('log')
        plt.show()

    def get_best_config(self, results: pd.DataFrame, metric: str = 'rmse') -> Dict[str, Any]:
        """
        Get the best performing configuration based on a specific metric.

        Args:
            results (pd.DataFrame): DataFrame with experiment results.
            metric (str): Metric to use for selecting the best configuration.

        Returns:
            Dict[str, Any]: Best performing configuration.
        """
        best_row = results.loc[results[metric].idxmin()]
        return best_row['config']

# Usage example:
# base_model = SomeForecasterModel()
# configs = [
#     {'time_features': True, 'lag_features': True, 'interaction_features': False},
#     {'time_features': True, 'lag_features': True, 'interaction_features': True},
#     {'time_features': True, 'lag_features': False, 'interaction_features': True},
# ]
# runner = FeatureExperimentRunner(base_model, configs)
# results = runner.run_experiments(df, 'date_column', 'target_column')
# runner.visualize_results(results)
# best_config = runner.get_best_config(results)