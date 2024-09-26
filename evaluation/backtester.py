import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from ..models.base_forecaster import BaseForecaster
from .metrics import calculate_mape, calculate_smape
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

class FastBacktester:
    def __init__(self, model: BaseForecaster, initial_train_size: int, step_size: int, horizon: int):
        """
        Initialize the FastBacktester.

        Args:
            model (BaseForecaster): The forecasting model to evaluate.
            initial_train_size (int): The size of the initial training set.
            step_size (int): The number of steps to move forward in each iteration.
            horizon (int): The forecasting horizon.
        """
        self.model = model
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.horizon = horizon
        self.metrics = {
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error,
            'mape': calculate_mape,
            'smape': calculate_smape
        }

    def backtest(self, X: pd.DataFrame, y: pd.Series, feature_engineering_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform walk-forward validation.

        Args:
            X (pd.DataFrame): The feature dataframe.
            y (pd.Series): The target series.
            feature_engineering_config (Dict[str, Any], optional): Configuration for feature engineering.

        Returns:
            Dict[str, Any]: Backtesting results including predictions and performance metrics.
        """
        results = []
        feature_importance = []
        total_samples = len(y)

        for i in range(self.initial_train_size, total_samples - self.horizon + 1, self.step_size):
            train_end = i
            test_end = min(i + self.horizon, total_samples)

            X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
            X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

            if feature_engineering_config:
                X_train, X_test = self._apply_feature_engineering(X_train, X_test, feature_engineering_config)

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            result = {
                'start_date': X.index[train_end],
                'end_date': X.index[test_end - 1],
                'y_true': y_test,
                'y_pred': y_pred
            }

            for metric_name, metric_func in self.metrics.items():
                result[metric_name] = metric_func(y_test, y_pred)

            results.append(result)

            # Calculate feature importance
            importance = self.model.get_feature_importance()
            importance['timestamp'] = X.index[train_end]
            feature_importance.append(importance)

        return {
            'results': pd.DataFrame(results),
            'feature_importance': pd.concat(feature_importance)
        }

    def _apply_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply feature engineering to the training and test sets.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.
            config (Dict[str, Any]): Feature engineering configuration.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Engineered training and test features.
        """
        from ..feature_engineering.auto_feature_engineer import AutoFeatureEngineer

        auto_fe = AutoFeatureEngineer(config)
        X_train_engineered = auto_fe.engineer_features(X_train)
        X_test_engineered = auto_fe.engineer_features(X_test)

        return X_train_engineered, X_test_engineered

    def plot_backtesting_results(self, results: pd.DataFrame):
        """
        Plot the backtesting results.

        Args:
            results (pd.DataFrame): The results dataframe from the backtest method.
        """
        plt.figure(figsize=(15, 10))
        plt.plot(results['end_date'], results['y_true'], label='Actual', color='blue')
        plt.plot(results['end_date'], results['y_pred'], label='Predicted', color='red')
        plt.fill_between(results['end_date'], 
                         results['y_pred'] - results['rmse'], 
                         results['y_pred'] + results['rmse'], 
                         color='red', alpha=0.2)
        plt.title('Backtesting Results')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def plot_metric_over_time(self, results: pd.DataFrame, metric: str):
        """
        Plot a specific metric over time.

        Args:
            results (pd.DataFrame): The results dataframe from the backtest method.
            metric (str): The name of the metric to plot.
        """
        plt.figure(figsize=(15, 5))
        plt.plot(results['end_date'], results[metric])
        plt.title(f'{metric.upper()} Over Time')
        plt.xlabel('Date')
        plt.ylabel(metric.upper())
        plt.show()

    def plot_feature_importance_over_time(self, feature_importance: pd.DataFrame, top_n: int = 10):
        """
        Plot feature importance over time.

        Args:
            feature_importance (pd.DataFrame): The feature importance dataframe from the backtest method.
            top_n (int): The number of top features to display.
        """
        top_features = feature_importance.groupby('feature')['importance'].mean().nlargest(top_n).index

        plt.figure(figsize=(15, 10))
        for feature in top_features:
            feature_data = feature_importance[feature_importance['feature'] == feature]
            plt.plot(feature_data['timestamp'], feature_data['importance'], label=feature)

        plt.title(f'Top {top_n} Feature Importance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Importance')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_confidence_interval_coverage(self, results: pd.DataFrame):
        """
        Plot the confidence interval coverage.

        Args:
            results (pd.DataFrame): The results dataframe from the backtest method.
        """
        coverage = (results['y_true'] >= results['y_pred'] - results['rmse']) & \
                   (results['y_true'] <= results['y_pred'] + results['rmse'])
        coverage_rate = coverage.mean()

        plt.figure(figsize=(15, 5))
        plt.plot(results['end_date'], coverage)
        plt.axhline(y=coverage_rate, color='r', linestyle='--', label=f'Average Coverage: {coverage_rate:.2f}')
        plt.title('Confidence Interval Coverage')
        plt.xlabel('Date')
        plt.ylabel('Coverage (1 = within CI, 0 = outside CI)')
        plt.legend()
        plt.show()

# Usage example:
# from models.tree_based_models import GradientBoostForecaster
#
# model = GradientBoostForecaster(n_estimators=100, learning_rate=0.1)
# backtester = FastBacktester(model, initial_train_size=1000, step_size=30, horizon=30)
# feature_engineering_config = {
#     'lag_features': True,
#     'rolling_features': True,
#     'date_features': True
# }
# results = backtester.backtest(X, y, feature_engineering_config)
#
# backtester.plot_backtesting_results(results['results'])
# backtester.plot_metric_over_time(results['results'], 'mape')
# backtester.plot_feature_importance_over_time(results['feature_importance'])
# backtester.plot_confidence_interval_coverage(results['results'])