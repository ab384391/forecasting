from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional

class BaseForecaster(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the forecaster to the training data.

        Args:
            X (pd.DataFrame): Training data features.
            y (pd.Series): Training data target values.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate point forecasts for the given features.

        Args:
            X (pd.DataFrame): Features for which to generate forecasts.

        Returns:
            np.ndarray: Predicted values.
        """
        pass

    @abstractmethod
    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate interval forecasts for the given features.

        Args:
            X (pd.DataFrame): Features for which to generate forecasts.
            alpha (float): Significance level for the prediction interval (default: 0.05 for 95% interval).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Predicted values, lower bounds, and upper bounds.
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance or coefficients from the model.

        Returns:
            pd.DataFrame: DataFrame with feature names and their importance/coefficients.
        """
        pass

    @abstractmethod
    def clone(self) -> 'BaseForecaster':
        """
        Create a clone of the current model instance.

        Returns:
            BaseForecaster: A new instance of the model with the same parameters.
        """
        pass

    def fit_predict(self, X: pd.DataFrame, y: pd.Series, X_forecast: pd.DataFrame) -> np.ndarray:
        """
        Fit the model and immediately generate forecasts.

        Args:
            X (pd.DataFrame): Training data features.
            y (pd.Series): Training data target values.
            X_forecast (pd.DataFrame): Features for which to generate forecasts.

        Returns:
            np.ndarray: Predicted values.
        """
        self.fit(X, y)
        return self.predict(X_forecast)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, metrics: Optional[list] = None) -> dict:
        """
        Evaluate the model on the given data using specified metrics.

        Args:
            X (pd.DataFrame): Features for evaluation.
            y (pd.Series): True target values.
            metrics (Optional[list]): List of metric functions to use. If None, uses default metrics.

        Returns:
            dict: Dictionary of metric names and their values.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from ..evaluation.metrics import calculate_mape, calculate_smape

        if metrics is None:
            metrics = [
                ('mse', mean_squared_error),
                ('rmse', lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))),
                ('mae', mean_absolute_error),
                ('mape', calculate_mape),
                ('smape', calculate_smape)
            ]

        y_pred = self.predict(X)
        return {name: metric(y, y_pred) for name, metric in metrics}

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path (str): Path to save the model.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseForecaster':
        """
        Load the model from a file.

        Args:
            path (str): Path to load the model from.

        Returns:
            BaseForecaster: Loaded model instance.
        """
        pass

# Example of how a concrete forecaster might be implemented:
# 
# class SimpleForecaster(BaseForecaster):
#     def __init__(self, model):
#         self.model = model
# 
#     def fit(self, X, y):
#         self.model.fit(X, y)
# 
#     def predict(self, X):
#         return self.model.predict(X)
# 
#     def predict_interval(self, X, alpha=0.05):
#         # Implementation depends on the specific model
#         pass
# 
#     def get_feature_importance(self):
#         # Implementation depends on the specific model
#         pass
# 
#     def clone(self):
#         return SimpleForecaster(clone(self.model))
# 
#     def save(self, path):
#         joblib.dump(self.model, path)
# 
#     @classmethod
#     def load(cls, path):
#         model = joblib.load(path)
#         return cls(model)