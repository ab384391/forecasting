import numpy as np
import pandas as pd
from typing import Union, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_mape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true (Union[np.ndarray, pd.Series]): True values.
        y_pred (Union[np.ndarray, pd.Series]): Predicted values.

    Returns:
        float: MAPE value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_smape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true (Union[np.ndarray, pd.Series]): True values.
        y_pred (Union[np.ndarray, pd.Series]): Predicted values.

    Returns:
        float: SMAPE value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def calculate_rmse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    Args:
        y_true (Union[np.ndarray, pd.Series]): True values.
        y_pred (Union[np.ndarray, pd.Series]): Predicted values.

    Returns:
        float: RMSE value.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Args:
        y_true (Union[np.ndarray, pd.Series]): True values.
        y_pred (Union[np.ndarray, pd.Series]): Predicted values.

    Returns:
        float: MAE value.
    """
    return mean_absolute_error(y_true, y_pred)

def calculate_mase(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series], y_train: Union[np.ndarray, pd.Series], seasonality: int = 1) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).

    Args:
        y_true (Union[np.ndarray, pd.Series]): True values.
        y_pred (Union[np.ndarray, pd.Series]): Predicted values.
        y_train (Union[np.ndarray, pd.Series]): Training data used to scale the error.
        seasonality (int): The seasonality of the time series (default is 1 for non-seasonal data).

    Returns:
        float: MASE value.
    """
    y_true, y_pred, y_train = np.array(y_true), np.array(y_pred), np.array(y_train)
    n = y_train.shape[0]
    d = np.abs(np.diff(y_train, seasonality)).sum() / (n - seasonality)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

def calculate_wape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Weighted Absolute Percentage Error (WAPE).

    Args:
        y_true (Union[np.ndarray, pd.Series]): True values.
        y_pred (Union[np.ndarray, pd.Series]): Predicted values.

    Returns:
        float: WAPE value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def calculate_r_squared(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate R-squared (Coefficient of Determination).

    Args:
        y_true (Union[np.ndarray, pd.Series]): True values.
        y_pred (Union[np.ndarray, pd.Series]): Predicted values.

    Returns:
        float: R-squared value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def calculate_directional_accuracy(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Directional Accuracy.

    Args:
        y_true (Union[np.ndarray, pd.Series]): True values.
        y_pred (Union[np.ndarray, pd.Series]): Predicted values.

    Returns:
        float: Directional Accuracy value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    return np.mean((y_true_diff * y_pred_diff) > 0)

def calculate_all_metrics(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series], y_train: Union[np.ndarray, pd.Series] = None, seasonality: int = 1) -> dict:
    """
    Calculate all implemented metrics.

    Args:
        y_true (Union[np.ndarray, pd.Series]): True values.
        y_pred (Union[np.ndarray, pd.Series]): Predicted values.
        y_train (Union[np.ndarray, pd.Series], optional): Training data for MASE calculation.
        seasonality (int): Seasonality for MASE calculation.

    Returns:
        dict: Dictionary containing all calculated metrics.
    """
    metrics = {
        'MAPE': calculate_mape(y_true, y_pred),
        'SMAPE': calculate_smape(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'WAPE': calculate_wape(y_true, y_pred),
        'R-squared': calculate_r_squared(y_true, y_pred),
        'Directional Accuracy': calculate_directional_accuracy(y_true, y_pred)
    }
    
    if y_train is not None:
        metrics['MASE'] = calculate_mase(y_true, y_pred, y_train, seasonality)
    
    return metrics

# Usage example:
# y_true = np.array([1, 2, 3, 4, 5])
# y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
# y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# 
# all_metrics = calculate_all_metrics(y_true, y_pred, y_train, seasonality=1)
# for metric, value in all_metrics.items():
#     print(f"{metric}: {value}")