from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import joblib
from src.config import SAVE_LOAD_CONFIG

class Forecaster(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.model = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Forecaster':
        """
        Fit the forecasting model.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
        
        Returns:
            Forecaster: Fitted forecaster.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make point predictions.
        
        Args:
            X (pd.DataFrame): Input features.
        
        Returns:
            np.ndarray: Point predictions.
        """
        pass

    @abstractmethod
    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            X (pd.DataFrame): Input features.
            alpha (float): Significance level for confidence intervals.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Point predictions, lower bounds, and upper bounds.
        """
        pass

    def save(self, path: str = None):
        """
        Save the fitted forecaster to disk.
        
        Args:
            path (str, optional): Path to save the forecaster. If None, use default path.
        """
        if path is None:
            path = f"{SAVE_LOAD_CONFIG['model_save_path']}/{self.name}_model.joblib"
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'Forecaster':
        """
        Load a fitted forecaster from disk.
        
        Args:
            path (str): Path to load the forecaster from.
        
        Returns:
            Forecaster: Loaded forecaster.
        """
        return joblib.load(path)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance if the model supports it.
        
        Returns:
            pd.Series: Feature importance scores.
        """
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(self.model.feature_importances_, index=self.feature_names)
        elif hasattr(self.model, 'coef_'):
            return pd.Series(np.abs(self.model.coef_), index=self.feature_names)
        else:
            raise NotImplementedError("Feature importance not available for this model.")

    def set_feature_names(self, feature_names: list):
        """
        Set feature names for the model.
        
        Args:
            feature_names (list): List of feature names.
        """
        self.feature_names = feature_names

class DeepForecaster(Forecaster):
    @abstractmethod
    def create_dataset(self, X: pd.DataFrame, y: pd.Series = None) -> Any:
        """
        Create a dataset suitable for deep learning models.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series, optional): Target variable.
        
        Returns:
            Any: Dataset object suitable for the specific deep learning framework.
        """
        pass

    @abstractmethod
    def compile_model(self):
        """
        Compile the deep learning model.
        """
        pass

class TreeForecaster(Forecaster):
    @abstractmethod
    def set_params(self, params: Dict[str, Any]):
        """
        Set the parameters of the tree-based model.
        
        Args:
            params (Dict[str, Any]): Dictionary of parameters.
        """
        pass

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance for tree-based models.
        
        Returns:
            pd.Series: Feature importance scores.
        """
        return super().get_feature_importance()

def create_forecaster(model_type: str, config: Dict[str, Any]) -> Forecaster:
    """
    Factory function to create a forecaster based on the model type.
    
    Args:
        model_type (str): Type of the forecaster to create.
        config (Dict[str, Any]): Configuration for the forecaster.
    
    Returns:
        Forecaster: An instance of the specified forecaster.
    """
    if model_type == 'nbeats':
        from .nbeats import NBEATSForecaster
        return NBEATSForecaster(config)
    elif model_type == 'tcn':
        from .tcn import TCNForecaster
        return TCNForecaster(config)
    elif model_type == 'transformer':
        from .transformer import TransformerForecaster
        return TransformerForecaster(config)
    elif model_type == 'lstm':
        from .lstm import LSTMForecaster
        return LSTMForecaster(config)
    elif model_type == 'gradient_boost':
        from .gradient_boost import GradientBoostForecaster
        return GradientBoostForecaster(config)
    elif model_type == 'lightgbm':
        from .lightgbm import LightGBMForecaster
        return LightGBMForecaster(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
