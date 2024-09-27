import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from src.models.base import Forecaster
from src.config import ENSEMBLE_CONFIG, SAVE_LOAD_CONFIG
import joblib

class EnsembleForecaster:
    def __init__(self, models: List[Forecaster], weights: Dict[str, float] = None):
        self.models = models
        self.weights = weights or ENSEMBLE_CONFIG
        self._validate_weights()

    def _validate_weights(self):
        model_names = [model.name for model in self.models]
        for name in self.weights.keys():
            if name not in model_names:
                raise ValueError(f"Weight provided for unknown model: {name}")
        for model in self.models:
            if model.name not in self.weights:
                raise ValueError(f"No weight provided for model: {model.name}")
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Weights must sum to 1")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleForecaster':
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        for model in self.models:
            model_pred = model.predict(X)
            predictions.append(model_pred * self.weights[model.name])
        return np.sum(predictions, axis=0)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        point_predictions = self.predict(X)
        
        lower_bounds = []
        upper_bounds = []
        
        for model in self.models:
            _, lower, upper = model.predict_interval(X, alpha)
            lower_bounds.append(lower * self.weights[model.name])
            upper_bounds.append(upper * self.weights[model.name])
        
        lower_bound = np.sum(lower_bounds, axis=0)
        upper_bound = np.sum(upper_bounds, axis=0)
        
        return point_predictions, lower_bound, upper_bound

    def get_feature_importance(self) -> pd.DataFrame:
        feature_importances = {}
        for model in self.models:
            try:
                model_importance = model.get_feature_importance()
                feature_importances[model.name] = model_importance * self.weights[model.name]
            except NotImplementedError:
                print(f"Feature importance not available for {model.name}")
        
        return pd.DataFrame(feature_importances)

    def save(self, path: str = SAVE_LOAD_CONFIG['model_save_path'] + '/ensemble_model.joblib'):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = SAVE_LOAD_CONFIG['model_save_path'] + '/ensemble_model.joblib') -> 'EnsembleForecaster':
        return joblib.load(path)

def create_ensemble_forecaster(models: List[Forecaster], weights: Dict[str, float] = None) -> EnsembleForecaster:
    """
    Create an instance of EnsembleForecaster with the specified models and weights.
    
    Args:
        models (List[Forecaster]): List of forecaster models to ensemble.
        weights (Dict[str, float], optional): Dictionary of model weights. If None, use default weights from config.
    
    Returns:
        EnsembleForecaster: An instance of the EnsembleForecaster.
    """
    return EnsembleForecaster(models, weights)

if __name__ == "__main__":
    # Example usage
    from src.data_loading import load_and_split_data
    from src.preprocessing import preprocess_data
    from src.feature_engineering import engineer_features
    from src.models.nbeats import create_nbeats_forecaster
    from src.models.lstm import create_lstm_forecaster
    from src.models.gradient_boost import create_gradient_boost_forecaster
    from src.models.lightgbm import create_lightgbm_forecaster
    from src.config import DATA_CONFIG
    
    # Load and preprocess data
    train_data, test_data = load_and_split_data(DATA_CONFIG['raw_data_path'])
    train_preprocessed, test_preprocessed, _ = preprocess_data(train_data, test_data)
    train_engineered, test_engineered, _ = engineer_features(train_preprocessed, test_preprocessed)
    
    # Prepare data
    X_train = train_engineered.drop(columns=[DATA_CONFIG['target_column']])
    y_train = train_engineered[DATA_CONFIG['target_column']]
    X_test = test_engineered.drop(columns=[DATA_CONFIG['target_column']])
    y_test = test_engineered[DATA_CONFIG['target_column']]
    
    # Create individual models
    models = [
        create_nbeats_forecaster(),
        create_lstm_forecaster(),
        create_gradient_boost_forecaster(),
        create_lightgbm_forecaster()
    ]
    
    # Create and train ensemble
    ensemble = create_ensemble_forecaster(models)
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    predictions = ensemble.predict(X_test)
    point_predictions, lower_bound, upper_bound = ensemble.predict_interval(X_test)
    
    print(f"Ensemble predictions shape: {predictions.shape}")
    print(f"Point predictions shape: {point_predictions.shape}")
    print(f"Lower bound shape: {lower_bound.shape}")
    print(f"Upper bound shape: {upper_bound.shape}")
    
    # Get feature importance
    feature_importance = ensemble.get_feature_importance()
    print("\nEnsemble feature importance:")
    print(feature_importance.head())
    
    # Save the ensemble
    ensemble.save()
    print(f"Ensemble model saved to {SAVE_LOAD_CONFIG['model_save_path']}/ensemble_model.joblib")
