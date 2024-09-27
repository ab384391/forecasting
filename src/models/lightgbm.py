import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from src.models.base import TreeForecaster
from src.config import MODEL_CONFIGS

class LightGBMForecaster(TreeForecaster):
    def __init__(self, config: Dict[str, Any]):
        super().__init__('lightgbm', config)
        self.forecast_horizon = config.get('forecast_horizon', 1)
        self.model = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=config.get('n_estimators', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', -1),  # -1 means no limit
                num_leaves=config.get('num_leaves', 31),
                subsample=config.get('subsample', 1.0),
                colsample_bytree=config.get('colsample_bytree', 1.0),
                random_state=42
            )
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LightGBMForecaster':
        # Reshape y to have multiple outputs if forecast_horizon > 1
        y_reshaped = self._reshape_target(y)
        
        self.model.fit(X, y_reshaped)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        point_predictions = self.predict(X)
        
        # Use quantile regression for prediction intervals
        lower_model = MultiOutputRegressor(
            LGBMRegressor(
                **self.model.estimator.get_params(),
                objective='quantile',
                alpha=alpha/2
            )
        )
        upper_model = MultiOutputRegressor(
            LGBMRegressor(
                **self.model.estimator.get_params(),
                objective='quantile',
                alpha=1-alpha/2
            )
        )
        
        y_reshaped = self._reshape_target(y)
        lower_model.fit(X, y_reshaped)
        upper_model.fit(X, y_reshaped)
        
        lower_bound = lower_model.predict(X)
        upper_bound = upper_model.predict(X)
        
        return point_predictions, lower_bound, upper_bound

    def get_feature_importance(self) -> pd.Series:
        # For multioutput regression, we'll average feature importance across all outputs
        importance = np.mean([estimator.feature_importances_ for estimator in self.model.estimators_], axis=0)
        return pd.Series(importance, index=self.feature_names)

    def set_params(self, params: Dict[str, Any]):
        self.model.estimator.set_params(**params)

    def _reshape_target(self, y: pd.Series) -> np.ndarray:
        if self.forecast_horizon > 1:
            y_reshaped = np.column_stack([y.shift(-i) for i in range(self.forecast_horizon)])
            y_reshaped = y_reshaped[:-self.forecast_horizon]  # Remove last rows with NaN
        else:
            y_reshaped = y.values.reshape(-1, 1)
        return y_reshaped

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> np.ndarray:
        y_reshaped = self._reshape_target(y)
        scores = cross_val_score(self.model, X, y_reshaped, cv=cv, scoring='neg_mean_squared_error')
        return np.sqrt(-scores)  # Return RMSE

def create_lightgbm_forecaster(config: Dict[str, Any] = MODEL_CONFIGS['lightgbm']) -> LightGBMForecaster:
    """
    Create an instance of LightGBMForecaster with the specified configuration.
    
    Args:
        config (Dict[str, Any]): Configuration for the LightGBMForecaster.
    
    Returns:
        LightGBMForecaster: An instance of the LightGBMForecaster.
    """
    return LightGBMForecaster(config)

if __name__ == "__main__":
    # Example usage
    from src.data_loading import load_and_split_data
    from src.preprocessing import preprocess_data
    from src.feature_engineering import engineer_features
    from src.config import DATA_CONFIG, SAVE_LOAD_CONFIG
    
    # Load and preprocess data
    train_data, test_data = load_and_split_data(DATA_CONFIG['raw_data_path'])
    train_preprocessed, test_preprocessed, _ = preprocess_data(train_data, test_data)
    train_engineered, test_engineered, _ = engineer_features(train_preprocessed, test_preprocessed)
    
    # Prepare data for LightGBM
    X_train = train_engineered.drop(columns=[DATA_CONFIG['target_column']])
    y_train = train_engineered[DATA_CONFIG['target_column']]
    X_test = test_engineered.drop(columns=[DATA_CONFIG['target_column']])
    y_test = test_engineered[DATA_CONFIG['target_column']]
    
    # Create and train LightGBM model
    lgbm_forecaster = create_lightgbm_forecaster()
    lgbm_forecaster.fit(X_train, y_train)
    
    # Make predictions
    predictions = lgbm_forecaster.predict(X_test)
    point_predictions, lower_bound, upper_bound = lgbm_forecaster.predict_interval(X_test)
    
    print(f"LightGBM predictions shape: {predictions.shape}")
    print(f"Point predictions shape: {point_predictions.shape}")
    print(f"Lower bound shape: {lower_bound.shape}")
    print(f"Upper bound shape: {upper_bound.shape}")
    
    # Get feature importance
    feature_importance = lgbm_forecaster.get_feature_importance()
    print("\nTop 10 most important features:")
    print(feature_importance.sort_values(ascending=False).head(10))
    
    # Perform cross-validation
    cv_scores = lgbm_forecaster.cross_validate(X_train, y_train)
    print(f"\nCross-validation RMSE scores: {cv_scores}")
    print(f"Mean RMSE: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save the model
    lgbm_forecaster.save()
    print(f"LightGBM model saved to {SAVE_LOAD_CONFIG['model_save_path']}/lightgbm_model.joblib")
