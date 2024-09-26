import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from .base_forecaster import BaseForecaster
import joblib

class TreeBasedForecaster(BaseForecaster):
    def __init__(self, model_type: str, **model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = self._create_model()
        self.feature_scaler = StandardScaler()

    def _create_model(self):
        if self.model_type == 'XGBoost':
            return XGBRegressor(**self.model_params)
        elif self.model_type == 'LightGBM':
            return LGBMRegressor(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = self.feature_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.feature_scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_scaled = self.feature_scaler.transform(X)
        
        # Quantile regression for prediction intervals
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        if self.model_type == 'XGBoost':
            lower_model = XGBRegressor(**self.model_params, objective='reg:quantile', quantile_alpha=lower_quantile)
            upper_model = XGBRegressor(**self.model_params, objective='reg:quantile', quantile_alpha=upper_quantile)
        elif self.model_type == 'LightGBM':
            lower_model = LGBMRegressor(**self.model_params, objective='quantile', alpha=lower_quantile)
            upper_model = LGBMRegressor(**self.model_params, objective='quantile', alpha=upper_quantile)

        lower_model.fit(X_scaled, self.model.y_train_)
        upper_model.fit(X_scaled, self.model.y_train_)

        y_pred = self.predict(X)
        y_lower = lower_model.predict(X_scaled)
        y_upper = upper_model.predict(X_scaled)

        return y_pred, y_lower, y_upper

    def get_feature_importance(self) -> pd.DataFrame:
        feature_importance = self.model.feature_importances_
        feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
        return pd.DataFrame({'feature': feature_names, 'importance': feature_importance}).sort_values('importance', ascending=False)

    def clone(self) -> 'TreeBasedForecaster':
        return TreeBasedForecaster(self.model_type, **self.model_params)

    def save(self, path: str) -> None:
        joblib.dump({
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'model_type': self.model_type,
            'model_params': self.model_params
        }, path)

    @classmethod
    def load(cls, path: str) -> 'TreeBasedForecaster':
        data = joblib.load(path)
        forecaster = cls(data['model_type'], **data['model_params'])
        forecaster.model = data['model']
        forecaster.feature_scaler = data['feature_scaler']
        return forecaster

class GradientBoostForecaster(TreeBasedForecaster):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, **kwargs):
        super().__init__('XGBoost', n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, **kwargs)

class LightGBMForecaster(TreeBasedForecaster):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = -1, **kwargs):
        super().__init__('LightGBM', n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, **kwargs)

# Additional utility functions for tree-based models

def calculate_shap_values(model: TreeBasedForecaster, X: pd.DataFrame) -> np.ndarray:
    """
    Calculate SHAP (SHapley Additive exPlanations) values for the given model and data.
    
    Args:
        model (TreeBasedForecaster): The trained tree-based model.
        X (pd.DataFrame): The input data to explain.
    
    Returns:
        np.ndarray: SHAP values for each feature and instance.
    """
    import shap
    
    X_scaled = model.feature_scaler.transform(X)
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_scaled)
    
    return shap_values

def plot_feature_importance(model: TreeBasedForecaster, X: pd.DataFrame, top_n: int = 10) -> None:
    """
    Plot feature importance for the given model.
    
    Args:
        model (TreeBasedForecaster): The trained tree-based model.
        X (pd.DataFrame): The input data used for importance calculation.
        top_n (int): Number of top features to display.
    """
    import matplotlib.pyplot as plt
    
    feature_importance = model.get_feature_importance()
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

def plot_partial_dependence(model: TreeBasedForecaster, X: pd.DataFrame, features: List[str], n_cols: int = 3) -> None:
    """
    Plot partial dependence for specified features.
    
    Args:
        model (TreeBasedForecaster): The trained tree-based model.
        X (pd.DataFrame): The input data used for partial dependence calculation.
        features (List[str]): List of feature names to plot.
        n_cols (int): Number of columns in the plot grid.
    """
    from sklearn.inspection import partial_dependence, plot_partial_dependence
    import matplotlib.pyplot as plt
    
    n_features = len(features)
    n_rows = (n_features - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    for i, feature in enumerate(features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        plot_partial_dependence(model.model, model.feature_scaler.transform(X), 
                                [feature], ax=ax, feature_names=X.columns)
        ax.set_title(feature)
    
    plt.tight_layout()
    plt.show()

# Usage example:
# gb_forecaster = GradientBoostForecaster(n_estimators=200, learning_rate=0.05, max_depth=5)
# gb_forecaster.fit(X_train, y_train)
# predictions = gb_forecaster.predict(X_test)
# predictions, lower_bound, upper_bound = gb_forecaster.predict_interval(X_test)
# feature_importance = gb_forecaster.get_feature_importance()
# plot_feature_importance(gb_forecaster, X_test)
# plot_partial_dependence(gb_forecaster, X_test, ['feature_1', 'feature_2', 'feature_3'])
# shap_values = calculate_shap_values(gb_forecaster, X_test)