import optuna
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Union
from sklearn.model_selection import TimeSeriesSplit
from ..models.base_forecaster import BaseForecaster
from ..evaluation.metrics import calculate_mape, calculate_smape

class OptunaOptimizer:
    def __init__(self, model_class: type, param_space: Dict[str, Any], n_trials: int = 100, n_splits: int = 5, metric: str = 'mape'):
        """
        Initialize the OptunaOptimizer.

        Args:
            model_class (type): The forecaster class to optimize.
            param_space (Dict[str, Any]): The hyperparameter space to search.
            n_trials (int): Number of trials for optimization.
            n_splits (int): Number of splits for time series cross-validation.
            metric (str): The metric to optimize ('mape', 'smape', 'rmse', or 'mae').
        """
        self.model_class = model_class
        self.param_space = param_space
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.metric = metric
        self.study = None

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Run the optimization process.

        Args:
            X (pd.DataFrame): The feature dataframe.
            y (pd.Series): The target series.

        Returns:
            Dict[str, Any]: The best hyperparameters found.
        """
        def objective(trial):
            params = self._sample_params(trial)
            model = self.model_class(**params)
            score = self._cross_validate(model, X, y)
            return score

        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction='minimize', sampler=sampler)
        self.study.optimize(objective, n_trials=self.n_trials)

        return self.study.best_params

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters from the defined space.

        Args:
            trial (optuna.Trial): The current trial.

        Returns:
            Dict[str, Any]: The sampled hyperparameters.
        """
        params = {}
        for param_name, param_config in self.param_space.items():
            if param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'], log=param_config.get('log', False))
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config.get('log', False))
        return params

    def _cross_validate(self, model: BaseForecaster, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Perform time series cross-validation.

        Args:
            model (BaseForecaster): The forecaster to evaluate.
            X (pd.DataFrame): The feature dataframe.
            y (pd.Series): The target series.

        Returns:
            float: The mean score across all folds.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            if self.metric == 'mape':
                score = calculate_mape(y_val, y_pred)
            elif self.metric == 'smape':
                score = calculate_smape(y_val, y_pred)
            elif self.metric == 'rmse':
                score = np.sqrt(np.mean((y_val - y_pred)**2))
            elif self.metric == 'mae':
                score = np.mean(np.abs(y_val - y_pred))
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

            scores.append(score)

        return np.mean(scores)

    def plot_optimization_history(self):
        """
        Plot the optimization history.
        """
        if self.study is None:
            raise ValueError("No study has been conducted yet. Run optimize() first.")
        
        optuna.visualization.plot_optimization_history(self.study)

    def plot_param_importances(self):
        """
        Plot the importance of hyperparameters.
        """
        if self.study is None:
            raise ValueError("No study has been conducted yet. Run optimize() first.")
        
        optuna.visualization.plot_param_importances(self.study)

    def plot_slice(self):
        """
        Plot the slice plot for hyperparameters.
        """
        if self.study is None:
            raise ValueError("No study has been conducted yet. Run optimize() first.")
        
        optuna.visualization.plot_slice(self.study)

def create_param_space(model_type: str) -> Dict[str, Dict[str, Union[str, list, float, int, bool]]]:
    """
    Create a parameter space for a given model type.

    Args:
        model_type (str): The type of model ('LSTM', 'TCN', 'Transformer', 'XGBoost', or 'LightGBM').

    Returns:
        Dict[str, Dict[str, Union[str, list, float, int, bool]]]: The parameter space.
    """
    if model_type in ['LSTM', 'TCN', 'Transformer']:
        return {
            'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-1, 'log': True},
            'num_layers': {'type': 'int', 'low': 1, 'high': 5},
            'hidden_size': {'type': 'int', 'low': 32, 'high': 256},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5}
        }
    elif model_type == 'XGBoost':
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'learning_rate': {'type': 'float', 'low': 1e-3, 'high': 1.0, 'log': True},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
            'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0}
        }
    elif model_type == 'LightGBM':
        return {
            'num_leaves': {'type': 'int', 'low': 20, 'high': 3000},
            'learning_rate': {'type': 'float', 'low': 1e-3, 'high': 1.0, 'log': True},
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'min_child_samples': {'type': 'int', 'low': 1, 'high': 50},
            'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0}
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Usage example:
# from models.deep_learning_models import DeepLearningForecaster
# from models.tree_based_models import GradientBoostForecaster, LightGBMForecaster
#
# # For deep learning models
# dl_param_space = create_param_space('LSTM')
# dl_optimizer = OptunaOptimizer(DeepLearningForecaster, dl_param_space, n_trials=50)
# best_dl_params = dl_optimizer.optimize(X, y)
#
# # For tree-based models
# xgb_param_space = create_param_space('XGBoost')
# xgb_optimizer = OptunaOptimizer(GradientBoostForecaster, xgb_param_space, n_trials=100)
# best_xgb_params = xgb_optimizer.optimize(X, y)
#
# # Visualize results
# xgb_optimizer.plot_optimization_history()
# xgb_optimizer.plot_param_importances()
# xgb_optimizer.plot_slice()