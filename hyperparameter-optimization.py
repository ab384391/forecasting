import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable
from sklearn.model_selection import TimeSeriesSplit
from src.models.base import Forecaster
from src.config import OPTUNA_CONFIG, DATA_CONFIG

class OptunaOptimizer:
    def __init__(self, model_class: type, config: Dict[str, Any] = OPTUNA_CONFIG):
        self.model_class = model_class
        self.n_trials = config['n_trials']
        self.timeout = config['timeout']
        self.n_jobs = config['n_jobs']

    def optimize(self, X: pd.DataFrame, y: pd.Series, param_space: Dict[str, Any]) -> Dict[str, Any]:
        def objective(trial):
            params = {k: v(trial) for k, v in param_space.items()}
            model = self.model_class(params)
            
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_index, val_index in tscv.split(X):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                score = np.sqrt(np.mean((y_val - predictions) ** 2))  # RMSE
                scores.append(score)
            
            return np.mean(scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs)

        return study.best_params

def define_param_space(model_name: str) -> Dict[str, Callable]:
    if model_name == 'gradient_boost':
        return {
            'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': lambda trial: trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 10),
            'subsample': lambda trial: trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': lambda trial: trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        }
    elif model_name == 'lightgbm':
        return {
            'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': lambda trial: trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            'num_leaves': lambda trial: trial.suggest_int('num_leaves', 20, 3000),
            'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': lambda trial: trial.suggest_int('min_child_samples', 1, 300),
            'subsample': lambda trial: trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': lambda trial: trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        }
    elif model_name == 'lstm':
        return {
            'lstm_units': lambda trial: trial.suggest_int('lstm_units', 32, 256),
            'dropout_rate': lambda trial: trial.suggest_uniform('dropout_rate', 0.1, 0.5),
            'learning_rate': lambda trial: trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
        }
    elif model_name == 'nbeats':
        return {
            'num_stacks': lambda trial: trial.suggest_int('num_stacks', 2, 5),
            'num_blocks': lambda trial: trial.suggest_int('num_blocks', 1, 5),
            'num_layers': lambda trial: trial.suggest_int('num_layers', 2, 5),
            'layer_width': lambda trial: trial.suggest_int('layer_width', 64, 512),
            'learning_rate': lambda trial: trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

def optimize_model(model_class: type, X: pd.DataFrame, y: pd.Series, model_name: str) -> Forecaster:
    optimizer = OptunaOptimizer(model_class)
    param_space = define_param_space(model_name)
    best_params = optimizer.optimize(X, y, param_space)
    
    print(f"Best parameters for {model_name}:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    return model_class(best_params)

if __name__ == "__main__":
    # Example usage
    from src.data_loading import load_and_split_data
    from src.preprocessing import preprocess_data
    from src.feature_engineering import engineer_features
    from src.models.gradient_boost import create_gradient_boost_forecaster, GradientBoostForecaster
    from src.models.lightgbm import create_lightgbm_forecaster, LightGBMForecaster
    from src.models.lstm import create_lstm_forecaster, LSTMForecaster
    from src.models.nbeats import create_nbeats_forecaster, NBEATSForecaster

    # Load and preprocess data
    train_data, test_data = load_and_split_data(DATA_CONFIG['raw_data_path'])
    train_preprocessed, test_preprocessed, _ = preprocess_data(train_data, test_data)
    train_engineered, test_engineered, _ = engineer_features(train_preprocessed, test_preprocessed)

    # Prepare data
    X_train = train_engineered.drop(columns=[DATA_CONFIG['target_column']])
    y_train = train_engineered[DATA_CONFIG['target_column']]

    # Optimize models
    models_to_optimize = [
        (GradientBoostForecaster, 'gradient_boost'),
        (LightGBMForecaster, 'lightgbm'),
        (LSTMForecaster, 'lstm'),
        (NBEATSForecaster, 'nbeats')
    ]

    optimized_models = {}
    for model_class, model_name in models_to_optimize:
        print(f"\nOptimizing {model_name}...")
        optimized_model = optimize_model(model_class, X_train, y_train, model_name)
        optimized_models[model_name] = optimized_model

    # You can now use these optimized models for forecasting or ensemble creation
    print("\nOptimization complete. Optimized models are ready for use.")
