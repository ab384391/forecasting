import os
from typing import Dict, Any

# Data loading configuration
DATA_CONFIG: Dict[str, Any] = {
    'raw_data_path': os.path.join('data', 'raw', 'timeseries_data.csv'),
    'processed_data_path': os.path.join('data', 'processed', 'processed_data.csv'),
    'date_column': 'date',
    'target_column': 'target',
    'categorical_columns': ['category1', 'category2'],
    'numerical_columns': ['feature1', 'feature2', 'feature3'],
}

# Preprocessing configuration
PREPROCESSING_CONFIG: Dict[str, Any] = {
    'missing_value_strategy': 'ffill',
    'encoding_method': 'target_encoding',
    'scaling_method': 'standard',
}

# Feature engineering configuration
FEATURE_ENGINEERING_CONFIG: Dict[str, bool] = {
    'time_based_features': True,
    'lag_features': True,
    'interaction_features': False,
    'transformation_features': True,
    'differencing': True,
    'fourier_features': True,
    'target_encoding': True,
}

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'nbeats': {
        'num_epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
    },
    'tcn': {
        'num_epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
    },
    'transformer': {
        'num_epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
    },
    'lstm': {
        'num_epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
    },
    'gradient_boost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
    },
    'lightgbm': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
    },
}

# Hyperparameter optimization configuration
OPTUNA_CONFIG: Dict[str, Any] = {
    'n_trials': 100,
    'timeout': 3600,  # 1 hour
    'n_jobs': -1,  # Use all available cores
}

# Ensemble configuration
ENSEMBLE_CONFIG: Dict[str, float] = {
    'nbeats': 0.2,
    'tcn': 0.2,
    'transformer': 0.2,
    'lstm': 0.1,
    'gradient_boost': 0.15,
    'lightgbm': 0.15,
}

# Backtesting configuration
BACKTESTING_CONFIG: Dict[str, Any] = {
    'test_size': 0.2,
    'n_splits': 5,
    'horizon': 30,  # Forecast horizon in days
}

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    'log_file': 'fast_forecasting.log',
    'log_level': 'INFO',
}

# Saving and loading configuration
SAVE_LOAD_CONFIG: Dict[str, str] = {
    'model_save_path': os.path.join('data', 'models'),
    'preprocessor_save_path': os.path.join('data', 'models', 'preprocessor.joblib'),
    'feature_engineer_save_path': os.path.join('data', 'models', 'feature_engineer.joblib'),
}
