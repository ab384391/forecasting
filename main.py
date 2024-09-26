import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from data.data_loader import FastDataLoader
from data.preprocessor import FastPreprocessor
from feature_engineering.auto_feature_engineer import AutoFeatureEngineer
from models.deep_learning_models import DeepLearningForecaster
from models.tree_based_models import GradientBoostForecaster, LightGBMForecaster
from models.base_forecaster import BaseForecaster
from optimization.optuna_optimizer import OptunaOptimizer, create_param_space
from evaluation.backtester import FastBacktester
from evaluation.metrics import calculate_all_metrics
from utils.config import Config, create_default_config
from utils.logging_utils import setup_logger, log_execution_time, log_exception
import joblib

@log_execution_time(logger)
@log_exception(logger)
def load_data(config: Config) -> pd.DataFrame:
    """Load the data."""
    loader = FastDataLoader(config['data']['input_file'], config['data']['date_column'], config['data']['target_column'])
    df = loader.load_data()
    return df

@log_execution_time(logger)
@log_exception(logger)
def preprocess_and_engineer_features(df: pd.DataFrame, config: Config, is_training: bool = True) -> Tuple[pd.DataFrame, FastPreprocessor, AutoFeatureEngineer]:
    """Preprocess the data and engineer features."""
    if is_training:
        preprocessor = FastPreprocessor(config['data']['features'], [config['data']['target_column']], config['data']['date_column'])
        auto_fe = AutoFeatureEngineer(config['feature_engineering'])
    else:
        preprocessor = joblib.load(config['output']['preprocessor_path'])
        auto_fe = joblib.load(config['output']['feature_engineer_path'])
    
    df = preprocessor.preprocess(df, is_training=is_training)
    df = auto_fe.engineer_features(df, config['data']['date_column'], config['data']['target_column'], is_training=is_training)
    
    return df, preprocessor, auto_fe

@log_execution_time(logger)
@log_exception(logger)
def split_data(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and testing sets based on a specific date."""
    split_date = pd.to_datetime(config['training']['split_date'])
    
    train_df = df[df[config['data']['date_column']] < split_date]
    test_df = df[df[config['data']['date_column']] >= split_date]
    
    return train_df, test_df

@log_execution_time(logger)
@log_exception(logger)
def create_model(config: Config) -> BaseForecaster:
    """Create the forecasting model based on configuration."""
    model_type = config['model']['type']
    model_params = config['model']['params']
    
    if model_type == 'LSTM':
        return DeepLearningForecaster('LSTM', **model_params)
    elif model_type == 'GradientBoost':
        return GradientBoostForecaster(**model_params)
    elif model_type == 'LightGBM':
        return LightGBMForecaster(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

@log_execution_time(logger)
@log_exception(logger)
def optimize_model(model: BaseForecaster, X: pd.DataFrame, y: pd.Series, config: Config) -> Dict[str, Any]:
    """Optimize model hyperparameters."""
    param_space = create_param_space(config['model']['type'])
    optimizer = OptunaOptimizer(type(model), param_space, n_trials=config['optimization']['n_trials'])
    best_params = optimizer.optimize(X, y)
    return best_params

@log_execution_time(logger)
@log_exception(logger)
def train_model(model: BaseForecaster, X_train: pd.DataFrame, y_train: pd.Series) -> BaseForecaster:
    """Train the model on the training data."""
    model.fit(X_train, y_train)
    return model

@log_execution_time(logger)
@log_exception(logger)
def evaluate_model(model: BaseForecaster, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Evaluate the model on the test data."""
    y_pred = model.predict(X_test)
    metrics = calculate_all_metrics(y_test, y_pred)
    return {'predictions': y_pred, 'metrics': metrics}

@log_execution_time(logger)
def main():
    # Load configuration
    create_default_config('config.yaml')
    config = Config('config.yaml')
    
    # Load data
    df = load_data(config)
    logger.info(f"Data loaded. Shape: {df.shape}")
    
    # Split data into train and test sets
    train_df, test_df = split_data(df, config)
    logger.info(f"Data split into train and test sets. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Preprocess and engineer features for training data
    train_df, preprocessor, auto_fe = preprocess_and_engineer_features(train_df, config, is_training=True)
    logger.info(f"Training data preprocessed and features engineered. New shape: {train_df.shape}")
    
    # Save preprocessor and feature engineer
    joblib.dump(preprocessor, config['output']['preprocessor_path'])
    joblib.dump(auto_fe, config['output']['feature_engineer_path'])
    logger.info("Preprocessor and feature engineer saved.")
    
    # Prepare training data
    X_train = train_df.drop(columns=[config['data']['target_column'], config['data']['date_column']])
    y_train = train_df[config['data']['target_column']]
    
    # Create model
    model = create_model(config)
    logger.info(f"Model created: {type(model).__name__}")
    
    # Optimize model if specified
    if config['optimization']['perform_hyperopt']:
        best_params = optimize_model(model, X_train, y_train, config)
        model = create_model({**config['model'], 'params': best_params})
        logger.info(f"Model optimized. Best parameters: {best_params}")
    
    # Train model
    model = train_model(model, X_train, y_train)
    logger.info("Model training completed.")
    
    # Save model
    if config['output']['save_model']:
        model.save(config['output']['model_path'])
        logger.info(f"Model saved to {config['output']['model_path']}")
    
    # Preprocess and engineer features for test data
    test_df, _, _ = preprocess_and_engineer_features(test_df, config, is_training=False)
    logger.info(f"Test data preprocessed and features engineered. New shape: {test_df.shape}")
    
    # Prepare test data
    X_test = test_df.drop(columns=[config['data']['target_column'], config['data']['date_column']])
    y_test = test_df[config['data']['target_column']]
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    logger.info("Model evaluation completed.")
    logger.info(f"Test set metrics: {results['metrics']}")
    
    # Save predictions
    if config['output']['save_predictions']:
        pd.DataFrame({
            'date': test_df[config['data']['date_column']],
            'true_values': y_test,
            'predictions': results['predictions']
        }).to_csv(config['output']['predictions_path'], index=False)
        logger.info(f"Predictions saved to {config['output']['predictions_path']}")
    
    # Perform backtesting
    backtester = FastBacktester(model, config['evaluation']['initial_train_size'], 
                                config['evaluation']['step_size'], config['evaluation']['horizon'])
    backtest_results = backtester.backtest(df.drop(columns=[config['data']['target_column']]), 
                                           df[config['data']['target_column']])
    logger.info("Backtesting completed.")
    
    # Plot results
    backtester.plot_backtesting_results(backtest_results['results'])
    backtester.plot_metric_over_time(backtest_results['results'], 'mape')
    backtester.plot_feature_importance_over_time(backtest_results['feature_importance'])
    logger.info("Results plotted.")

if __name__ == "__main__":
    logger = setup_logger('forecasting_system', 'logs/forecasting_system.log')
    main()