import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
import yaml
from src.data_loading import load_and_split_data
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.models.gradient_boost import create_gradient_boost_forecaster, GradientBoostForecaster
from src.models.lightgbm import create_lightgbm_forecaster, LightGBMForecaster
from src.models.lstm import create_lstm_forecaster, LSTMForecaster
from src.models.nbeats import create_nbeats_forecaster, NBEATSForecaster
from src.ensemble import create_ensemble_forecaster
from src.backtesting import FastBacktester, plot_backtest_results
from src.hyperparameter_optimization import optimize_model
from src.config import DATA_CONFIG, SAVE_LOAD_CONFIG, ENSEMBLE_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_results(results: pd.DataFrame, metrics: Dict[str, float], path: str):
    results.to_csv(f"{path}/forecast_results.csv", index=False)
    with open(f"{path}/metrics.yaml", 'w') as file:
        yaml.dump(metrics, file)

def main(config_path: str):
    # Load configuration
    config = load_config(config_path)
    logging.info("Configuration loaded")

    # Load and preprocess data
    train_data, test_data = load_and_split_data(DATA_CONFIG['raw_data_path'])
    train_preprocessed, test_preprocessed, preprocessor = preprocess_data(train_data, test_data)
    train_engineered, test_engineered, feature_engineer = engineer_features(train_preprocessed, test_preprocessed)
    logging.info("Data loaded, preprocessed, and engineered")

    # Prepare data for modeling
    X_train = train_engineered.drop(columns=[DATA_CONFIG['target_column']])
    y_train = train_engineered[DATA_CONFIG['target_column']]
    X_test = test_engineered.drop(columns=[DATA_CONFIG['target_column']])
    y_test = test_engineered[DATA_CONFIG['target_column']]

    # Optimize and train models
    models = {}
    model_classes = {
        'gradient_boost': (GradientBoostForecaster, create_gradient_boost_forecaster),
        'lightgbm': (LightGBMForecaster, create_lightgbm_forecaster),
        'lstm': (LSTMForecaster, create_lstm_forecaster),
        'nbeats': (NBEATSForecaster, create_nbeats_forecaster)
    }

    for model_name, (model_class, create_func) in model_classes.items():
        if config['optimize_hyperparameters']:
            logging.info(f"Optimizing {model_name}")
            model = optimize_model(model_class, X_train, y_train, model_name)
        else:
            logging.info(f"Creating {model_name} with default parameters")
            model = create_func()
        
        logging.info(f"Training {model_name}")
        model.fit(X_train, y_train)
        models[model_name] = model

    # Create and train ensemble
    ensemble = create_ensemble_forecaster(list(models.values()), ENSEMBLE_CONFIG)
    logging.info("Training ensemble")
    ensemble.fit(X_train, y_train)

    # Backtesting
    backtester = FastBacktester()
    logging.info("Performing backtesting")
    backtest_results = backtester.backtest(train_engineered, ensemble)
    backtest_metrics = backtester.evaluate(backtest_results)
    interval_metrics = backtester.evaluate_intervals(train_engineered, ensemble)
    logging.info("Backtesting completed")

    # Final predictions on test set
    logging.info("Making final predictions")
    predictions = ensemble.predict(X_test)
    point_predictions, lower_bound, upper_bound = ensemble.predict_interval(X_test)

    # Prepare results
    results = pd.DataFrame({
        'date': test_data[DATA_CONFIG['date_column']],
        'actual': y_test,
        'predicted': predictions.flatten(),
        'lower_bound': lower_bound.flatten(),
        'upper_bound': upper_bound.flatten()
    })

    # Calculate final metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    final_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'MAE': mean_absolute_error(y_test, predictions),
        'MAPE': mean_absolute_percentage_error(y_test, predictions)
    }

    # Save results and models
    save_results(results, final_metrics, SAVE_LOAD_CONFIG['model_save_path'])
    ensemble.save()
    preprocessor.save()
    feature_engineer.save()
    logging.info("Results and models saved")

    # Plot backtesting results
    plot_backtest_results(backtest_results)
    logging.info("Backtesting results plotted")

    # Print final metrics
    logging.info("Final Metrics:")
    for metric, value in final_metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    logging.info("Forecasting process completed")

if __name__ == "__main__":
    config_path = "config.yaml"  # Path to your configuration file
    main(config_path)
