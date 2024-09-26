# main.py

import pandas as pd
import yaml
from typing import Dict, Any
import logging
from data.data_loader import FastDataLoader
from data.preprocessor import FastPreprocessor
from feature_engineering.auto_feature_engineer import AutoFeatureEngineer
from feature_engineering.feature_experiment_runner import FeatureExperimentRunner
from models.deep_learning_models import DeepLearningForecaster
from models.tree_based_models import TreeBasedForecaster
from models.ensemble_forecaster import EnsembleForecaster
from optimization.optuna_optimizer import OptunaOptimizer
from evaluation.backtester import FastBacktester
from evaluation.metrics import mean_absolute_error, root_mean_squared_error, symmetric_mean_absolute_percentage_error
from evaluation.visualization import plot_forecast, plot_metric_over_time, plot_feature_importance

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(log_level: str = 'INFO') -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main(config_path: str) -> None:
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config.get('log_level', 'INFO'))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting forecasting pipeline")
    
    try:
        # Load data
        logger.info("Loading data")
        data_loader = FastDataLoader(config['data']['file_path'], use_dask=config['data'].get('use_dask', False))
        data = data_loader.load_data()
        
        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = FastPreprocessor(
            categorical_cols=config['preprocessing']['categorical_cols'],
            numerical_cols=config['preprocessing']['numerical_cols']
        )
        data = preprocessor.fit_transform(data)
        
        # Feature engineering
        logger.info("Performing feature engineering")
        feature_engineer = AutoFeatureEngineer(config['feature_engineering'])
        data = feature_engineer.generate_features(data, config['target_column'])
        
        if config.get('run_feature_experiments', False):
            logger.info("Running feature engineering experiments")
            experiment_runner = FeatureExperimentRunner(config['feature_engineering'], config['target_column'])
            experiment_results = experiment_runner.run_experiments(data, config['feature_experiments'])
            best_experiment = experiment_runner.get_best_experiment(experiment_results, mean_absolute_error)
            logger.info(f"Best feature engineering experiment: {best_experiment}")
            data = experiment_results[best_experiment]
        
        # Split data
        X = data.drop(columns=[config['target_column']])
        y = data[config['target_column']]
        
        # Model selection and training
        logger.info("Training models")
        models = []
        for model_config in config['models']:
            if model_config['type'] == 'deep_learning':
                model = DeepLearningForecaster(
                    model_config['model_type'],
                    input_shape=(X.shape[1], 1),
                    output_shape=1,
                    **model_config.get('params', {})
                )
            elif model_config['type'] == 'tree_based':
                model = TreeBasedForecaster(model_config['model_type'])
            else:
                raise ValueError(f"Unsupported model type: {model_config['type']}")
            
            if model_config.get('optimize', False):
                logger.info(f"Optimizing {model_config['type']} model")
                optimizer = OptunaOptimizer(type(model), feature_engineer, X, y, mean_absolute_error)
                optimization_result = optimizer.optimize(n_trials=model_config.get('n_trials', 100))
                model = optimization_result['best_model']
            
            model.fit(X, y)
            models.append(model)
        
        # Create ensemble
        if len(models) > 1:
            logger.info("Creating ensemble model")
            ensemble = EnsembleForecaster(models)
        else:
            ensemble = models[0]
        
        # Backtesting
        logger.info("Performing backtesting")
        backtester = FastBacktester(
            ensemble, feature_engineer, 
            [mean_absolute_error, root_mean_squared_error, symmetric_mean_absolute_percentage_error]
        )
        backtest_results = backtester.walk_forward_validation(
            data, config['target_column'], 
            config['backtesting']['start_date'], config['backtesting']['end_date'],
            config['backtesting']['window_size'], config['backtesting']['step_size']
        )
        
        # Calculate overall metrics
        overall_metrics = backtester.calculate_overall_metrics(backtest_results)
        logger.info(f"Overall metrics: {overall_metrics}")
        
        # Visualizations
        logger.info("Generating visualizations")
        plot_forecast(
            backtest_results['test_start'], 
            backtest_results['actual'].iloc[-1], 
            backtest_results['predicted'].iloc[-1], 
            title='Last Period Forecast vs Actual'
        )
        plot_metric_over_time(backtest_results['test_start'], backtest_results['metric_0'], 'Mean Absolute Error')
        plot_feature_importance(X.columns, ensemble.get_feature_importance())
        
        logger.info("Forecasting pipeline completed successfully")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main("config.yaml")
