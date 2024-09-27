import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from src.models.base import Forecaster
from src.ensemble import EnsembleForecaster
from src.config import BACKTESTING_CONFIG, DATA_CONFIG

class FastBacktester:
    def __init__(self, config: Dict[str, Any] = BACKTESTING_CONFIG):
        self.test_size = config['test_size']
        self.n_splits = config['n_splits']
        self.horizon = config['horizon']
        self.target_column = DATA_CONFIG['target_column']
        self.date_column = DATA_CONFIG['date_column']

    def _create_folds(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        total_size = len(df)
        test_size = int(total_size * self.test_size)
        fold_size = test_size // self.n_splits

        folds = []
        for i in range(self.n_splits):
            test_end = total_size - i * fold_size
            test_start = test_end - fold_size
            train_end = test_start

            train = df.iloc[:train_end]
            test = df.iloc[test_start:test_end]
            folds.append((train, test))

        return folds

    def backtest(self, df: pd.DataFrame, model: Union[Forecaster, EnsembleForecaster]) -> pd.DataFrame:
        folds = self._create_folds(df)
        results = []

        for i, (train, test) in enumerate(folds):
            print(f"Processing fold {i+1}/{self.n_splits}")
            
            X_train = train.drop(columns=[self.target_column])
            y_train = train[self.target_column]
            X_test = test.drop(columns=[self.target_column])
            y_test = test[self.target_column]

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            fold_results = pd.DataFrame({
                'fold': i+1,
                'date': test[self.date_column],
                'actual': y_test,
                'predicted': predictions.flatten()
            })
            results.append(fold_results)

        return pd.concat(results, ignore_index=True)

    def evaluate(self, results: pd.DataFrame) -> Dict[str, float]:
        actual = results['actual']
        predicted = results['predicted']

        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

    def evaluate_intervals(self, df: pd.DataFrame, model: Union[Forecaster, EnsembleForecaster], alpha: float = 0.05) -> Dict[str, float]:
        folds = self._create_folds(df)
        coverage = []

        for train, test in folds:
            X_train = train.drop(columns=[self.target_column])
            y_train = train[self.target_column]
            X_test = test.drop(columns=[self.target_column])
            y_test = test[self.target_column]

            model.fit(X_train, y_train)
            _, lower, upper = model.predict_interval(X_test, alpha)

            in_interval = (y_test >= lower.flatten()) & (y_test <= upper.flatten())
            coverage.append(in_interval.mean())

        return {
            'Mean Coverage': np.mean(coverage),
            'Coverage Std': np.std(coverage)
        }

def plot_backtest_results(results: pd.DataFrame, output_path: str = 'backtest_results.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for fold in results['fold'].unique():
        fold_data = results[results['fold'] == fold]
        plt.plot(fold_data['date'], fold_data['actual'], label=f'Actual (Fold {fold})')
        plt.plot(fold_data['date'], fold_data['predicted'], label=f'Predicted (Fold {fold})', linestyle='--')
    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Backtest Results')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Example usage
    from src.data_loading import load_and_split_data
    from src.preprocessing import preprocess_data
    from src.feature_engineering import engineer_features
    from src.models.gradient_boost import create_gradient_boost_forecaster
    from src.ensemble import create_ensemble_forecaster
    from src.models.lstm import create_lstm_forecaster

    # Load and preprocess data
    train_data, _ = load_and_split_data(DATA_CONFIG['raw_data_path'])
    train_preprocessed, _, preprocessor = preprocess_data(train_data, train_data)  # Using train_data twice as we don't need test set for backtesting
    train_engineered, _, feature_engineer = engineer_features(train_preprocessed, train_preprocessed)

    # Create models
    gb_model = create_gradient_boost_forecaster()
    lstm_model = create_lstm_forecaster()
    ensemble = create_ensemble_forecaster([gb_model, lstm_model])

    # Create backtester
    backtester = FastBacktester()

    # Perform backtesting
    backtest_results = backtester.backtest(train_engineered, ensemble)

    # Evaluate results
    metrics = backtester.evaluate(backtest_results)
    print("Backtest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Evaluate prediction intervals
    interval_metrics = backtester.evaluate_intervals(train_engineered, ensemble)
    print("\nPrediction Interval Metrics:")
    for metric, value in interval_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot results
    plot_backtest_results(backtest_results)
    print("Backtest results plot saved as 'backtest_results.png'")
