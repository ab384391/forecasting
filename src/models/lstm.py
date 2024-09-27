import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.models.base import DeepForecaster
from src.config import MODEL_CONFIGS

class LSTMForecaster(DeepForecaster):
    def __init__(self, config: Dict[str, Any]):
        super().__init__('lstm', config)
        self.sequence_length = config.get('sequence_length', 30)
        self.forecast_horizon = config.get('forecast_horizon', 1)
        self.num_features = None
        self.lstm_units = config.get('lstm_units', [64, 32])
        self.dropout_rate = config.get('dropout_rate', 0.2)

    def build_model(self):
        self.model = Sequential([
            LSTM(self.lstm_units[0], return_sequences=True, input_shape=(self.sequence_length, self.num_features)),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units[1], return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(self.forecast_horizon)
        ])

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='mse'
        )

    def create_dataset(self, X: pd.DataFrame, y: pd.Series = None) -> tf.data.Dataset:
        data = X.values
        self.num_features = data.shape[1]
        
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=y.values if y is not None else None,
            sequence_length=self.sequence_length,
            sequence_stride=1,
            batch_size=self.config['batch_size']
        )
        
        return dataset

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LSTMForecaster':
        self.num_features = X.shape[1]
        self.build_model()
        self.compile_model()
        
        train_dataset = self.create_dataset(X, y)
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        self.model.fit(
            train_dataset,
            epochs=self.config['num_epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        test_dataset = self.create_dataset(X)
        predictions = self.model.predict(test_dataset)
        return predictions

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        point_predictions = self.predict(X)
        
        # Use Monte Carlo Dropout for uncertainty estimation
        mc_predictions = []
        for _ in range(100):  # Perform 100 stochastic forward passes
            mc_predictions.append(self.model(X.values, training=True))
        
        mc_predictions = np.array(mc_predictions)
        lower_bound = np.percentile(mc_predictions, alpha/2 * 100, axis=0)
        upper_bound = np.percentile(mc_predictions, (1 - alpha/2) * 100, axis=0)
        
        return point_predictions, lower_bound, upper_bound

    def get_feature_importance(self) -> pd.Series:
        # LSTM doesn't provide direct feature importance
        # You could implement a custom method, e.g., based on input perturbation
        raise NotImplementedError("Feature importance not available for LSTM model.")

def create_lstm_forecaster(config: Dict[str, Any] = MODEL_CONFIGS['lstm']) -> LSTMForecaster:
    """
    Create an instance of LSTMForecaster with the specified configuration.
    
    Args:
        config (Dict[str, Any]): Configuration for the LSTMForecaster.
    
    Returns:
        LSTMForecaster: An instance of the LSTMForecaster.
    """
    return LSTMForecaster(config)

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
    
    # Prepare data for LSTM
    X_train = train_engineered.drop(columns=[DATA_CONFIG['target_column']])
    y_train = train_engineered[DATA_CONFIG['target_column']]
    X_test = test_engineered.drop(columns=[DATA_CONFIG['target_column']])
    y_test = test_engineered[DATA_CONFIG['target_column']]
    
    # Create and train LSTM model
    lstm_forecaster = create_lstm_forecaster()
    lstm_forecaster.fit(X_train, y_train)
    
    # Make predictions
    predictions = lstm_forecaster.predict(X_test)
    point_predictions, lower_bound, upper_bound = lstm_forecaster.predict_interval(X_test)
    
    print(f"LSTM predictions shape: {predictions.shape}")
    print(f"Point predictions shape: {point_predictions.shape}")
    print(f"Lower bound shape: {lower_bound.shape}")
    print(f"Upper bound shape: {upper_bound.shape}")
    
    # Save the model
    lstm_forecaster.save()
    print(f"LSTM model saved to {SAVE_LOAD_CONFIG['model_save_path']}/lstm_model.joblib")
