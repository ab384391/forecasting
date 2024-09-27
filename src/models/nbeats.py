import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.models.base import DeepForecaster
from src.config import MODEL_CONFIGS

class NBEATSBlock(tf.keras.layers.Layer):
    def __init__(self, units, thetas_dim, backcast_length, forecast_length, share_weights=False):
        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_weights = share_weights

        self.fc1 = Dense(units, activation='relu')
        self.fc2 = Dense(units, activation='relu')
        self.fc3 = Dense(units, activation='relu')
        self.fc4 = Dense(units, activation='relu')
        if share_weights:
            self.theta_b_fc = self.theta_f_fc = Dense(thetas_dim, activation='linear')
        else:
            self.theta_b_fc = Dense(thetas_dim, activation='linear')
            self.theta_f_fc = Dense(thetas_dim, activation='linear')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.theta_b_fc(x), self.theta_f_fc(x)

class NBEATSForecaster(DeepForecaster):
    def __init__(self, config: Dict[str, Any]):
        super().__init__('nbeats', config)
        self.backcast_length = config.get('backcast_length', 10)
        self.forecast_length = config.get('forecast_length', 5)
        self.stack_types = config.get('stack_types', ['trend', 'seasonality'])
        self.nb_blocks_per_stack = config.get('nb_blocks_per_stack', 3)
        self.thetas_dim = config.get('thetas_dim', 8)
        self.share_weights_in_stack = config.get('share_weights_in_stack', False)
        self.hidden_layer_units = config.get('hidden_layer_units', 256)

    def build_model(self):
        input_layer = Input(shape=(self.backcast_length, 1))
        backcast, forecast = input_layer, input_layer
        for stack_id in range(len(self.stack_types)):
            for _ in range(self.nb_blocks_per_stack):
                backcast, block_forecast = self._create_block(backcast)
                forecast = Lambda(lambda x: x[0] + x[1])([forecast, block_forecast])
        self.model = Model(input_layer, forecast)

    def _create_block(self, x):
        block = NBEATSBlock(
            self.hidden_layer_units,
            self.thetas_dim,
            self.backcast_length,
            self.forecast_length,
            self.share_weights_in_stack
        )
        theta_b, theta_f = block(x)
        backcast = self._create_forecast(theta_b, 'backcast')
        forecast = self._create_forecast(theta_f, 'forecast')
        return Lambda(lambda x: x[0] - x[1])([x, backcast]), forecast

    def _create_forecast(self, thetas, forecast_type):
        if forecast_type == 'backcast':
            size = self.backcast_length
        else:
            size = self.forecast_length
        return Dense(size, activation='linear')(thetas)

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='mse'
        )

    def create_dataset(self, X: pd.DataFrame, y: pd.Series = None) -> tf.data.Dataset:
        data = X.values.reshape(-1, self.backcast_length, 1)
        if y is not None:
            targets = y.values.reshape(-1, self.forecast_length, 1)
            dataset = tf.data.Dataset.from_tensor_slices((data, targets))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(data)
        return dataset.batch(self.config['batch_size'])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NBEATSForecaster':
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
        return predictions.reshape(-1, self.forecast_length)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        point_predictions = self.predict(X)
        
        # For simplicity, we'll use a constant scaling factor for the intervals
        # In practice, you might want to use a more sophisticated method
        lower_bound = point_predictions * 0.9
        upper_bound = point_predictions * 1.1
        
        return point_predictions, lower_bound, upper_bound

    def get_feature_importance(self) -> pd.Series:
        # N-BEATS doesn't provide direct feature importance
        # You could implement a custom method, e.g., based on input perturbation
        raise NotImplementedError("Feature importance not available for N-BEATS model.")

def create_nbeats_forecaster(config: Dict[str, Any] = MODEL_CONFIGS['nbeats']) -> NBEATSForecaster:
    """
    Create an instance of NBEATSForecaster with the specified configuration.
    
    Args:
        config (Dict[str, Any]): Configuration for the NBEATSForecaster.
    
    Returns:
        NBEATSForecaster: An instance of the NBEATSForecaster.
    """
    return NBEATSForecaster(config)

if __name__ == "__main__":
    # Example usage
    from src.data_loading import load_and_split_data
    from src.preprocessing import preprocess_data
    from src.feature_engineering import engineer_features
    
    # Load and preprocess data
    train_data, test_data = load_and_split_data(DATA_CONFIG['raw_data_path'])
    train_preprocessed, test_preprocessed, _ = preprocess_data(train_data, test_data)
    train_engineered, test_engineered, _ = engineer_features(train_preprocessed, test_preprocessed)
    
    # Prepare data for N-BEATS
    X_train = train_engineered.drop(columns=[DATA_CONFIG['target_column']])
    y_train = train_engineered[DATA_CONFIG['target_column']]
    X_test = test_engineered.drop(columns=[DATA_CONFIG['target_column']])
    y_test = test_engineered[DATA_CONFIG['target_column']]
    
    # Create and train N-BEATS model
    nbeats_forecaster = create_nbeats_forecaster()
    nbeats_forecaster.fit(X_train, y_train)
    
    # Make predictions
    predictions = nbeats_forecaster.predict(X_test)
    point_predictions, lower_bound, upper_bound = nbeats_forecaster.predict_interval(X_test)
    
    print(f"N-BEATS predictions shape: {predictions.shape}")
    print(f"Point predictions shape: {point_predictions.shape}")
    print(f"Lower bound shape: {lower_bound.shape}")
    print(f"Upper bound shape: {upper_bound.shape}")
    
    # Save the model
    nbeats_forecaster.save()
    print(f"N-BEATS model saved to {SAVE_LOAD_CONFIG['model_save_path']}/nbeats_model.joblib")
