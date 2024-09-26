# models/deep_learning_models.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, GlobalAveragePooling1D, Input, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam
from .base_forecaster import Forecaster
import numpy as np

class DeepLearningForecaster(Forecaster):
    def __init__(self, model_type: str, input_shape: tuple, output_shape: int, **kwargs):
        self.model_type = model_type
        self.model = self._build_model(input_shape, output_shape, **kwargs)

    def _build_model(self, input_shape: tuple, output_shape: int, **kwargs) -> tf.keras.Model:
        if self.model_type == 'LSTM':
            return self._build_lstm(input_shape, output_shape)
        elif self.model_type == 'TCN':
            return self._build_tcn(input_shape, output_shape)
        elif self.model_type == 'Transformer':
            return self._build_transformer(input_shape, output_shape)
        elif self.model_type == 'TFT':
            return self._build_tft(input_shape, output_shape, **kwargs)
        elif self.model_type == 'NBeats':
            return self._build_nbeats(input_shape, output_shape, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _build_lstm(self, input_shape: tuple, output_shape: int) -> tf.keras.Model:
        model = tf.keras.Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            LSTM(32),
            Dense(output_shape)
        ])
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def _build_tcn(self, input_shape: tuple, output_shape: int) -> tf.keras.Model:
        model = tf.keras.Sequential([
            Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
            Conv1D(32, kernel_size=2, activation='relu'),
            GlobalAveragePooling1D(),
            Dense(output_shape)
        ])
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def _build_transformer(self, input_shape: tuple, output_shape: int) -> tf.keras.Model:
        inputs = Input(shape=input_shape)
        x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=32)(inputs, inputs)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(output_shape)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def _build_tft(self, input_shape: tuple, output_shape: int, 
                   num_heads: int = 4, dropout: float = 0.1, 
                   ff_dim: int = 256) -> tf.keras.Model:
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(2):  # Using 2 Transformer layers
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=input_shape[-1])(x, x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
            ff = Dense(ff_dim, activation="relu")(x)
            ff = Dense(input_shape[-1])(ff)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = Dense(output_shape)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def _build_nbeats(self, input_shape: tuple, output_shape: int, 
                      num_stacks: int = 30, num_blocks: int = 1, 
                      num_layers: int = 4, layer_width: int = 256) -> tf.keras.Model:
        def create_block(x, theta_layer):
            backcast, forecast = theta_layer(x)
            return backcast, forecast

        inputs = Input(shape=input_shape)
        backcast = inputs
        forecast = tf.zeros_like(inputs)[:, -output_shape:]

        for _ in range(num_stacks):
            for _ in range(num_blocks):
                theta_layer = Dense(2 * layer_width, activation='relu')
                for _ in range(num_layers - 1):
                    theta_layer = Dense(layer_width, activation='relu')(theta_layer)
                theta_layer = Dense(input_shape[0] + output_shape)(theta_layer)
                backcast_block, forecast_block = create_block(backcast, theta_layer)
                backcast = tf.keras.layers.subtract([backcast, backcast_block])
                forecast = tf.keras.layers.add([forecast, forecast_block])

        model = Model(inputs=inputs, outputs=forecast)
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        self.model.fit(X_reshaped, y, epochs=100, batch_size=32, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        return self.model.predict(X_reshaped).flatten()

    def get_feature_importance(self) -> np.ndarray:
        # For deep learning models, feature importance is not straightforward
        # We'll use a simple approach based on the first layer weights
        importance = np.abs(self.model.layers[0].get_weights()[0]).mean(axis=1).flatten()
        return importance