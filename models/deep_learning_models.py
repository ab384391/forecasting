import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input, Dropout, GlobalAveragePooling1D, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from .base_forecaster import BaseForecaster

class DeepLearningForecaster(BaseForecaster):
    def __init__(self, model_type: str, input_shape: Tuple[int, int], output_shape: int, learning_rate: float = 0.001):
        self.model_type = model_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def _build_model(self) -> Model:
        if self.model_type == 'LSTM':
            return self._build_lstm_model()
        elif self.model_type == 'TCN':
            return self._build_tcn_model()
        elif self.model_type == 'Transformer':
            return self._build_transformer_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _build_lstm_model(self) -> Model:
        model = Sequential([
            LSTM(50, activation='relu', input_shape=self.input_shape, return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(self.output_shape)
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def _build_tcn_model(self) -> Model:
        def tcn_block(x, dilation_rate):
            x = Conv1D(filters=64, kernel_size=2, dilation_rate=dilation_rate, padding='causal')(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = Dropout(0.2)(x)
            return x

        inputs = Input(shape=self.input_shape)
        x = inputs

        for dilation_rate in [1, 2, 4, 8]:
            x = tcn_block(x, dilation_rate)

        x = GlobalAveragePooling1D()(x)
        outputs = Dense(self.output_shape)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def _build_transformer_model(self) -> Model:
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            x = Attention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
            x = Dropout(dropout)(x)
            res = x + inputs
            x = Dense(ff_dim, activation="relu")(res)
            x = Dense(inputs.shape[-1])(x)
            return x + res

        inputs = Input(shape=self.input_shape)
        x = Dense(32, activation="relu")(inputs)
        x = transformer_encoder(x, head_size=32, num_heads=2, ff_dim=32, dropout=0.1)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(self.output_shape)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1))

        X_reshaped = X_scaled.reshape((X_scaled.shape[0], self.input_shape[0], self.input_shape[1]))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X_reshaped, y_scaled, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.feature_scaler.transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], self.input_shape[0], self.input_shape[1]))
        y_pred_scaled = self.model.predict(X_reshaped)
        return self.target_scaler.inverse_transform(y_pred_scaled)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_scaled = self.feature_scaler.transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], self.input_shape[0], self.input_shape[1]))

        # Monte Carlo Dropout for uncertainty estimation
        n_iterations = 100
        y_pred_list = [self.model(X_reshaped, training=True) for _ in range(n_iterations)]
        y_pred_scaled = np.mean(y_pred_list, axis=0)
        y_pred_std = np.std(y_pred_list, axis=0)

        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        y_pred_std = y_pred_std * self.target_scaler.scale_

        lower_bound = y_pred - 1.96 * y_pred_std
        upper_bound = y_pred + 1.96 * y_pred_std

        return y_pred.squeeze(), lower_bound.squeeze(), upper_bound.squeeze()

    def get_feature_importance(self) -> pd.DataFrame:
        # For deep learning models, we'll use a simple sensitivity analysis
        X_scaled = self.feature_scaler.transform(np.random.randn(1000, self.input_shape[0] * self.input_shape[1]))
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], self.input_shape[0], self.input_shape[1]))
        
        base_prediction = self.model.predict(X_reshaped)
        
        importance = []
        for i in range(X_scaled.shape[1]):
            X_perturbed = X_scaled.copy()
            X_perturbed[:, i] += np.std(X_scaled[:, i])
            X_perturbed_reshaped = X_perturbed.reshape((X_perturbed.shape[0], self.input_shape[0], self.input_shape[1]))
            perturbed_prediction = self.model.predict(X_perturbed_reshaped)
            importance.append(np.mean(np.abs(perturbed_prediction - base_prediction)))
        
        feature_names = [f'feature_{i}' for i in range(X_scaled.shape[1])]
        return pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)

    def clone(self) -> 'DeepLearningForecaster':
        return DeepLearningForecaster(self.model_type, self.input_shape, self.output_shape, self.learning_rate)

    def save(self, path: str) -> None:
        self.model.save(path)
        np.save(f"{path}_feature_scaler", self.feature_scaler.get_params())
        np.save(f"{path}_target_scaler", self.target_scaler.get_params())

    @classmethod
    def load(cls, path: str) -> 'DeepLearningForecaster':
        model = tf.keras.models.load_model(path)
        feature_scaler = StandardScaler()
        feature_scaler.set_params(**np.load(f"{path}_feature_scaler.npy", allow_pickle=True).item())
        target_scaler = StandardScaler()
        target_scaler.set_params(**np.load(f"{path}_target_scaler.npy", allow_pickle=True).item())
        
        forecaster = cls(model.name, model.input_shape[1:], model.output_shape[1])
        forecaster.model = model
        forecaster.feature_scaler = feature_scaler
        forecaster.target_scaler = target_scaler
        return forecaster

# Usage example:
# input_shape = (10, 5)  # 10 time steps, 5 features
# output_shape = 1  # Single step forecast
# forecaster = DeepLearningForecaster('LSTM', input_shape, output_shape)
# forecaster.fit(X_train, y_train)
# predictions = forecaster.predict(X_test)
# predictions, lower_bound, upper_bound = forecaster.predict_interval(X_test)
# feature_importance = forecaster.get_feature_importance()