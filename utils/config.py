import yaml
import os
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str):
        """
        Initialize the Config object.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the YAML file.

        Returns:
            Dict[str, Any]: Loaded configuration dictionary.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key (str): Configuration key.
            default (Any, optional): Default value if key is not found.

        Returns:
            Any: Configuration value.
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key (str): Configuration key.
            value (Any): Configuration value.
        """
        self.config[key] = value

    def save(self) -> None:
        """
        Save the current configuration to the YAML file.
        """
        with open(self.config_path, 'w') as config_file:
            yaml.dump(self.config, config_file, default_flow_style=False)

    def update(self, new_config: Dict[str, Any]) -> None:
        """
        Update the configuration with new values.

        Args:
            new_config (Dict[str, Any]): New configuration dictionary to merge.
        """
        self.config.update(new_config)

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access to configuration values.

        Args:
            key (str): Configuration key.

        Returns:
            Any: Configuration value.

        Raises:
            KeyError: If the key is not found in the configuration.
        """
        if key not in self.config:
            raise KeyError(f"Configuration key not found: {key}")
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allow dictionary-style setting of configuration values.

        Args:
            key (str): Configuration key.
            value (Any): Configuration value.
        """
        self.config[key] = value

def create_default_config(config_path: str) -> None:
    """
    Create a default configuration file if it doesn't exist.

    Args:
        config_path (str): Path to the configuration file.
    """
    if os.path.exists(config_path):
        return

    default_config = {
        'data': {
            'input_file': 'data/input.csv',
            'date_column': 'date',
            'target_column': 'target',
            'features': ['feature1', 'feature2', 'feature3']
        },
        'preprocessing': {
            'handle_missing_values': True,
            'handle_outliers': True,
            'scaling_method': 'standardization'
        },
        'feature_engineering': {
            'create_lag_features': True,
            'lag_periods': [1, 7, 30],
            'create_rolling_features': True,
            'rolling_windows': [7, 30, 90],
            'create_date_features': True
        },
        'model': {
            'type': 'LightGBM',
            'params': {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 100
            }
        },
        'training': {
            'test_size': 0.2,
            'random_state': 42,
            'early_stopping_rounds': 10
        },
        'evaluation': {
            'metrics': ['mape', 'rmse', 'mae']
        },
        'optimization': {
            'perform_hyperopt': True,
            'n_trials': 100,
            'timeout': 3600
        },
        'output': {
            'save_model': True,
            'model_path': 'models/best_model.pkl',
            'save_predictions': True,
            'predictions_path': 'output/predictions.csv'
        }
    }

    with open(config_path, 'w') as config_file:
        yaml.dump(default_config, config_file, default_flow_style=False)

# Usage example:
# create_default_config('config.yaml')
# config = Config('config.yaml')
# print(config['data']['input_file'])
# config['model']['params']['learning_rate'] = 0.1
# config.save()