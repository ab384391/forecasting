# Fast Forecasting System

## Overview

This Fast Forecasting System is a comprehensive, modular solution for time series forecasting. It integrates multiple advanced forecasting models, including gradient boosting, LightGBM, LSTM, and N-BEATS, along with automated feature engineering, hyperparameter optimization, and ensemble techniques. The system is designed for efficiency and scalability, making it suitable for handling large datasets and complex forecasting tasks.

## Features

- Fast data loading and preprocessing
- Automated feature engineering
- Multiple forecasting models:
  - Gradient Boosting (XGBoost)
  - LightGBM
  - LSTM (Long Short-Term Memory)
  - N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting)
- Ensemble forecasting
- Hyperparameter optimization using Optuna
- Backtesting with walk-forward validation
- Confidence interval estimation
- Easy configuration via YAML file

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/fast-forecasting-system.git
   cd fast-forecasting-system
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data:
   - Place your time series data in the `data/raw/` directory.
   - Ensure your data has a date column and a target column.

2. Configure the system:
   - Open `config.yaml` and adjust the settings according to your needs.
   - Ensure the `raw_data_path` in the config file points to your data file.

3. Run the forecasting system:
   ```
   python main.py
   ```

4. View the results:
   - Forecasts will be saved in `data/models/forecast_results.csv`
   - Metrics will be saved in `data/models/metrics.yaml`
   - Backtesting plot will be saved as `backtest_results.png`

## Project Structure

```
fast-forecasting/
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models/
│   │   ├── gradient_boost.py
│   │   ├── lightgbm.py
│   │   ├── lstm.py
│   │   └── nbeats.py
│   ├── ensemble.py
│   ├── backtesting.py
│   └── hyperparameter_optimization.py
├── main.py
├── config.yaml
├── requirements.txt
└── README.md
```

## Configuration

The `config.yaml` file allows you to control various aspects of the forecasting process:

- Data paths and column names
- Preprocessing strategies
- Feature engineering techniques
- Model parameters
- Hyperparameter optimization settings
- Ensemble weights
- Backtesting parameters

Refer to the comments in `config.yaml` for more details on each setting.

## Extending the System

To add a new forecasting model:

1. Create a new file in the `src/models/` directory.
2. Implement the model following the interface defined in `src/models/base.py`.
3. Add the model to the `model_classes` dictionary in `main.py`.
4. Update `config.yaml` to include parameters for your new model.

## Contributing

Contributions to improve the Fast Forecasting System are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or feedback, please open an issue on the GitHub repository or contact [your-email@example.com].
