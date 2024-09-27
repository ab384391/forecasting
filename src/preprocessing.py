import pandas as pd
import numpy as np
from typing import List, Dict, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import joblib
from src.config import PREPROCESSING_CONFIG, DATA_CONFIG, SAVE_LOAD_CONFIG

class FastPreprocessor:
    def __init__(self):
        self.numeric_columns = DATA_CONFIG['numerical_columns']
        self.categorical_columns = DATA_CONFIG['categorical_columns']
        self.target_column = DATA_CONFIG['target_column']
        self.date_column = DATA_CONFIG['date_column']
        
        self.numeric_imputer = SimpleImputer(strategy=PREPROCESSING_CONFIG['missing_value_strategy'])
        self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        
        if PREPROCESSING_CONFIG['scaling_method'] == 'standard':
            self.scaler = StandardScaler()
        elif PREPROCESSING_CONFIG['scaling_method'] == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
        
        if PREPROCESSING_CONFIG['encoding_method'] == 'target_encoding':
            self.encoder = TargetEncoder()
        else:
            self.encoder = None

    def fit(self, df: pd.DataFrame) -> 'FastPreprocessor':
        """
        Fit the preprocessor to the data.
        
        Args:
            df (pd.DataFrame): Input dataframe.
        
        Returns:
            FastPreprocessor: Fitted preprocessor.
        """
        # Fit numeric imputer and scaler
        numeric_data = df[self.numeric_columns]
        self.numeric_imputer.fit(numeric_data)
        imputed_numeric = self.numeric_imputer.transform(numeric_data)
        
        if self.scaler:
            self.scaler.fit(imputed_numeric)
        
        # Fit categorical imputer and encoder
        categorical_data = df[self.categorical_columns]
        self.categorical_imputer.fit(categorical_data)
        imputed_categorical = self.categorical_imputer.transform(categorical_data)
        
        if self.encoder:
            self.encoder.fit(imputed_categorical, df[self.target_column])
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted preprocessor.
        
        Args:
            df (pd.DataFrame): Input dataframe.
        
        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        # Transform numeric features
        numeric_data = df[self.numeric_columns]
        imputed_numeric = self.numeric_imputer.transform(numeric_data)
        
        if self.scaler:
            scaled_numeric = self.scaler.transform(imputed_numeric)
        else:
            scaled_numeric = imputed_numeric
        
        # Transform categorical features
        categorical_data = df[self.categorical_columns]
        imputed_categorical = self.categorical_imputer.transform(categorical_data)
        
        if self.encoder:
            encoded_categorical = self.encoder.transform(imputed_categorical)
        else:
            encoded_categorical = pd.get_dummies(imputed_categorical, columns=self.categorical_columns)
        
        # Combine transformed features
        transformed_df = pd.DataFrame(scaled_numeric, columns=self.numeric_columns, index=df.index)
        transformed_df = pd.concat([transformed_df, encoded_categorical], axis=1)
        
        # Add back the target and date columns
        transformed_df[self.target_column] = df[self.target_column]
        transformed_df[self.date_column] = df[self.date_column]
        
        return transformed_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data in one step.
        
        Args:
            df (pd.DataFrame): Input dataframe.
        
        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        self.fit(df)
        return self.transform(df)

    def save(self, path: str = SAVE_LOAD_CONFIG['preprocessor_save_path']):
        """
        Save the fitted preprocessor to disk.
        
        Args:
            path (str): Path to save the preprocessor.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = SAVE_LOAD_CONFIG['preprocessor_save_path']) -> 'FastPreprocessor':
        """
        Load a fitted preprocessor from disk.
        
        Args:
            path (str): Path to load the preprocessor from.
        
        Returns:
            FastPreprocessor: Loaded preprocessor.
        """
        return joblib.load(path)

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, FastPreprocessor]:
    """
    Preprocess train and test data using FastPreprocessor.
    
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, FastPreprocessor]: Preprocessed train and test data, and the fitted preprocessor.
    """
    preprocessor = FastPreprocessor()
    train_preprocessed = preprocessor.fit_transform(train_df)
    test_preprocessed = preprocessor.transform(test_df)
    
    return train_preprocessed, test_preprocessed, preprocessor

if __name__ == "__main__":
    # Example usage
    from src.data_loading import load_and_split_data
    
    train_data, test_data = load_and_split_data(DATA_CONFIG['raw_data_path'])
    train_preprocessed, test_preprocessed, preprocessor = preprocess_data(train_data, test_data)
    
    print(f"Preprocessed train data shape: {train_preprocessed.shape}")
    print(f"Preprocessed test data shape: {test_preprocessed.shape}")
    
    # Save the preprocessor
    preprocessor.save()
    print(f"Preprocessor saved to {SAVE_LOAD_CONFIG['preprocessor_save_path']}")
