import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import os
import sys
# Ensure project root is on sys.path so `from src...` imports work when
# running scripts directly (e.g. `python src/data/data_preprocessing.py`).
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from src.logger import logging


def standard_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the numerical features in the DataFrame."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logging.info('Data standardized successfully using StandardScaler.')
        return df

    except Exception as e:
        logging.error('Error during data standardization: %s', e)
        raise

def label_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features in the DataFrame using Label Encoding."""
    try:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(df)
        logging.info('Data encoded successfully using LabelEncoder.')
        return y_encoded
    
    except Exception as e:
        logging.error('Error during label encoding: %s', e)
        raise

def save_data(data, file_path: str):
    """Save the preprocessed data (DataFrame or array) to a specified file path."""
    try:
        raw_data_path = os.path.join(file_path)
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        
        # Handle both DataFrames and NumPy arrays
        if isinstance(data, np.ndarray):
            # Convert array to DataFrame before saving
            data = pd.DataFrame(data)
        
        data.to_csv(raw_data_path, index=False)
        logging.info('Preprocessed data saved successfully to %s', raw_data_path)

    except IOError as e:
        logging.error('I/O error while saving data: %s', e)
        raise

def main():
    """Main function to execute the data preprocessing steps."""
    try:
        # Load the data (this should be replaced with actual data loading logic)
        df = pd.read_csv('data/preprocessed_data.csv')

        # Separate features and target variable (this should be customized based on your dataset)
        X = df.drop(columns=['species'])
        y = df['species']

        # Standardize the features
        X_scaled = standard_scaler(X)

        # Encode the target variable
        y_encoded = label_encoder(y)

        # Save the data after preprocessing
        save_data(X_scaled, 'data/processed/X_scaled.csv')
        save_data(y_encoded, 'data/processed/y_encoded.csv')

    except FileNotFoundError as e:
        logging.error('File not found: %s', e)
        raise

    
if __name__ == "__main__":
    main()
