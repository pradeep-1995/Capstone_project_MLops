import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import yaml
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from src.logger import logging

from src.data.data_preprocessing import save_data


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on the DataFrame and return the modified DataFrame."""
    try:
        pca = PCA(n_components=2)  # Example: Reduce to 2 principal components
        data_pca = pca.fit_transform(data)

        logging.info('Feature engineering completed successfully using PCA.')
        return data_pca
    
    except Exception as e:
        logging.error('Error during feature engineering: %s', e)
        raise

def train_test_split_data(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    """Split the data into training and testing sets."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info('Data split into training and testing sets successfully.')
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error('Error during train-test split: %s', e)
        raise

def main():
    '''Main function to execute the feature engineering process.'''
    try:
        # Load the preprocessed data
        preprocessed_data_path = 'data/processed/X_scaled.csv'
        data = pd.read_csv(preprocessed_data_path)

        # Perform feature engineering
        engineered_data = feature_engineering(data)

        # Save the engineered features
        save_data(engineered_data, 'data/processed/final_df.csv')

        logging.info('Feature engineering process completed successfully.')

        # train_test_split_data(X, y, test_size=0.2, random_state=42)
        params = load_params('params.yaml')
        test_size = params['feature_engineering']['test_size']
        random_state = params['feature_engineering']['random_state']

        target = pd.read_csv('data/processed/y_encoded.csv')
        X_train, X_test, y_train, y_test = train_test_split_data(engineered_data, 
                                                                 target, 
                                                                 test_size=test_size, 
                                                                 random_state=random_state)
        
        # Save the train-test split data
        save_data(X_train, 'data/interim/X_train.csv')
        save_data(X_test, 'data/interim/X_test.csv')
        save_data(y_train, 'data/interim/y_train.csv')
        save_data(y_test, 'data/interim/y_test.csv')
        logging.info('Train-test split data saved successfully.')


    except Exception as e:
        logging.error('Error in the main function: %s', e)
        raise

if __name__ == "__main__":
    main()



