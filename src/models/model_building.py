import numpy as np
import pandas as pd

import pickle
from sklearn.ensemble import RandomForestClassifier

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import logging
from src.logger import logging
import yaml


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int, max_depth: int, criterion: str, random_state: int ) -> RandomForestClassifier:
    """Train a Random Forest Classifier and return the trained model."""
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=random_state)
        model.fit(X_train, y_train)
        logging.info('Model trained successfully with n_estimators=%d, max_depth=%d, criterion=%s', n_estimators, max_depth, criterion)
        return model
    
    except ValueError as e:
        logging.error('Value error during model training: %s', e)
        raise

def save_model(model: RandomForestClassifier, file_path: str):
    """Save the trained model to a specified file path."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info('Model saved successfully to %s', file_path)
    
    except IOError as e:
        logging.error('I/O error while saving the model: %s', e)
        raise


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


def main():
    """Main function to execute the model training process."""
    try:
        # Load the preprocessed data
        X_train = load_data(r'data\interim\X_train.csv')
        y_train = load_data(r'data\interim\y_train.csv').squeeze()  # Convert DataFrame to Series if needed

        # Load model parameters from YAML file
        params = load_params('params.yaml')
        n_estimators = params['model_building']['n_estimators']
        max_depth = params['model_building']['max_depth']
        criterion = params['model_building']['criterion']
        random_state = params['model_building']['random_state']

        # Train the model
        model = train_model(X_train, y_train, n_estimators, max_depth, criterion, random_state)

        # Save the trained model
        save_model(model, 'models/random_forest_model.pkl')

    except Exception as e:
        logging.error('Error in main function: %s', e)
        raise

if __name__ == "__main__":
    main()