from http import client
import numpy as np
import pandas as pd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sklearn.model_selection import train_test_split

import logging
from src.logger import logging

from dotenv import load_dotenv

# for s3 connection
#from src.connections import s3_connection


# Function to load parameters from a YAML file
'''
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
'''


def load_data(file_path: list) -> pd.DataFrame:
    """Load data from MongoDB or S3 and return as a DataFrame."""
    try:

        df = pd.DataFrame(file_path)
        logging.info('Data loaded successfully from MongoDB collection: %s', file_path)
        return df

    except pd.errors.EmptyDataError as e:
        logging.error('No data found in the collection: %s', e)
        raise

    except Exception as e:
        logging.error('Unexpected error while loading data: %s', e)
        raise



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data and return the cleaned DataFrame."""
    try:
        # Example preprocessing steps (these should be customized based on your dataset)
        # Drop MongoDB's default _id column if it exists
        logging.info('Starting data preprocessing')
        df = df.drop(columns=['_id'], errors='ignore')  
        
        df = df.dropna()  # Drop rows with missing values
        df = df.drop_duplicates()  # Drop duplicate rows
        logging.info('Data preprocessed successfully')
        return df
    
    except KeyError as e:
        logging.error('Missing expected column: %s', e)
        raise

    except Exception as e:
        logging.error('Error during data preprocessing: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str):
    """Save the preprocessed DataFrame to a specified file path."""
    try:
        raw_data_path = os.path.join(file_path, 'preprocessed_data.csv')
        df.to_csv(raw_data_path, index=False)
        logging.info('Preprocessed data saved successfully to %s', raw_data_path)

    except IOError as e:
        logging.error('I/O error while saving data: %s', e)
        raise


def main():
    """Main function to execute the data ingestion process."""
    try:
        # Load parameters (if using a YAML file for configuration)
        # params = load_params('params.yaml')
        # Load environment variables
        load_dotenv()

        # Retrieve database credentials from environment variables
        db_name = os.getenv("DB_NAME")
        collection_name = os.getenv("COLLECTION_NAME")
        connection_url = os.getenv("CONNECTION_URL")

        # Connect to MongoDB and load data into a DataFrame
        import pymongo
        client = pymongo.MongoClient(connection_url)
        data_base = client[db_name]
        collection = data_base[collection_name]

        # Pass the appropriate file path or connection details
        df = load_data(list(collection.find()))

        # Preprocess the data
        final_df = preprocess_data(df)

        # Save the preprocessed data to a CSV file
        save_data(final_df, 'data/')

    except (pymongo.errors.ServerSelectionTimeoutError, TypeError, AttributeError) as e:
        logging.error('MongoDB connection error: %s', e)
        raise


if __name__ == "__main__":
    main()