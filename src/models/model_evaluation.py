import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import pickle

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.logger import logging

import mlflow
import mlflow.sklearn
import dagshub
import json


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
#dagshub_token = os.getenv("CAPSTONE_TEST")
#if not dagshub_token:
#    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

#os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
#os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

#dagshub_url = "https://dagshub.com"
#repo_owner = "pradeep-1995"
#repo_name = "Capstone_project_MLops"

# Set up MLflow tracking URI
#mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri("https://dagshub.com/pradeep-1995/Capstone_project_MLops.mlflow")
dagshub.init(repo_owner='pradeep-1995', repo_name='Capstone_project_MLops', mlflow=True)
# -------------------------------------------------------------------------------------

def load_model(file_path: str) -> pickle:
    """Load a trained model from a specified file path."""
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logging.info('Model loaded successfully from %s', file_path)
        return model
    
    except FileNotFoundError as e:
        logging.error('Model file not found: %s', e)
        raise

    except Exception as e:
        logging.error('Unexpected error while loading the model: %s', e)
        raise


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


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model on the test set and return evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Handle both binary and multiclass classification for AUC
        if y_pred_prob.shape[1] == 2:
            # Binary classification: use only class 1 probabilities
            auc = roc_auc_score(y_test, y_pred_prob[:, 1])
        else:
            # Multiclass classification: use one-vs-rest approach
            auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr', average='weighted')

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation completed successfully with metrics: %s', metrics_dict)
        return metrics_dict
    
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise


def log_metrics(metrics_dict: dict, file_path: str) -> None:
    """Log evaluation metrics to a specified file path."""
    try:
        with open(file_path, 'w') as f:
            json.dump(metrics_dict, f)
        logging.info('Evaluation metrics logged successfully to %s', file_path)

    except Exception as e:
        logging.error('Error while logging metrics: %s', e)
        raise


def save_model_info(run_id: str, file_path: str, model_path: str) -> None:
    """Save model information including run_id and model artifact path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path, 'file_path': file_path}
        with open(file_path, 'w') as f:
            json.dump(model_info, f, indent=4)

        logging.info('Model information saved successfully to %s', file_path)

    except Exception as e:
        logging.error('Error while saving model information: %s', e)
        raise


def main():
    """Main function to execute the model evaluation process"""
    mlflow.set_experiment('dvc_pipeline')
    with mlflow.start_run() as run:
        try:
            clf = load_model('models/random_forest_model.pkl')

            X_test = load_data('data/interim/X_test.csv').values
            y_test = load_data('data/interim/y_test.csv').values.squeeze()

            metrics_dict = evaluate_model(clf, X_test, y_test)

            log_metrics(metrics_dict, 'reports/metrics.json')

            #Log metrics to MLflow
            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(metric_name, metric_value)

            #Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            #Log model to mlflow
            
            mlflow.sklearn.log_model(clf, "model")

            #Save model info
            save_model_info(run.info.run_id, 'reports/model_metrics.json', "model")

            #Log the model metrics file as an artifact in MLflow
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error('Error in main function: %s', e)
            raise

if __name__ == "__main__":
    main()
    
        