import json

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.logger import logging

import mlflow
import dagshub

# Warnings filter to suppress specific warnings during execution
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore")


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

def load_model_info(file_path: str) -> dict:
    """Load model information from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            model_info = json.load(f)
        
        logging.info('Model information loaded successfully from %s', file_path)
        return model_info

    except FileNotFoundError as e:
        logging.error('Model information file not found: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error while loading model information: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register a model in MLflow using the provided model information."""
    try:
        run_id = model_info['run_id']
        model_path = model_info['model_path']
        model_uri = f"runs:/{run_id}/{model_path}"
        
        logging.info(f"Attempting to register model with URI: {model_uri}")
        
        try:
            # Check if the model exists in the run before registering
            mlflow_client = mlflow.tracking.MlflowClient()
            run = mlflow_client.get_run(run_id)
            
            if run is None:
                raise ValueError(f"Run ID {run_id} not found in MLflow")
            
            logging.info(f"Found run {run_id} in MLflow")
        except Exception as e:
            logging.error(f"Could not verify run {run_id}: {e}")
            logging.error("Please ensure model_evaluation.py was run successfully to log the model to MLflow")
            raise

        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logging.info(f'Model {model_name} registered successfully with version {model_version.version} and moved to Staging stage.')

    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def main():
    """Main function to execute the model registry"""
    try:
        # Load model information from JSON file
        model_info_path = 'reports/model_metrics.json'
        model_info = load_model_info(model_info_path)

        # Register the model in MLflow
        model_name = '1st_model'
        register_model(model_name, model_info)

    except Exception as e:
        logging.error('Error in main execution: %s', e)
        raise


if __name__ == "__main__":
    main()

