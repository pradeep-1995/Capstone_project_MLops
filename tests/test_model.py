# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise unittest.SkipTest("CAPSTONE_TEST not set; skipping tests that require DagsHub/MLflow access")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "pradeep-1995"
        repo_name = "Capstone_project_MLops"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(cls.new_model_name, stages=["Staging"])
        if not versions:
            raise unittest.SkipTest(f"No model versions found for {cls.new_model_name} in Staging stage; skipping tests that require an external MLflow model")
        cls.new_model_version = versions[0].version
        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        cls.new_model = mlflow.sklearn.load_model(cls.new_model_uri)

        # Load the test dataset
        cls.X_test_data = pd.read_csv("data/interim/X_test.csv")
        cls.y_test_data = pd.read_csv("data/interim/y_test.csv")

    @staticmethod
    def get_model_version(model_name: str, stage: str) -> str:
        """Helper method to get the latest model version for a given stage."""
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        return versions[0].version if versions else None
    
    def test_model_loaded(self):
        """Test if the model is loaded successfully."""
        self.assertIsNotNone(self.new_model, "Model loading failed, got None")

    def test_model_signature(self):
        """Test if the model has the expected signature."""
        input_df = pd.DataFrame([self.X_test_data.iloc[0].values], columns=self.X_test_data.columns)    

        predict_new = self.new_model.predict(input_df)
        self.assertEqual(len(predict_new), 1, "Model prediction output length mismatch")
        self.assertIsInstance(predict_new[0], (int, float), "Model prediction output type mismatch")

    def test_model_performance(self):
        """Test if the model performance meets the expected threshold."""
        X_holdout = self.X_test_data.iloc[:, :-1]
        y_holdout = self.y_test_data.iloc[:, -1]

        y_pred_new = self.new_model.predict(X_holdout)

        accuracy = accuracy_score(y_holdout, y_pred_new)
        precision = precision_score(y_holdout, y_pred_new, average='weighted')

        expected_accuracy = 0.4
        expected_precision = 0.4

        self.assertGreaterEqual(accuracy, expected_accuracy, f"Model accuracy {accuracy:.4f} is below the expected threshold of {expected_accuracy:.4f}")
        self.assertGreaterEqual(precision, expected_precision, f"Model precision {precision:.4f} is below the expected threshold of {expected_precision:.4f}")


if __name__ == '__main__':
    unittest.main()


