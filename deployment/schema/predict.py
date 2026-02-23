import pickle
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import asyncio


def load_model(relative_path: str) -> object:
    """Load a model located under the project `models/` directory.

    Tries `joblib.load` first (works for scikit-learn savers) and falls
    back to `pickle.load` if needed. Raises ValueError on failure.
    """
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / relative_path
    if not file_path.exists():
        raise ValueError(f"Model file not found: {file_path}")

    # Prefer joblib (handles large sklearn objects reliably)
    try:
        return joblib.load(file_path)
    except Exception as joblib_exc:
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as pickle_exc:
            raise ValueError(
                f"Error loading model from {file_path}: joblib error: {joblib_exc}; pickle error: {pickle_exc}"
            )


async def scaler_predict(data: pd.DataFrame) -> np.ndarray:
    try:
        scaler = load_model("models/scaler.pkl")
        return await asyncio.to_thread(scaler.transform, data)
    except Exception as e:
        raise ValueError(f"Error in scaler prediction: {e}")


async def pca_predict(data: np.ndarray) -> np.ndarray:
    try:
        pca = load_model("models/pca.pkl")
        return await asyncio.to_thread(pca.transform, data)
    except Exception as e:
        raise ValueError(f"Error in PCA prediction: {e}")

async def model_predict(user_input) -> dict:
    try:
        model = load_model("models/model.pkl")

        # Data preparation: accept dict or DataFrame
        if isinstance(user_input, pd.DataFrame):
            data = user_input
        else:
            # Ensure column order matches training data for scaler/pca
            columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            data = pd.DataFrame([user_input], columns=columns)

        # Scaling and PCA (run in thread to avoid blocking)
        scaled_data = await scaler_predict(data)
        pca_data = await pca_predict(scaled_data)

        # Prediction
        prediction = await asyncio.to_thread(model.predict, pca_data)
        prediction = prediction[0]

        probabilities = await asyncio.to_thread(model.predict_proba, pca_data)
        probabilities = probabilities[0]
        confidence = float(max(probabilities))

        class_labels = {0: "setosa", 1: "versicolor", 2: "virginica"}
        
        predicted_class = class_labels[prediction]
        probabilities_dict = {class_labels[i]: float(probabilities[i]) for i in range(len(probabilities))}
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities_dict
        }
    except Exception as e:
        raise ValueError(f"Error in model prediction: {e}")
    
'''
def test_model_predict():
    # Sample input data
    sample_input = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    # Convert to DataFrame
    # Run prediction
    result = asyncio.run(model_predict(sample_input))
    print(result)

if __name__ == "__main__":
    test_model_predict()
'''