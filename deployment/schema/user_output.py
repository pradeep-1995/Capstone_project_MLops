from pydantic import BaseModel, Field
from typing import Dict

class UserOutput(BaseModel):
    """
    User output schema for the deployment.
    """
    predicted_class: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Confidence score for the prediction")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")