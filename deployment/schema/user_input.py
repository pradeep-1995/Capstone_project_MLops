from pydantic import BaseModel, Field
from typing import Annotated


class UserInput(BaseModel):
    """
    User input schema for the deployment.
    """
    sepal_length: Annotated[float, Field(..., gt=0, lt=10, description="Sepal length in cm")]
    sepal_width: Annotated[float, Field(..., gt=0, lt=10, description="Sepal width in cm")]
    petal_length: Annotated[float, Field(..., gt=0, lt=10, description="Petal length in cm")]
    petal_width: Annotated[float, Field(..., gt=0, lt=10, description="Petal width in cm")]


    

