from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from deployment.schema.user_input import UserInput
from deployment.schema.user_output import UserOutput
from deployment.schema.predict import model_predict
import asyncio
import pandas as pd
import time
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

app = FastAPI()


#-------------------------------------------
# Create custom Registry
'''
Prometheus metrics for monitoring the application. 
- Request count and latency for API endpoints.
- Prediction count for each class.
- These metrics can be scraped by Prometheus for monitoring and alerting.
'''
registry = CollectorRegistry()

# Define your custom metrics using this registry
# Counter
REQUEST_COUNT = Counter(
    "app_request_count", 
    "Total number of requests to the app", 
     ["method", "endpoint"], 
    registry=registry
)

# Histogram
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", 
    "Latency of requests in seconds", 
     ["endpoint"], 
    registry=registry
)

# Prediction
PREDICTION_COUNT = Counter(
    "model_prediction_count", 
    "Count of predictions for each class", 
     ["prediction"], 
    registry=registry
)
#-------------------------------------------

@app.get("/")
def home():
    # Increment request count metric
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()

    response = {"message": "MLops Deployment with FastAPI"}

    # Increment request latency metric
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response



@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=UserOutput)
async def predict(data: UserInput):

    # Increment request count metric
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    user_input = {
        "sepal_length": data.sepal_length,
        "sepal_width": data.sepal_width,
        "petal_length": data.petal_length,
        "petal_width": data.petal_width
    }
    try:
        prediction_result = await model_predict(user_input)

        # Increment prediction count metric
        PREDICTION_COUNT.labels(prediction=prediction_result["predicted_class"]).inc()

        # Measure latency
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return prediction_result

    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
@app.get("/metrics")
def metrics():
    '''
    Expose Prometheus metrics for scraping.
    - This endpoint will be scraped by Prometheus to collect metrics data.
    '''
    response = generate_latest(registry)
    media_type = CONTENT_TYPE_LATEST

    # `generate_latest` returns bytes; return them directly with proper media type
    return Response(content=response, media_type=media_type)