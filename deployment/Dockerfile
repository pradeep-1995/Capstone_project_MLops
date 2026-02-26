# Base Image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the application code to the container
COPY deployment/ /app/

# Copy the requirements file and install dependencies
COPY deployment\requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy Rest application code
COPY ../ /app/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
