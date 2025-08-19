# Use official lightweight Python image
FROM python:3.11-slim

# Set wokring directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from root
COPY requirements.txt .

# Install Python dependecies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ .

# Copy data & inidces into the container
COPY data/ ./data
COPY indices/ ./indices

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI (main.py)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]