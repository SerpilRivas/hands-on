# Dockerfile
# Use Python 3.11, the latest version officially supported by PyCaret
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install system-level build dependencies
# This is a critical step to install compilers (like gcc, g++) needed by
# some of PyCaret's underlying packages (e.g., shap, wordcloud).
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy only the requirements file to leverage Docker's layer caching.
# This layer only gets rebuilt if requirements.txt changes.
COPY requirements.txt .

# 3. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit runs on
EXPOSE 8501

# The CMD to start the application is located in the docker-compose.yml file
# for better development/production flexibility.

