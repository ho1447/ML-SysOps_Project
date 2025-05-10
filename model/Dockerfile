# Use a slim Python image
FROM python:3.10-slim

# Prevent .pyc and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ML Flow
ENV MLFLOW_TRACKING_URI=http://localhost:5000 

# Set working directory inside the container
WORKDIR /app

# Install system-level packages needed for audio + PyTorch
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to use Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual model code
COPY model/ /app/


# Set default command — override with `docker run ... python eval.py`
CMD ["python", "train.py"]
