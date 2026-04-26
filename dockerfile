FROM python:3.11-slim

# Install system dependencies needed by dlib
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libx11-dev \
    libopenblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of app
COPY . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run app
CMD ["python", "app.py"]