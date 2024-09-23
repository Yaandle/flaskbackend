# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster as builder

# Set environment variable to ensure Python output is sent straight to terminal without buffering
ENV PYTHONUNBUFFERED=1 \
    # Prevents Python from writing pyc files to disc
    PYTHONDONTWRITEBYTECODE=1 \
    # Prevents pip from caching downloaded packages
    PIP_NO_CACHE_DIR=off \
    # Prevents pip from checking for newer versions on PyPI
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgtk2.0-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file initially
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start a new stage for the final image
FROM python:3.9-slim-buster

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy necessary libraries from builder stage
COPY --from=builder /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY --from=builder /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Set the working directory in the container
WORKDIR /app

# Copy the application code
COPY . .

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run gunicorn when the container launches
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app