# Use an official Python image with GDAL support for geospatial libraries
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.0

# Set working directory
WORKDIR /app

# Install Python, pip, and PostgreSQL development libraries
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY ../requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# --workers: number of worker processes
CMD ["gunicorn", "main:oApp", "-k", "uvicorn.workers.UvicornWorker", "--workers", "8", "--bind", "0.0.0.0:8000", "--timeout", "120"]
