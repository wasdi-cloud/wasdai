# Use an official Python image with GDAL support for geospatial libraries
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.0

ENV WASDAI_WORKERS=2

# Set working directory
WORKDIR /app

# Install Python, pip, and PostgreSQL development libraries
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file from the subfolder relative to the root build context
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY llm_api_server/ ./llm_api_server/
COPY utils/ ./utils/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application (Shell form allows env var expansion)
CMD gunicorn llm_api_server.wasdiLLMServer:oApp \
    -k uvicorn.workers.UvicornWorker \
    --workers $WASDAI_WORKERS \
    --bind 0.0.0.0:8000 \
    --timeout 120