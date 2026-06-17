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

# Copy requirements file
COPY ./requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download the Flashrank model to prevent runtime worker race conditions
RUN python3 -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MultiBERT-L-12')"

# Pre-download Hugging Face BGE-M3 Embeddings model to prevent worker race conditions
RUN python3 -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-m3')"

# Copy the application code
COPY mcp_server/ ./mcp_server/
COPY ai_agent/ ./ai_agent/
COPY utils/ ./utils/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# --workers: number of worker processes
CMD ["gunicorn", "mcp_server.wasdiMCPServer:oApp", "-k", "uvicorn.workers.UvicornWorker", "--workers", "$WASDAI_WORKERS", "--bind", "0.0.0.0:8000", "--timeout", "120"]