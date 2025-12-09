# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in one go
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data models

# Expose ports
# Render.com uses PORT environment variable
EXPOSE ${PORT}

# Streamlit command for Render (single port)
CMD streamlit run app/ui.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
