# Python 3.9 slim imaj
FROM python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data models .streamlit
COPY .streamlit/config.toml .streamlit/

EXPOSE ${PORT}

CMD streamlit run app/ui.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --server.maxUploadSize=500
