FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y build-essential && \
    pip install -U pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt \
    --timeout 600 \
    --retries 3 \
    --resume-retries 5
ENV PYTHONPATH=/app