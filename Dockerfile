FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    pip install -U pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt
