FROM python:3.11-slim-bullseye

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get upgrade -y && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

CMD ["python3","app.py"]