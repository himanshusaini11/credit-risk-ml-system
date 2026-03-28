FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY configs /app/configs
COPY src /app/src

ENV PYTHONPATH=/app/src
ENV CREDITRISK_CONFIG=/app/configs/default.yaml

EXPOSE 8000

CMD ["uvicorn", "creditrisk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
