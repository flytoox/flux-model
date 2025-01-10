FROM python:latest

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir diffusers transformers accelerate fastapi torch uvicorn pydantic

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]

