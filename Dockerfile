FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py token_utils.py memory.py graph.py api.py eval.py ./

EXPOSE 8000

CMD ["uvicorn", "api:api", "--host", "0.0.0.0", "--port", "8000"]
