FROM python:3.11-slim

WORKDIR /app

# libgomp1 es requerido por sentence-transformers/torch en CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
