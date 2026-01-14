# Dockerfile.api - Para deploy en Koyeb (100% GRATIS, sin tarjeta de crédito)
FROM python:3.11-slim

WORKDIR /app

# Instala dependencias del sistema mínimas
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiamos requirements de la API
COPY requirements-api.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiamos el código
COPY main.py /app/

# Koyeb usa el puerto 8000 por defecto
ENV PORT=8000
EXPOSE 8000

# Comando de arranque
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
