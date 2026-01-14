FROM python:3.11-slim

# Optimizaciones de Python para contenedores
# PYTHONDONTWRITEBYTECODE: Evita crear archivos .pyc innecesarios (ahorra espacio)
# PYTHONUNBUFFERED: Asegura que los logs se vean en tiempo real en Koyeb
# PIP_NO_CACHE_DIR: No guarda caché de pip (ahorra espacio en la imagen final)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instalar dependencias del sistema mínimas
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Copiamos requirements primero (para aprovechar caché de Docker Layers)
COPY requirements-api.txt /app/requirements.txt

# 2. ESTRATEGIA DE REDUCCIÓN DE PESO (CRÍTICO):
# Forzamos la instalación de PyTorch versión CPU antes que nada.
# Si alguna librería en tu requirements pide torch, esto evitará que descargue la versión GPU (2GB+).
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Instalamos el resto de las dependencias
RUN pip install -r /app/requirements.txt

# 4. Copiamos el código fuente
# Gracias al .dockerignore, esto ya NO copiará 'venv' ni archivos basura
COPY . /app/

# Configuración de puerto
ENV PORT=8000
EXPOSE 8000

# Comando de inicio compatible con la inyección de puerto de Koyeb
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]