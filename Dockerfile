# Archivo: Dockerfile

# 1. Parte de la imagen oficial de Airflow
# (Esto usa la misma versión que tu docker-compose.yaml)
FROM apache/airflow:3.1.1-python3.11
# 2. Copia tu archivo de requirements
COPY requirements.txt /

# 3. Instala permanentemente esas librerías
RUN pip install --no-cache-dir -r /requirements.txt