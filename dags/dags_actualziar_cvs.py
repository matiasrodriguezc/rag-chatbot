# --- Archivo: dags/dag_actualziar_cvs.py (SIMPLIFICADO) ---

from __future__ import annotations
from datetime import datetime
import sys

# Imports corregidos para Airflow 3+
from airflow.sdk.dag import dag
from airflow.sdk.task import task
from airflow.providers.http.sensors.http import HttpSensor
# (Ya no necesitamos EmptyOperator ni TriggerRule)

# --- Configuración de Ruta ---
PROJECT_PATH = '/opt/airflow/dags' 
sys.path.append(PROJECT_PATH)
# -----------------------------

try:
    from pipeline import cv_etl 
except ImportError:
    print(f"Error: No se pudo importar 'cv_etl' desde {PROJECT_PATH}")
    raise

# El endpoint del CV en ESPAÑOL
CV_ENDPOINT_ES = "/matiasrodriguezc/portfolio/main/assets/CV-ES%20-%20MR.pdf"

@dag(
    dag_id="pipeline_actualizacion_cv_github", # <--- ID simplificado
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    doc_md="DAG que monitorea UN CV (ES) en GitHub y actualiza la DB vectorial."
)
def actualizar_cv_pipeline_github():
    """
    Define el pipeline de ETL para UN CV desde GitHub.
    """

    @task
    def task_descargar_cv():
        cv_etl.descargar_cv_de_github() # <--- Función singular

    @task
    def task_borrar_vectores_viejos():
        cv_etl.borrar_vectores_viejos() # <--- Función singular

    @task
    def task_procesar_y_almacenar():
        chunks = cv_etl.extraer_y_dividir_texto()
        cv_etl.crear_y_almacenar_vectores(chunks)

    @task
    def task_limpiar_temporal():
        cv_etl.limpiar_archivo_local() # <--- Función singular

    # --- FLUJO SIMPLIFICADO ---
    
    # 1. El Sensor (solo uno)
    sensor_http_es = HttpSensor(
        task_id="sensor_cv_es_github",
        http_conn_id="github_raw_http",
        endpoint=CV_ENDPOINT_ES,
        method="HEAD",
        response_check=lambda response: response.status_code == 200,
        poke_interval=60 * 60, # Revisa cada hora para no enojar a GitHub
        timeout=60 * 60 * 24,
        mode="poke"
    )
    
    # 2. El flujo ahora es lineal
    (
        sensor_http_es # Si el sensor tiene éxito...
        >> task_descargar_cv()
        >> task_borrar_vectores_viejos()
        >> task_procesar_y_almacenar()
        >> task_limpiar_temporal()
    )

# Llama a la función para que Airflow registre el DAG
actualizar_cv_pipeline_github()