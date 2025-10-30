# --- Archivo: dags/dag_actualziar_cvs.py (CORREGIDO v5 - Imports Revertidos) ---

from __future__ import annotations
from datetime import datetime
import sys

# --- Imports Revertidos a la versión que funcionaba (ignoramos los warnings) ---
from airflow.decorators import dag, task # <--- REVERTIDO
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.empty import EmptyOperator # <--- REVERTIDO
from airflow.utils.trigger_rule import TriggerRule # <--- REVERTIDO

# --- Configuración de Ruta ---
PROJECT_PATH = '/opt/airflow/dags'
sys.path.append(PROJECT_PATH)
# -----------------------------

# Este 'import' ahora es "ligero" y no causará timeout
# (Asumiendo que guardaste los cambios en cv_etl.py)
try:
    from pipeline import cv_etl
except ImportError:
    print(f"Error: No se pudo importar 'cv_etl' desde {PROJECT_PATH}")
    raise

# El endpoint del CV en ESPAÑOL
CV_ENDPOINT_ES = "/matiasrodriguezc/portfolio/main/assets/CV-ES%20-%20MR.pdf"

@dag(
    dag_id="pipeline_actualizacion_cv_github",
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
        cv_etl.descargar_cv_de_github()

    @task
    def task_borrar_vectores_viejos():
        cv_etl.borrar_vectores_viejos()

    @task
    def task_procesar_y_almacenar():
        chunks = cv_etl.extraer_y_dividir_texto()
        cv_etl.crear_y_almacenar_vectores(chunks)

    @task
    def task_limpiar_temporal():
        cv_etl.limpiar_archivo_local()

    # --- FLUJO SIMPLIFICADO ---

    sensor_http_es = HttpSensor(
        task_id="sensor_cv_es_github",
        http_conn_id="github_raw_http",
        endpoint=CV_ENDPOINT_ES,
        method="HEAD",
        response_check=lambda response: response.status_code == 200,
        poke_interval=60 * 60, # Revisa cada hora
        timeout=60 * 60 * 24,
        mode="poke"
    )

    (
        sensor_http_es
        >> task_descargar_cv()
        >> task_borrar_vectores_viejos()
        >> task_procesar_y_almacenar()
        >> task_limpiar_temporal()
    )

actualizar_cv_pipeline_github()