"""
Script para ejecutar el ETL completo del CV.
Este script se ejecuta desde GitHub Actions.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cv_etl import (
    descargar_cv_de_google_docs,
    extraer_y_dividir_texto,
    borrar_vectores_viejos,
    crear_y_almacenar_vectores,
    limpiar_archivo_local,
)

def main():
    """Ejecuta el pipeline completo de ETL."""
    try:
        print("=" * 50)
        print("Iniciando pipeline de actualización de CV")
        print("=" * 50)
        
        # Paso 1: Descargar CV desde Google Docs
        descargar_cv_de_google_docs()
        
        # Paso 2: Extraer y dividir texto
        chunks = extraer_y_dividir_texto()
        
        # Paso 3: Borrar vectores antiguos
        borrar_vectores_viejos()
        
        # Paso 4: Crear y almacenar nuevos vectores
        crear_y_almacenar_vectores(chunks)
        
        # Paso 5: Limpiar archivo temporal
        limpiar_archivo_local()
        
        print("=" * 50)
        print("Pipeline completado exitosamente")
        print("=" * 50)
        
    except Exception as e:
        print(f"ERROR en el pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
