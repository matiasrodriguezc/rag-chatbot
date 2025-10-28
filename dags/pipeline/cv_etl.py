import os
import requests
import sys
from dotenv import load_dotenv
load_dotenv() 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURACIÓN (AHORA PARA UN SOLO CV) ---
CV_RAW_URL = "https://raw.githubusercontent.com/matiasrodriguezc/portfolio/main/assets/CV-ES%20-%20MR.pdf"
CV_FILENAME = "CV-ES.pdf"
LOCAL_CV_PATH = "./temp_cv.pdf"

# Esta es la ruta DENTRO del contenedor de Docker (donde Airflow escribe)
CHROMA_DIR = "/opt/airflow/dags/cv_vector_db"
EMBEDDING_MODEL = "models/embedding-001"
# -----------------------------------------------

def descargar_cv_de_github():
    """Descarga el CV de GitHub."""
    print(f"Descargando {CV_FILENAME} desde {CV_RAW_URL}...")
    try:
        r = requests.get(CV_RAW_URL)
        r.raise_for_status()
        with open(LOCAL_CV_PATH, 'wb') as f:
            f.write(r.content)
        print(f"{CV_FILENAME} descargado.")
    except Exception as e:
        print(f"Error al descargar {CV_FILENAME}: {e}")
        raise

def extraer_y_dividir_texto():
    """Carga el PDF local, lo limpia y lo divide en chunks."""
    print(f"Cargando y dividiendo texto de {CV_FILENAME}...")
    
    loader_pdf = PyPDFLoader(LOCAL_CV_PATH)
    pages_pdf = loader_pdf.load()
    
    for page in pages_pdf:
        page.page_content = ' '.join(page.page_content.split())
        page.metadata['source'] = CV_FILENAME

    char_splitter = CharacterTextSplitter(
                                    separator = ".", 
                                    chunk_size = 500, 
                                    chunk_overlap = 50)
    
    chunks = char_splitter.split_documents(pages_pdf)
    print(f"Total de chunks generados: {len(chunks)}")
    return chunks

def borrar_vectores_viejos():
    """Borra todos los vectores antiguos del CV de la DB."""
    print(f"Borrando vectores antiguos de {CV_FILENAME}...")
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR, 
            embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        )
        
        ids_a_borrar = []
        for doc_id in vectorstore.get(where={'source': CV_FILENAME})['ids']:
             ids_a_borrar.append(doc_id)

        if ids_a_borrar:
            print(f"Encontrados {len(ids_a_borrar)} vectores para borrar.")
            vectorstore.delete(ids=ids_a_borrar)
            print("Vectores antiguos borrados.")
        else:
            print(f"No se encontraron vectores antiguos para '{CV_FILENAME}'.")
            
    except Exception as e:
        print(f"Error borrando vectores (puede que sea la primera ejecución): {e}")

def crear_y_almacenar_vectores(chunks):
    """Toma los chunks de texto y los almacena en Chroma."""
    if not chunks:
        print("No hay chunks para almacenar.")
        return
        
    print(f"Creando y almacenando {len(chunks)} nuevos vectores...")
    embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    vectorstore = Chroma.from_documents(
                                    documents = chunks, 
                                    embedding = embedding, 
                                    persist_directory = CHROMA_DIR)
    print("¡Nuevos vectores almacenados exitosamente!")

def limpiar_archivo_local():
    """Borra el archivo PDF temporal."""
    if os.path.exists(LOCAL_CV_PATH):
        os.remove(LOCAL_CV_PATH)
        print(f"Archivo local temporal {LOCAL_CV_PATH} borrado.")