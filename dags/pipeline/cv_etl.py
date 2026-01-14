import os
import requests
import sys
from dotenv import load_dotenv
load_dotenv() 

# --- CONFIGURACIÓN ---
CV_RAW_URL = "https://raw.githubusercontent.com/matiasrodriguezc/portfolio/main/assets/CV-ES%20-%20MR.pdf"
CV_FILENAME = "CV-ES.pdf"
LOCAL_CV_PATH = "./temp_cv.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Configuración de Pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "cv-matias")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY no encontrada en las variables de entorno")
# -----------------------------------------------

def get_embedding_model():
    """Usa la API de Inferencia de Hugging Face (Clase y Parámetros Correctos)."""
    # Usamos la clase que nos recomendó la advertencia
    from langchain_huggingface import HuggingFaceEndpointEmbeddings 
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN no encontrada en las variables de entorno")
    return HuggingFaceEndpointEmbeddings(
        # El nombre correcto del parámetro es 'huggingfacehub_api_token'
        huggingfacehub_api_token=hf_token,  
        # El nombre correcto del parámetro es 'model'
        model=EMBEDDING_MODEL_NAME          
    )

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
    # --- IMPORTS PESADOS MOVIDOS ADENTRO ---
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters.character import CharacterTextSplitter
    print(f"Cargando y dividiendo texto de {CV_FILENAME}...")
    loader_pdf = PyPDFLoader(LOCAL_CV_PATH)
    pages_pdf = loader_pdf.load()
    for page in pages_pdf:
        page.page_content = ' '.join(page.page_content.split())
        page.metadata['source'] = CV_FILENAME
    char_splitter = CharacterTextSplitter(separator = ".", chunk_size = 500, chunk_overlap = 50)
    chunks = char_splitter.split_documents(pages_pdf)
    print(f"Total de chunks generados: {len(chunks)}")
    return chunks

def borrar_vectores_viejos():
    """Borra todos los vectores antiguos del CV de Pinecone."""
    # --- IMPORT PESADO MOVIDO ADENTRO ---
    from pinecone import Pinecone
    print(f"Borrando vectores antiguos de {CV_FILENAME}...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Usar delete con filter para borrar todos los vectores con source = CV_FILENAME
        # Esto es más eficiente que listar y borrar individualmente
        index.delete(filter={"source": CV_FILENAME})
        print(f"Vectores antiguos de '{CV_FILENAME}' borrados exitosamente.")
            
    except Exception as e:
        print(f"Error borrando vectores (puede ser la primera ejecución o no hay vectores): {e}")

def crear_y_almacenar_vectores(chunks):
    """Toma los chunks de texto y los almacena en Pinecone."""
    # --- IMPORT PESADO MOVIDO ADENTRO ---
    from langchain_pinecone import PineconeVectorStore
    if not chunks:
        print("No hay chunks para almacenar.")
        return
    print(f"Creando y almacenando {len(chunks)} nuevos vectores en Pinecone (esto puede tardar)...")
    embedding_function = get_embedding_model()
    
    # Usar PineconeVectorStore para almacenar los documentos
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_function,
        index_name=PINECONE_INDEX_NAME,
    )
    print("¡Nuevos vectores almacenados exitosamente en Pinecone!")

def limpiar_archivo_local():
    """Borra el archivo PDF temporal."""
    if os.path.exists(LOCAL_CV_PATH):
        os.remove(LOCAL_CV_PATH)
        print(f"Archivo local temporal {LOCAL_CV_PATH} borrado.")