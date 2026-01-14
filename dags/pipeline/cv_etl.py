import os
import requests
import sys
from dotenv import load_dotenv
load_dotenv() 

# --- CONFIGURACIÓN GOOGLE DOCS ---
DOC_ID = "1C1k0zGNiE_OqICy2PAaHsAV9zCJWY0IHn2FmfovPM28"
CV_RAW_URL = f"https://docs.google.com/document/d/{DOC_ID}/export?format=txt"

CV_FILENAME = "matias_background.txt"
LOCAL_CV_PATH = "./matias_background.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Configuración de Pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "cv-matias")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY no encontrada en las variables de entorno")

def get_embedding_model():
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN no encontrada")
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=hf_token,  
        model=EMBEDDING_MODEL_NAME          
    )

def descargar_cv_de_google_docs():
    """Descarga el texto plano directamente desde Google Docs."""
    print(f"Descargando contenido desde Google Docs...")
    try:
        r = requests.get(CV_RAW_URL)
        r.raise_for_status()
        # Guardamos el contenido. Google suele mandar UTF-8 con BOM a veces,
        # pero al guardarlo como binary ('wb') preservamos lo que llega.
        with open(LOCAL_CV_PATH, 'wb') as f:
            f.write(r.content)
        print(f"Documento guardado localmente como {CV_FILENAME}.")
    except Exception as e:
        print(f"Error al descargar desde Google Docs: {e}")
        raise

def extraer_y_dividir_texto():
    """Carga el TXT local y lo divide en chunks."""
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print(f"Procesando texto de {CV_FILENAME}...")
    # IMPORTANTE: encoding='utf-8' es vital para tildes y ñ que vengan de Google
    loader = TextLoader(LOCAL_CV_PATH, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata['source'] = "GoogleDoc_Matias" # Etiqueta personalizada
    # Usamos RecursiveCharacterTextSplitter, ideal para textos narrativos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Total de chunks generados: {len(chunks)}")
    return chunks


def borrar_vectores_viejos():
    from pinecone import Pinecone
    print(f"Borrando vectores antiguos...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        # Borramos todo lo que venga de la fuente anterior o la nueva
        # Para asegurar limpieza total, podrías borrar todo el namespace si solo usas este CV
        index.delete(delete_all=True) 
        print(f"Índice limpiado exitosamente.")
    except Exception as e:
        print(f"Nota sobre borrado: {e}")


def crear_y_almacenar_vectores(chunks):
    from langchain_pinecone import PineconeVectorStore
    if not chunks:
        return
    print(f"Subiendo {len(chunks)} vectores a Pinecone...")
    embedding_function = get_embedding_model()
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_function,
        index_name=PINECONE_INDEX_NAME,
    )
    print("¡Carga a Pinecone completada!")


def limpiar_archivo_local():
    if os.path.exists(LOCAL_CV_PATH):
        os.remove(LOCAL_CV_PATH)
        print("Archivo temporal eliminado.")