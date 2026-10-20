import os
import requests
import sys
import uuid
from dotenv import load_dotenv
load_dotenv() 

# --- CONFIGURACIÓN GOOGLE DOCS ---
GOOGLE_DOC_ID = os.environ.get("GOOGLE_DOC_ID")
if not GOOGLE_DOC_ID:
    raise ValueError("GOOGLE_DOC_ID no encontrada en variables de entorno")

CV_RAW_URL = f"https://docs.google.com/document/d/{GOOGLE_DOC_ID}/export?format=txt"

CV_FILENAME = "matias_background.txt"
LOCAL_CV_PATH = "./matias_background.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Configuración de Pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "cv-matias")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY no encontrada en las variables de entorno")

def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    local_only = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu", "local_files_only": local_only},
        encode_kwargs={"normalize_embeddings": False},
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
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Total de chunks generados: {len(chunks)}")
    return chunks


def get_pinecone_index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def obtener_ids_existentes():
    """IDs de los vectores actuales, para borrarlos recién después de subir los nuevos."""
    index = get_pinecone_index()
    ids = [vector_id for batch in index.list() for vector_id in batch]
    print(f"Vectores existentes en el índice: {len(ids)}")
    return ids


def crear_y_almacenar_vectores(chunks):
    from langchain_pinecone import PineconeVectorStore
    if not chunks:
        raise ValueError("No se generaron chunks; no se toca el índice.")
    print(f"Subiendo {len(chunks)} vectores a Pinecone...")
    embedding_function = get_embedding_model()
    vectorstore = PineconeVectorStore(
        index=get_pinecone_index(),
        embedding=embedding_function,
    )
    vectorstore.add_documents(chunks, ids=[str(uuid.uuid4()) for _ in chunks])
    print("¡Carga a Pinecone completada!")


def borrar_vectores_viejos(ids_viejos):
    """Borra solo los vectores previos. Si falla, el índice queda con datos viejos y nuevos, nunca vacío."""
    if not ids_viejos:
        return
    print(f"Borrando {len(ids_viejos)} vectores antiguos...")
    index = get_pinecone_index()
    for i in range(0, len(ids_viejos), 1000):
        index.delete(ids=ids_viejos[i : i + 1000])
    print("Vectores antiguos eliminados.")


def limpiar_archivo_local():
    if os.path.exists(LOCAL_CV_PATH):
        os.remove(LOCAL_CV_PATH)
        print("Archivo temporal eliminado.")