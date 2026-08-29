import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi.responses import StreamingResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIGURACIÓN ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Configuración de Pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "cv-matias")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY no encontrada. Asegúrate de que esté en .env o en las variables de entorno.")

# --- Modelos Pydantic ---
class Pregunta(BaseModel):
    texto: str

# --- Lógica de Carga ---
@lru_cache(maxsize=1)
def get_embedding_model():
    """Embeddings locales (mismo modelo que Pinecone). Evita la Inference API de Hugging Face."""
    local_only = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu", "local_files_only": local_only},
        encode_kwargs={"normalize_embeddings": False},
    )

@lru_cache(maxsize=1)
def get_rag_chain():
    """Construye y devuelve la cadena RAG con Self-Refinement."""
    embedding_function = get_embedding_model()
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding_function,
    )
    
    retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k':5, 'lambda_mult':0.7})
    
    # 1. Draft Prompt
    DRAFT_TEMPLATE = """
        Eres el asistente de IA de Matías Rodríguez.
        Genera una respuesta DIRECTA y PROFESIONAL basada SOLO en el contexto.
        Máximo 3 oraciones. No uses bullets.

        Contexto:
        {context}

        Pregunta:
        {question}
    """
    
    # 2. Refinement Prompt (Self-RAG)
    REFINE_TEMPLATE = """
        Eres un editor experto. Tu tarea es pulir la respuesta de un chatbot para Matías Rodríguez.
        
        REGLAS DE REFINAMIENTO:
        1. Si la respuesta está cortada, complétala de forma lógica.
        2. Asegúrate de que detecte el idioma correcto (Español o Inglés).
        3. Si la respuesta es redundante, comprímela.
        4. No menciones el proceso de edición ("Aquí está la versión pulida", etc), solo devuelve el texto final.

        RESPUESTA A EVALUAR:
        {draft}

        VERSION FINAL PULIDA:
    """

    chat = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    temperature=0.2,
                    max_tokens = 800
            )

    # Cadena de Generación Inicial
    draft_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | PromptTemplate.from_template(DRAFT_TEMPLATE)
        | chat
        | StrOutputParser()
    )

    # Cadena de Refinamiento (Self-RAG)
    # Nota: Aquí encadenamos el borrador al segundo paso
    refined_chain = (
        {"draft": draft_chain}
        | PromptTemplate.from_template(REFINE_TEMPLATE)
        | chat
        | StrOutputParser()
    )
    
    return refined_chain

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_rag_chain()
    yield

app = FastAPI(title="Chatbot de CV", lifespan=lifespan)

# --- CONFIGURACIÓN DE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://matiasrodriguezc.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoint de Streaming ---
async def stream_rag_response(pregunta_texto: str):
    """Generador asíncrono para la respuesta del chat."""
    try:
        rag_chain = get_rag_chain()
        # .astream() es la versión asíncrona de .stream()
        # Esto devuelve los "chunks" (palabras/tokens) a medida que el LLM los genera
        async for chunk in rag_chain.astream(pregunta_texto):
            yield chunk
            await asyncio.sleep(0.01) # Pequeña pausa para el "efecto" de streaming
    except Exception as e:
        print(f"Error durante el streaming: {e}")
        yield "Lo siento, ocurrió un error al procesar la respuesta."

@app.post("/ask")
async def ask_cv(pregunta: Pregunta):
    """
    Recibe una pregunta y devuelve una respuesta en streaming.
    """
    print(f"Recibida pregunta (stream): {pregunta.texto}")
    # Devuelve un StreamingResponse que consume el generador
    return StreamingResponse(
        stream_rag_response(pregunta.texto), 
        media_type="text/event-stream"
    )

@app.get("/")
def root():
    return {"mensaje": "API del Chatbot de CV está funcionando. Usa el endpoint /ask"}