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
from pinecone import Pinecone

# --- CONFIGURACIÓN ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "cv-matias")
PINECONE_INDEX_HOST = os.environ.get("PINECONE_INDEX_HOST") or os.environ.get("PINECONE_HOST")

# flash-latest suele ir a Gemini 3.x (streaming vacío + 503). Probar alternativas.
GEMINI_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_MODEL",
        "gemini-2.5-flash,gemini-flash-lite-latest,gemini-2.0-flash",
    ).split(",")
    if m.strip()
]

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
def get_pinecone_index():
    """Conecta al índice sin list_indexes() (ese endpoint da 403 desde Koyeb)."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_HOST:
        return pc.Index(host=PINECONE_INDEX_HOST)
    return pc.Index(PINECONE_INDEX_NAME)

@lru_cache(maxsize=4)
def get_rag_chain(model_name: str):
    """Construye y devuelve la cadena RAG con Self-Refinement."""
    embedding_function = get_embedding_model()
    vectorstore = PineconeVectorStore(
        index=get_pinecone_index(),
        embedding=embedding_function,
    )
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.7})

    DRAFT_TEMPLATE = """
        Eres el asistente de IA de Matías Rodríguez.
        Genera una respuesta DIRECTA y PROFESIONAL basada SOLO en el contexto.
        Máximo 3 oraciones. No uses bullets.

        Contexto:
        {context}

        Pregunta:
        {question}
    """

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
        model=model_name,
        temperature=0.2,
        max_tokens=800,
        disable_streaming=True,
    )

    draft_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | PromptTemplate.from_template(DRAFT_TEMPLATE)
        | chat
        | StrOutputParser()
    )

    refined_chain = (
        {"draft": draft_chain}
        | PromptTemplate.from_template(REFINE_TEMPLATE)
        | chat
        | StrOutputParser()
    )

    return refined_chain, draft_chain

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Solo MiniLM. Pinecone/Gemini se conectan en el primer /ask para no tumbar el health check.
    get_embedding_model()
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

async def generate_answer(pregunta_texto: str) -> str:
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            print(f"Intentando modelo: {model_name}")
            refined_chain, draft_chain = get_rag_chain(model_name)
            try:
                text = await refined_chain.ainvoke(pregunta_texto)
            except Exception as refine_error:
                print(f"Refine falló ({model_name}): {refine_error}. Uso el borrador.")
                text = await draft_chain.ainvoke(pregunta_texto)
            if text and str(text).strip():
                return str(text).strip()
            print(f"Modelo {model_name} devolvió una respuesta vacía")
        except Exception as e:
            last_error = e
            print(f"Modelo {model_name} falló: {e}")
    if last_error:
        raise last_error
    return "No pude generar una respuesta. Intentá de nuevo en unos segundos."

async def stream_rag_response(pregunta_texto: str):
    """Genera la respuesta completa y la envía en chunks (evita el stream vacío de Gemini 3 + AFC)."""
    try:
        text = await generate_answer(pregunta_texto)
        print(f"Respuesta ({len(text)} chars): {text[:200]}")
        for i in range(0, len(text), 16):
            yield text[i : i + 16]
            await asyncio.sleep(0.01)
    except Exception as e:
        print(f"Error durante el streaming: {e}")
        yield "Lo siento, ocurrió un error al procesar la respuesta. Probá de nuevo en unos segundos."

@app.post("/ask")
async def ask_cv(pregunta: Pregunta):
    print(f"Recibida pregunta (stream): {pregunta.texto}")
    return StreamingResponse(
        stream_rag_response(pregunta.texto),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/")
def root():
    return {"mensaje": "API del Chatbot de CV está funcionando. Usa el endpoint /ask"}
