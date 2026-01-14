import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import sys
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

app = FastAPI(title="Chatbot de CV")

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

# --- Modelos Pydantic ---
class Pregunta(BaseModel):
    texto: str

# --- Lógica de Carga ---
def get_embedding_model():
    """Usa la API de Inferencia de Hugging Face (Clase y Parámetros Correctos)."""
    
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN no encontrada. Asegúrate de que esté en .env o en las variables de entorno.")
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=hf_token,  
        model=EMBEDDING_MODEL_NAME
    )

def get_rag_chain():
    """Construye y devuelve la cadena RAG completa."""
    embedding_function = get_embedding_model()
    
    # Conectar a Pinecone usando el índice existente
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding_function,
    )
    
    retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k':5, 'lambda_mult':0.7})
    TEMPLATE = """
        Eres el asistente de IA personal de Matías Rodríguez.
        
        Tu objetivo: Responder a la pregunta del usuario de forma **HUMANA, NATURAL y DIRECTA**.
        
        Reglas de Estilo:
        1. **Concisión:** No cuentes la historia de su vida si no te la piden. Ve al grano.
        2. **Longitud:** Tus respuestas deben tener idealmente entre **2 y 4 oraciones**. Máximo 1 párrafo breve.
        3. **Formato:** NO uses listas, ni bullets (*), ni introducciones robóticas como "La información indica que...". Habla como una persona normal.
        4. **Idioma:** Responde SIEMPRE en el mismo idioma que el usuario (Español o Inglés).
        
        Si la respuesta no está en el contexto, dilo brevemente.

        Contexto sobre Matías:
        {context}

        Pregunta del usuario:
        {question}
    """
    prompt_template = PromptTemplate.from_template(TEMPLATE)

    chat = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.2,
                    max_tokens = 800
            )

    chain = ({'context': retriever, 
            'question': RunnablePassthrough()} 
            | prompt_template 
            | chat 
            | StrOutputParser())
    
    return chain

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