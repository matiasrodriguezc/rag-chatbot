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
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

from fastapi.middleware.cors import CORSMiddleware


# --- CONFIGURACIÓN ---
# La API (local) lee la DB de la carpeta 'dags'
CHROMA_DIR = "./dags/cv_vector_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

app = FastAPI(title="Chatbot de CV")

# --- CONFIGURACIÓN DE CORS ---
# Esto es VITAL para que tu portafolio pueda llamar a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://matiasrodriguezc.github.io" # <-- ¡Permite tu portafolio!
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic ---
class Pregunta(BaseModel):
    texto: str

# --- Lógica de Carga (Lazy Loading) ---
# Mantenemos las funciones de carga separadas para no alentar el inicio
def get_embedding_model():
    """Usa la API de Inferencia de Hugging Face (Clase Correcta)."""
    # Importamos la clase correcta DESDE el nuevo paquete
    from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN no encontrada. Asegúrate de que esté en .env o en las variables de entorno.")

    return HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,               # Esta clase SÍ acepta 'api_key'
        model_name=EMBEDDING_MODEL_NAME, # Esta clase usa 'model_name'
        api_url="https://router.huggingface.co/hf-inference/" # Esta clase usa 'api_url'
    )

def get_rag_chain():
    """Construye y devuelve la cadena RAG completa."""
    embedding_function = get_embedding_model()
    
    vectorstore = Chroma(
                    persist_directory = CHROMA_DIR, 
                    embedding_function = embedding_function)
    
    retriever = vectorstore.as_retriever(search_type = 'mmr', search_kwargs = {'k':5, 'lambda_mult':0.7})

    TEMPLATE = '''
    Responde la pregunta del usuario de forma breve y concisa, usando únicamente el contexto proporcionado.
    Tu respuesta debe ir directo al grano.
    NO saludes (No digas "Hola").
    NO uses frases introductorias (No digas "Basándome en el contexto..." o "Aquí tienes...").
    Si la respuesta es una lista, enumérala directamente.
    Si la respuesta no está en el contexto, di "Esa información no se encuentra en el CV de Matías. No puedo responder a eso."

    Contexto:
    {context}

    Pregunta:
    {question}
    '''
    prompt_template = PromptTemplate.from_template(TEMPLATE)

    chat = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.1,
                    max_tokens = 500
                    # No necesitamos los safety_settings si funcionó sin ellos
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