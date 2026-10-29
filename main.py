import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel
import os
import time
from dotenv import load_dotenv
load_dotenv()

from fastapi.responses import StreamingResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastembed import TextEmbedding
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
import re

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

# Claude va al final: solo responde si los modelos de Gemini fallan o se quedan sin cuota
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
LLM_MODELS = GEMINI_MODELS + ([CLAUDE_MODEL] if os.environ.get("ANTHROPIC_API_KEY") else [])

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY no encontrada. Asegúrate de que esté en .env o en las variables de entorno.")

_EN_HINTS = {
    "hi", "hello", "hey", "how", "what", "who", "where", "when", "why", "which",
    "tell", "about", "experience", "experiences", "skill", "skills", "project",
    "projects", "background", "resume", "work", "job", "please", "thanks",
    "thank", "your", "you", "the", "is", "are", "his", "he", "can", "does",
    "did", "was", "were", "with", "from", "this", "that", "have", "has",
}
_ES_HINTS = {
    "hola", "buenas", "buen", "qué", "que", "quién", "quien", "cómo", "como",
    "dónde", "donde", "cuándo", "cuando", "cuál", "cual", "experiencia",
    "habilidades", "proyectos", "trabajo", "gracias", "decime", "conta",
    "contame", "sobre", "del", "una", "unos", "unas", "los", "las", "por",
    "para", "su", "sus", "el", "la", "de", "en", "es", "con",
}
# Las tildes del nombre propio no indican idioma ("Where does Matías work?" es inglés)
_NAME_RE = re.compile(r"\b(mat[ií]as|rodr[ií]guez|c[aá]rdenas)\b")

_EN_GREETINGS = {
    "hi", "hey", "hello", "yo", "sup", "hiya", "hi there", "hello there",
    "good morning", "good afternoon", "good evening",
}
_ES_GREETINGS = {
    "hola", "holis", "buenas", "buen dia", "buen día", "buenos dias", "buenos días",
    "buenas tardes", "buenas noches", "hola que tal", "hola qué tal",
}

def _without_name(text: str) -> str:
    return _NAME_RE.sub(" ", (text or "").strip().lower()).strip()

def _only_words(raw: str) -> str:
    return " ".join(re.sub(r"[^a-záéíóúüñ\s]", " ", raw).split())

def detect_language(text: str) -> str:
    """Devuelve 'en' o 'es' según la pregunta del usuario."""
    raw = _without_name(text)
    if not raw:
        return "es"
    compact = _only_words(raw)
    if compact in _EN_GREETINGS:
        return "en"
    if compact in _ES_GREETINGS:
        return "es"
    if re.search(r"[áéíóúüñ¿¡]", raw):
        return "es"
    words = [w for w in compact.split() if w]
    en_score = sum(1 for w in words if w in _EN_HINTS)
    es_score = sum(1 for w in words if w in _ES_HINTS)
    if en_score > es_score:
        return "en"
    if es_score > en_score:
        return "es"
    return "en" if re.fullmatch(r"[a-z\s']+", compact.strip()) else "es"

def greeting_reply(text: str, lang_code: str):
    """Un saludo solo no necesita Pinecone ni el LLM: se contesta al instante."""
    if _only_words(_without_name(text)) not in _EN_GREETINGS | _ES_GREETINGS:
        return None
    if lang_code == "en":
        return "Hi! I'm Matías's assistant. Ask me about his experience, projects or skills."
    return "¡Hola! Soy el asistente de Matías. Preguntame por su experiencia, proyectos o habilidades."

def language_name(code: str) -> str:
    return "English" if code == "en" else "Spanish"

def user_error_message(lang_code: str) -> str:
    if lang_code == "en":
        return "Sorry, something went wrong while answering. Please try again in a few seconds."
    return "Lo siento, ocurrió un error al procesar la respuesta. Probá de nuevo en unos segundos."

# --- Modelos Pydantic ---
class Pregunta(BaseModel):
    texto: str

# --- Lógica de Carga ---
class MiniLMEmbeddings(Embeddings):
    """MiniLM en ONNX con fastembed: los mismos vectores que sentence-transformers, sin torch.
    Con torch la imagen tardaba ~1 min en arrancar y casi no entraba en la RAM de Koyeb."""

    def __init__(self):
        # Un solo hilo: con 0.1 vCPU los hilos de onnxruntime agotan la cuota y frenan todo el proceso
        self._model = TextEmbedding(
            EMBEDDING_MODEL_NAME,
            cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
            threads=1,
        )

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self._model.embed(list(texts))]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

@lru_cache(maxsize=1)
def get_embedding_model():
    """Embeddings locales (mismo modelo que Pinecone). Evita la Inference API de Hugging Face."""
    return MiniLMEmbeddings()

def _pinecone_host() -> str:
    raw = (PINECONE_INDEX_HOST or "").strip().strip('"').strip("'")
    raw = raw.replace("https://", "").replace("http://", "").rstrip("/")
    if not raw:
        raise ValueError(
            "PINECONE_INDEX_HOST no está definida. "
            "Koyeb no puede usar el control plane de Pinecone (403 en /indexes). "
            "En la consola de Pinecone, abrí el índice cv-matias, copiá Host "
            "(algo como cv-matias-xxxx.svc.xxx.pinecone.io) y agregala en Koyeb como PINECONE_INDEX_HOST."
        )
    return raw

@lru_cache(maxsize=1)
def get_pinecone_index():
    """Conecta al data plane por host. Evita describe/list_indexes (403 desde Koyeb)."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(host=_pinecone_host())

def format_docs(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

@lru_cache(maxsize=1)
def get_context_chain():
    """Recupera los chunks de Pinecone y los une en un solo texto de contexto."""
    embedding_function = get_embedding_model()
    vectorstore = PineconeVectorStore(
        index=get_pinecone_index(),
        embedding=embedding_function,
    )
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.7})
    return retriever | format_docs

@lru_cache(maxsize=1)
def get_anthropic_client():
    from anthropic import AsyncAnthropic
    # Un solo reintento, igual que con Gemini: es el último respaldo.
    # Sin timeout el SDK espera hasta 10 min por llamada y el chat queda mudo.
    return AsyncAnthropic(max_retries=1, timeout=60)

async def stream_claude(model_name: str, prompt: str):
    """Llama a Claude con el SDK oficial y va devolviendo el texto a medida que llega."""
    async with get_anthropic_client().messages.stream(
        model=model_name,
        # Incluye el razonamiento del modelo; con 800 la respuesta se cortaría
        max_tokens=4000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text

# Una sola llamada al LLM: antes había un borrador y una verificación en serie, que duplicaban la espera.
# Las reglas de la verificación (nada que no esté en el contexto) ahora van en este mismo prompt.
ANSWER_PROMPT = PromptTemplate.from_template("""
    You are Matías Rodríguez's AI assistant.
    Answer the question using ONLY the context. Be direct and professional.
    Maximum 3 sentences. No bullets.

    FACTS (mandatory):
    - Every claim must be stated in the context. Never add a tool, company, date,
      number or detail the context does not mention.
    - If the context does not answer the question, say briefly that you don't
      have that information about Matías.
    - If the user is only making small talk, reply in one friendly sentence and
      invite them to ask about Matías's experience, projects or skills.
    - Never mention the context, documents or sources.
    - Copy every date, year, number and name exactly as written in the context.
    - The context is up to date: a recent year is correct, never replace it with an earlier one.

    LANGUAGE (mandatory):
    - The user wrote in {language}.
    - Write the ENTIRE answer in {language}. Do not mix languages.
    - If {language} is English, use no Spanish.
    - If {language} is Spanish, use no English.

    Context:
    {context}

    Question:
    {question}
""")

@lru_cache(maxsize=4)
def get_gemini_chain(model_name: str):
    chat = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        # 2.5-flash cuenta su razonamiento dentro de este límite; con 800 cortaba respuestas
        max_tokens=4000,
        # Para 3 oraciones sacadas del contexto el razonamiento solo suma segundos
        thinking_budget=0 if "2.5-flash" in model_name else None,
        # Ante un 429 conviene pasar al siguiente modelo de LLM_MODELS en vez de reintentar ~2 min
        max_retries=1,
    )
    return ANSWER_PROMPT | chat | StrOutputParser()

def stream_model(model_name: str, payload: dict):
    if model_name.startswith("claude"):
        return stream_claude(model_name, ANSWER_PROMPT.format(**payload))
    return get_gemini_chain(model_name).astream(payload)

def warm_up():
    """Conecta Pinecone y hace una búsqueda de prueba para que la primera pregunta no pague el arranque."""
    try:
        get_context_chain().invoke("experiencia laboral")
        print("Pinecone listo")
    except Exception as e:
        print(f"Precalentamiento de Pinecone falló: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # MiniLM antes de aceptar pedidos; Pinecone en segundo plano para no demorar el health check.
    get_embedding_model()
    warm_up_task = asyncio.create_task(asyncio.to_thread(warm_up))
    yield
    warm_up_task.cancel()

app = FastAPI(title="Chatbot de CV", lifespan=lifespan)

# --- CONFIGURACIÓN DE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://matiasrodriguezc.github.io",
        "https://matiasrodriguezc.com.ar",
        "https://www.matiasrodriguezc.com.ar"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def stream_answer(pregunta_texto: str, lang_code: str):
    """Manda el texto apenas el modelo lo genera. Si un modelo falla antes de escribir algo, prueba el siguiente."""
    payload = {
        "question": pregunta_texto,
        "language": language_name(lang_code),
    }
    # El camino async de langchain-pinecone comparte una sesión entre requests:
    # con dos preguntas a la vez, todas menos una se cuelgan para siempre.
    start = time.perf_counter()
    payload["context"] = await asyncio.wait_for(
        asyncio.to_thread(get_context_chain().invoke, pregunta_texto),
        timeout=30,
    )
    print(f"Contexto de Pinecone en {time.perf_counter() - start:.1f}s")
    last_error = None
    for model_name in LLM_MODELS:
        sent = False
        try:
            print(f"Intentando modelo: {model_name} (idioma={payload['language']})")
            async for piece in stream_model(model_name, payload):
                if piece:
                    if not sent:
                        print(f"Primer texto de {model_name} a los {time.perf_counter() - start:.1f}s")
                    sent = True
                    yield piece
            if sent:
                print(f"Respuesta completa a los {time.perf_counter() - start:.1f}s")
                return
            print(f"Modelo {model_name} devolvió una respuesta vacía")
        except Exception as e:
            # Con parte de la respuesta ya enviada no se puede empezar de nuevo con otro modelo
            if sent:
                raise
            last_error = e
            print(f"Modelo {model_name} falló: {e}")
    if last_error:
        raise last_error
    yield (
        "I couldn't generate an answer. Please try again in a few seconds."
        if lang_code == "en"
        else "No pude generar una respuesta. Intentá de nuevo en unos segundos."
    )

async def stream_rag_response(pregunta_texto: str):
    lang_code = detect_language(pregunta_texto)
    greeting = greeting_reply(pregunta_texto, lang_code)
    if greeting:
        yield greeting
        return
    sent = []
    try:
        async for piece in stream_answer(pregunta_texto, lang_code):
            sent.append(piece)
            yield piece
        text = "".join(sent)
        print(f"Respuesta ({len(text)} chars, {lang_code}): {text[:200]}")
    except Exception as e:
        print(f"Error durante el streaming: {e}")
        if not sent:
            yield user_error_message(lang_code)

@app.post("/ask")
async def ask_cv(pregunta: Pregunta):
    print(f"Recibida pregunta (stream): {pregunta.texto} [lang={detect_language(pregunta.texto)}]")
    return StreamingResponse(
        stream_rag_response(pregunta.texto),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/")
def root():
    return {"mensaje": "API del Chatbot de CV está funcionando. Usa el endpoint /ask"}
