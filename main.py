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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
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

def detect_language(text: str) -> str:
    """Devuelve 'en' o 'es' según la pregunta del usuario."""
    raw = _NAME_RE.sub(" ", (text or "").strip().lower()).strip()
    if not raw:
        return "es"
    compact = re.sub(r"[^a-záéíóúüñ\s]", " ", raw)
    if compact.strip() in {"hi", "hey", "hello", "yo", "sup", "hiya"}:
        return "en"
    if compact.strip() in {"hola", "holis", "buenas", "buen dia", "buen día"}:
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
@lru_cache(maxsize=1)
def get_embedding_model():
    """Embeddings locales (mismo modelo que Pinecone). Evita la Inference API de Hugging Face."""
    local_only = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu", "local_files_only": local_only},
        encode_kwargs={"normalize_embeddings": False},
    )

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
    """Recupera los chunks una sola vez: los usan el borrador y la verificación."""
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

def claude_llm(model_name: str):
    """Llama a Claude con el SDK oficial y devuelve solo el texto."""
    async def ask_claude(prompt) -> str:
        response = await get_anthropic_client().messages.create(
            model=model_name,
            # Incluye el razonamiento del modelo; con 800 la respuesta se cortaría
            max_tokens=4000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt.to_string()}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{model_name} rechazó la respuesta")
        return "".join(block.text for block in response.content if block.type == "text")

    return RunnableLambda(ask_claude)

@lru_cache(maxsize=4)
def get_rag_chain(model_name: str):
    """Construye el borrador y la verificación contra el contexto (Self-RAG)."""

    DRAFT_TEMPLATE = """
        You are Matías Rodríguez's AI assistant.
        Answer using ONLY the context. Be direct and professional.
        Maximum 3 sentences. No bullets.

        FACTS (mandatory):
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
    """

    REFINE_TEMPLATE = """
        You are the fact-checker and editor for Matías Rodríguez's chatbot.

        RULES:
        1. Check every claim in the draft against the CONTEXT. Remove any claim,
           detail, tool, company, date or number the CONTEXT does not state.
           Do not add anything that is not in the CONTEXT.
        2. If nothing in the draft is supported by the CONTEXT, say briefly that
           this information is not available about Matías.
        3. If the draft is cut off, complete it using only the CONTEXT.
        4. The user's question is in {language}. The final answer MUST be 100% in {language}.
           If the draft is in the wrong language, translate it.
        5. If the draft is redundant, compress it. Maximum 3 sentences. No bullets.
        6. Copy every date, year, number and name exactly as written in the CONTEXT.
           The CONTEXT is up to date: a recent year is correct, never replace it with an earlier one.
        7. Do not mention editing or checking. Return only the final text.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        DRAFT:
        {draft}

        FINAL ANSWER:
    """

    if model_name.startswith("claude"):
        chat = claude_llm(model_name)
    else:
        chat = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            # 2.5-flash cuenta su razonamiento dentro de este límite; con 800 cortaba respuestas
            max_tokens=4000,
            disable_streaming=True,
            # Ante un 429 conviene pasar al siguiente modelo de LLM_MODELS en vez de reintentar ~2 min
            max_retries=1,
        )

    # Ambas reciben {context, question, language}; la verificación además el {draft}
    draft_chain = PromptTemplate.from_template(DRAFT_TEMPLATE) | chat | StrOutputParser()
    verify_chain = PromptTemplate.from_template(REFINE_TEMPLATE) | chat | StrOutputParser()

    return verify_chain, draft_chain

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
        "https://matiasrodriguezc.github.io",
        "https://matiasrodriguezc.com.ar",
        "https://www.matiasrodriguezc.com.ar"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generate_answer(pregunta_texto: str, lang_code: str) -> str:
    payload = {
        "question": pregunta_texto,
        "language": language_name(lang_code),
    }
    last_error = None
    empty_fallback = (
        "I couldn't generate an answer. Please try again in a few seconds."
        if lang_code == "en"
        else "No pude generar una respuesta. Intentá de nuevo en unos segundos."
    )
    for model_name in LLM_MODELS:
        try:
            print(f"Intentando modelo: {model_name} (idioma={payload['language']})")
            if "context" not in payload:
                # El camino async de langchain-pinecone comparte una sesión entre requests:
                # con dos preguntas a la vez, todas menos una se cuelgan para siempre.
                payload["context"] = await asyncio.wait_for(
                    asyncio.to_thread(get_context_chain().invoke, pregunta_texto),
                    timeout=30,
                )
            verify_chain, draft_chain = get_rag_chain(model_name)
            draft = await draft_chain.ainvoke(payload)
            try:
                text = await verify_chain.ainvoke({**payload, "draft": draft})
            except Exception as verify_error:
                print(f"Verificación falló ({model_name}): {verify_error}. Uso el borrador.")
                text = draft
            if text and str(text).strip():
                return str(text).strip()
            print(f"Modelo {model_name} devolvió una respuesta vacía")
        except Exception as e:
            last_error = e
            print(f"Modelo {model_name} falló: {e}")
    if last_error:
        raise last_error
    return empty_fallback

async def stream_rag_response(pregunta_texto: str):
    """Genera la respuesta completa y la envía en chunks (evita el stream vacío de Gemini 3 + AFC)."""
    lang_code = detect_language(pregunta_texto)
    try:
        text = await generate_answer(pregunta_texto, lang_code)
        print(f"Respuesta ({len(text)} chars, {lang_code}): {text[:200]}")
        for i in range(0, len(text), 16):
            yield text[i : i + 16]
            await asyncio.sleep(0.01)
    except Exception as e:
        print(f"Error durante el streaming: {e}")
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
