from fastapi import FastAPI
from pydantic import BaseModel
import sys
# Carga las variables de entorno (GOOGLE_API_KEY)
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Usamos la misma configuración del ETL
CHROMA_DIR = "./dags/cv_vector_db"
EMBEDDING_MODEL = "models/embedding-001"

# 1. Inicializa la App
app = FastAPI(title="Chatbot de CV")

# 2. Define los modelos de entrada/salida
class Pregunta(BaseModel):
    texto: str
class Respuesta(BaseModel):
    respuesta: str

# 3. Carga la cadena RAG (lógica de tu 'generator.py')
def get_rag_chain():
    # Se conecta a la DB de Chroma creada por el ETL
    vectorstore = Chroma(
                    persist_directory = CHROMA_DIR, 
                    embedding_function = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL))
    
    # Define el Retriever
    retriever = vectorstore.as_retriever(
                            search_type = 'mmr', 
                            search_kwargs = {'k':5, 'lambda_mult':0.7})

    # El Prompt (plantilla)
    TEMPLATE = '''
    Eres un asistente de reclutamiento profesional. Responde la pregunta del usuario 
    basándote únicamente en el siguiente contexto, que es el CV de Matías.
    El contexto puede estar en inglés o español, responde en el idioma de la pregunta.
    Sé amable y profesional. Si la respuesta no está en el contexto, di 
    "Esa información no se encuentra en el CV de Matías."

    Contexto:
    {context}

    Pregunta:
    {question}
    '''
    prompt_template = PromptTemplate.from_template(TEMPLATE)

    # El modelo de Chat (Gemini)
    chat = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash-latest", 
                    temperature=0,
                    max_tokens = 500)

    # La cadena RAG (chain)
    chain = ({'context': retriever, 
            'question': RunnablePassthrough()} 
            | prompt_template 
            | chat 
            | StrOutputParser())
    
    return chain

# 4. Crea el Endpoint del API
@app.post("/ask", response_model=Respuesta)
async def ask_cv(pregunta: Pregunta):
    """
    Recibe una pregunta sobre el CV y devuelve una respuesta generada por RAG.
    """
    try:
        print(f"Recibida pregunta: {pregunta.texto}")
        rag_chain = get_rag_chain()
        respuesta_generada = rag_chain.invoke(pregunta.texto)
        print(f"Respuesta generada: {respuesta_generada}")
        return Respuesta(respuesta=respuesta_generada)
    except Exception as e:
        print(f"Error en RAG chain: {e}")
        return Respuesta(respuesta="Error al procesar la solicitud.")

# 5. Endpoint raíz
@app.get("/")
def root():
    return {"mensaje": "API del Chatbot de CV está funcionando. Usa el endpoint /ask"}