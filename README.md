# RAG Chatbot para CV Personal

Este proyecto es un chatbot de IA conversacional diseñado para responder preguntas sobre la experiencia profesional y académica de una persona, basándose en la información de su currículum. Utiliza un enfoque de Generación Aumentada por Recuperación (RAG) para proporcionar respuestas precisas y contextualmente relevantes en un formato de conversación natural.

El chatbot está construido con FastAPI y LangChain, y está diseñado para ser desplegado como una API que puede ser integrada en cualquier sitio web, como un portafolio personal.

## ✨ Características

- **Generación Aumentada por Recuperación (RAG):** El chatbot utiliza un pipeline RAG para asegurar que las respuestas se basen exclusivamente en la información del CV proporcionado, minimizando la posibilidad de alucinaciones o información incorrecta.
- **API basada en FastAPI:** La lógica del chatbot está expuesta a través de una API de FastAPI, lo que permite una fácil integración con aplicaciones web y otros sistemas.
- **Streaming de Respuestas:** Las respuestas del chatbot se transmiten en tiempo real, palabra por palabra, para una experiencia de usuario más dinámica y atractiva.
- **Soporte Multilingüe (Español/Inglés):** El chatbot está diseñado para responder en el mismo idioma en el que se le pregunta (español o inglés).
- **Pipeline de ETL Automatizado:** Incluye un pipeline de ETL (Extract, Transform, Load) para procesar el CV desde un Google Doc, crear los embeddings y cargarlos en una base de datos vectorial.
- **Despliegue con Docker:** El proyecto está completamente containerizado con Docker, lo que facilita su despliegue en cualquier plataforma que soporte contenedores.

## 🚀 Stack Tecnológico

- **Backend:** FastAPI, Python 3.11
- **Framework de IA:** LangChain
- **Modelos de Lenguaje (LLM):** Google Gemini (gemini-2.5-flash)
- **Modelos de Embeddings:** Hugging Face Sentence Transformers (all-MiniLM-L6-v2) a través de Inference Endpoints.
- **Base de Datos Vectorial:** Pinecone
- **Orquestación de ETL:** El pipeline de ETL está diseñado para ser ejecutado con Apache Airflow, aunque puede ser ejecutado manualmente.
- **Despliegue:** Docker, Uvicorn

## ⚙️ ¿Cómo Funciona? (Arquitectura RAG)

El chatbot sigue una arquitectura de Generación Aumentada por Recuperación (RAG) para generar sus respuestas:

1.  **Pipeline de ETL (Offline):**
    *   **Extracción:** El contenido del CV se descarga desde un Google Doc.
    *   **Transformación:** El texto se divide en fragmentos más pequeños (chunks).
    *   **Carga:** Cada chunk se convierte en un vector numérico (embedding) y se almacena en un índice de Pinecone. Este proceso se realiza una sola vez o cada vez que el CV se actualiza.

2.  **Inferencia en Tiempo Real (Online):**
    *   **Pregunta del Usuario:** El usuario envía una pregunta a la API.
    *   **Creación de Embedding:** La pregunta del usuario también se convierte en un embedding.
    *   **Búsqueda de Similitud:** El sistema busca en Pinecone los chunks de texto del CV cuyos embeddings son más similares al embedding de la pregunta.
    *   **Generación de Respuesta:** Los chunks recuperados (el contexto) y la pregunta original se envían al LLM (Google Gemini) a través de un prompt cuidadosamente diseñado. El LLM genera una respuesta en lenguaje natural basada únicamente en el contexto proporcionado.

## 🏁 Instalación y Uso Local

### Prerrequisitos

- Python 3.11
- Docker
- Una cuenta de Pinecone y una clave de API.
- Una cuenta de Hugging Face y un token de acceso.

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/rag-chatbot.git
cd rag-chatbot
```

### 2. Configurar las Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto y añade las siguientes variables:

```
PINECONE_API_KEY="TU_API_KEY_DE_PINECONE"
PINECONE_INDEX_NAME="cv-matias" # O el nombre de tu índice
HF_TOKEN="TU_TOKEN_DE_HUGGING_FACE"
```

### 3. Ejecutar el Pipeline de ETL

Este paso es necesario para poblar la base de datos de Pinecone con la información de tu CV.

Primero, instala las dependencias del pipeline:

```bash
pip install -r requirements.txt
```

Luego, ejecuta el script del pipeline:

```bash
python dags/pipeline/run_etl.py
```

### 4. Ejecutar la API Localmente

Instala las dependencias de la API:

```bash
pip install -r requirements-api.txt
```

Inicia el servidor de FastAPI:

```bash
uvicorn main:app --reload
```

La API estará disponible en `http://localhost:8000`.

## 🐳 Despliegue con Docker

El proyecto incluye un `Dockerfile` para construir una imagen de contenedor para la API.

### 1. Construir la Imagen de Docker

```bash
docker build -t rag-chatbot-api .
```

### 2. Ejecutar el Contenedor

```bash
docker run -d -p 8000:8000 \
  --env-file .env \
  rag-chatbot-api
```

El contenedor expondrá la API en el puerto 8000 de tu máquina local.

## 📂 Estructura del Proyecto

```
.
├── .github/workflows/   # Workflows de GitHub Actions
├── dags/                # Directorio para pipelines de Airflow
│   └── pipeline/
│       ├── cv_etl.py    # Lógica principal del pipeline de ETL
│       └── run_etl.py   # Script para ejecutar el pipeline manualmente
├── .gitignore
├── Dockerfile           # Define la imagen de Docker para la API
├── main.py              # Lógica principal de la API de FastAPI
├── README.md            # Este archivo
├── requirements-api.txt # Dependencias de Python para la API
└── requirements.txt     # Dependencias de Python para el pipeline de ETL
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para discutir los cambios que te gustaría hacer.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.