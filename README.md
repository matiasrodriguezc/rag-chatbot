# 🤖 AI Portfolio Chatbot with Automated RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green?logo=chainlink&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

This project is a **Conversational AI Assistant** designed to answer questions about my professional experience, skills, and projects based on a living knowledge base.

Unlike traditional RAG chatbots that read static PDFs, this system implements a **CI/CD Pipeline** that automatically synchronizes a Vector Database (Pinecone) with a **Google Doc** in real-time. This allows the portfolio to remain constantly up-to-date without redeploying code.

## ✨ Key Features

- **🧠 RAG (Retrieval-Augmented Generation):** Generates accurate answers by retrieving context solely from my curated bio, minimizing hallucinations.
- **🔄 "Live Resume" Architecture:** The knowledge base is not hardcoded. A GitHub Actions workflow automatically fetches the latest data from Google Docs, processes it, and updates Pinecone.
- **⚡ Streaming Responses:** Provides a fluid user experience by streaming the LLM response token-by-token (Server-Sent Events), similar to ChatGPT.
- **🌐 Multilingual Support:** Automatically detects the user's language and responds fluently in either English or Spanish.
- **🐳 Dockerized:** Fully containerized application ready for deployment on serverless platforms like Koyeb, Render, or AWS ECS.

## 🚀 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend** | `FastAPI` | High-performance async API framework. |
| **AI Orchestration** | `LangChain` | RAG chain management and prompt engineering. |
| **LLM** | `Google Gemini 2.5 Flash` | Fast and efficient generative model. |
| **Vector DB** | `Pinecone` | Semantic search and vector storage. |
| **Embeddings** | `Hugging Face` | `all-MiniLM-L6-v2` model via Inference Endpoints. |
| **ETL Pipeline** | `GitHub Actions` + `Python` | Automated extraction from Google Docs to Pinecone. |
| **Infrastructure** | `Docker` | Lightweight containerization. |

## ⚙️ System Architecture

1.  **Data Ingestion (ETL):** A Python script (`cv_etl.py`) downloads the latest biography from a specific Google Doc ID, splits the text into semantic chunks, and generates embeddings.
2.  **Storage:** The vectors are stored in **Pinecone**, tagged with metadata.
3.  **Inference (Online):**
    * User sends a question via the API.
    * The system converts the question into a vector.
    * It retrieves the most relevant text chunks from Pinecone.
    * A custom Prompt is built with this context.
    * **Gemini** generates the final response in a natural, conversational style.

## 🏁 Local Installation & Usage

### Prerequisites
* Python 3.11+
* Docker (Optional)
* API Keys: Pinecone, Hugging Face Token, Google Gemini API Key.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/rag-chatbot.git](https://github.com/your-username/rag-chatbot.git)
cd rag-chatbot

```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```ini
PINECONE_API_KEY="your_pinecone_key"
PINECONE_INDEX_NAME="cv-matias"
HF_TOKEN="your_huggingface_token"
GOOGLE_API_KEY="your_google_gemini_key"
GOOGLE_DOC_ID="your_google_doc_id" # Only needed for running ETL locally

```

### 3. Load the Knowledge Base (ETL)

Before chatting, you need to populate the database:

```bash
# Install ETL dependencies
pip install -r requirements.txt

# Run the pipeline script
python dags/pipeline/run_etl.py

```

### 4. Run the API Server

```bash
# Install API dependencies
pip install -r requirements-api.txt

# Start server with Hot-Reload
uvicorn main:app --reload

```

The API will be available at `http://localhost:8000/docs`.

## 🐳 Docker Deployment

The project includes an optimized `Dockerfile` for production.

```bash
# Build the image
docker build -t rag-chatbot-api .

# Run the container
docker run -d -p 8000:8000 --env-file .env rag-chatbot-api

```

## 📂 Project Structure

```text
.
├── .github/workflows/   # CI/CD: Automated Pipeline (Google Docs -> Pinecone)
├── dags/pipeline/       # ETL Scripts
│   ├── cv_etl.py        # Core logic for processing Google Docs
│   └── run_etl.py       # Execution entry point
├── main.py              # FastAPI Application & RAG Logic
├── Dockerfile           # Optimized container configuration
├── requirements.txt     # Dependencies for ETL Pipeline
└── requirements-api.txt # Dependencies for API (Production)

```

## 📄 License

This project is licensed under the MIT License.
