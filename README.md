# AI Chatbot with Automated ETL Pipeline (RAG + Data Engineering)

This is a "Chat with your CV" project, built to demonstrate a professional, end-to-end architecture that combines **AI Engineering** (RAG) with robust **Data Engineering** (automated ETL).

The chatbot answers questions based on my resume. But more importantly, the system automatically updates its own knowledge base whenever I push a new version of my CV to GitHub.

## 🏛️ Architecture

This project is built on a decoupled, event-driven architecture, just as you would in a production environment.

1.  **Document Source (GitHub):** My CVs (in PDF format) are stored in this GitHub repository. This acts as the "single source of truth."
2.  **ETL Orchestrator (Apache Airflow):** An Airflow DAG runs on a schedule. It uses an `HttpSensor` to check if the CV files on GitHub have changed (by monitoring their `ETag`).
3.  **ETL Pipeline (Python Tasks):** If a change is detected, the pipeline automatically triggers:
    * **Extract:** Downloads the new PDF from GitHub.
    * **Transform:** Cleans the text, splits it into semantic chunks, and generates vector embeddings using the Google Gemini API.
    * **Load:** Deletes the *old* vectors from the database and inserts the *new* ones.
4.  **Vector Database (ChromaDB):** A persistent vector database that stores the CV chunks and their embeddings.
5.  **AI Chatbot API (FastAPI):** A separate Python server that provides a RAG (Retrieval-Augmented Generation) endpoint.
    * It receives a user's question (e.g., "What is your experience with Python?").
    * It generates an embedding for the *question*.
    * It **Retrieves** the most relevant CV chunks from ChromaDB.
    * It **Augments** a prompt with this context and sends it to the Gemini LLM.
    * It **Generates** and returns the final answer.

This decoupled design means the **Chatbot API** (`main.py`) and the **ETL Pipeline** (`dags/`) are two separate services that only communicate via the database.

## 🛠️ Technology Stack

* **Orchestration:** Apache Airflow (running via Docker)
* **AI Backend:** FastAPI
* **LLM & Embeddings:** Google Gemini (via Google AI Studio API)
* **RAG & ETL Logic:** LangChain (using `PyPDFLoader`, `CharacterTextSplitter`)
* **Vector Database:** ChromaDB
* **Containerization:** Docker & Docker Compose

## 🚀 How to Run

### 1. Prerequisites
* [Docker](https://www.docker.com/get-started) and Docker Compose
* Python 3.11+
* A Google Gemini API Key (from [Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Clone & Setup
1.  Clone this repository.
2.  Create a `.env` file in the project root. Add your API key:
    ```
    GOOGLE_API_KEY="AIzaSy...your...key...here"
    ```
3.  Ensure your `Dockerfile` and `requirements.txt` (for Docker) are present.

### 3. Run the Airflow Pipeline
This command builds the custom Airflow image (with all Python dependencies) and starts all Airflow services (scheduler, worker, webserver).

```bash
# Build the custom image
docker-compose build --no-cache

# Start all services in the background
docker-compose up -d
```

### 4. Run the Chatbot API
This server runs locally and answers user questions.

1.  (In a new terminal) Create and activate a local virtual environment:
    ```bash
    python3 -m venv venv_app
    source venv_app/bin/activate
    ```
2.  Install the API dependencies:
    ```bash
    pip install -r requirements-api.txt
    ```
3.  Start the API server:
    ```bash
    # --reload-exclude prevents the server from restarting every time Airflow writes a log
    uvicorn main:app --reload --reload-exclude "dags/*"
    ```

### 5. First-Time Database Population
The API is running, but its database is empty. We need to run the Airflow pipeline *once* manually.

1.  Open the Airflow UI at `http://localhost:8080` (login: `airflow` / `airflow`).
2.  **Create the Connection:**
    * Go to **Admin** > **Connections** > **Add a new record**.
    * **Conn Id:** `github_raw_http`
    * **Conn Type:** `HTTP`
    * **Host:** `https://raw.githubusercontent.com`
    * Save the connection.
3.  **Run the DAG:**
    * Go to the "DAGs" homepage.
    * Find `pipeline_actualizacion_cv_github` and toggle it **On**.
    * Click the "Play" button (Trigger DAG) on the right to start a manual run.
4.  **Force the Run:**
    * Click the DAG name to go to the "Grid" view.
    * The first task (`sensor_cv_es_github`) will be running (light green). Click on it.
    * In the panel that opens, click the square icon **"Mark as Success"**.
    * This will skip the sensor and force the rest of the pipeline to run, which will download the CV and populate the vector database.

### 6. Test the Chatbot!
Now that the database (`./dags/cv_vector_db`) is populated:

1.  Go to the API documentation at `http://localhost:8000/docs`.
2.  Open the `/ask` endpoint, click "Try it out".
3.  Ask a question:
    ```json
    {
      "texto": "¿Qué experiencia tienes con Python?"
    }
    ```
4.  You will get a full answer generated from your CV!
