# 📚 BOOKai: The Hybrid Recommendation Engine

### Real-time, Topic-Aware Book Recommendations with LLM-Powered Explanations

This project demonstrates a production-ready **hybrid architecture** that uses two different data models and multiple AI APIs to deliver the best possible book recommendations, optimizing for both speed and relevance simultaneously.

## ✨ Core Features

| Feature | Description | Technical Implementation |
| :--- | :--- | :--- |
| **Hybrid Routing** | Intelligently routes queries to the fastest, most reliable engine (Matrix for exact matches, RAG for topic searches). | Python router logic manages model selection based on `book_titles_set` lookup. |
| **Explainable Results** | Recommendations include a **"Chance of Liking" score** (derived from vector similarity) and an **AI-generated explanation** for the similarity [cite: user]. | `Recommendation` object includes `score: float`. Groq model generates `explanation`. |
| **Agent Verbose Mode** | The UI features a collapsible panel to show the Agent's thought process and tool calls, providing full transparency [cite: user]. | FastAPI streams newline-delimited JSON (NDJSON) events to the React UI. |
| **Tiered Model Strategy** | Optimizes for speed and cost by using the fastest available LLM for each specific task (Groq for formatting, Gemini for web search). | Dedicated API clients for **Groq**, **Gemini**, and **Hugging Face** backup. |

## 📐 System Architecture

The project's core strength is its logic flow. The application never assumes a single model is the best answer.

!

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend/API** | **Python 3.11+**, **FastAPI** | High-performance, asynchronous web server. |
| **AI Orchestration** | **LangChain** (Agents, Tooling) | Handles reasoning, web search (`search_tool`), and model routing. |
| **Data Storage** | **ChromaDB** | Persistent Vector Store for semantic search. |
| **AI Providers** | **Google AI (Gemini)**, **Groq**, **Hugging Face (Mixtral/GPT-OSS)** | Multi-tier model access and failover [cite: user]. |
| **Frontend/UI** | **React/Next.js** | Interactive, dynamic chat interface. |

## 🚀 Setup & Installation

### 1. Backend Setup

1.  **Clone the Repository (Monorepo Structure):**
    ```bash
    git clone [your-repo-link] BOOKai
    cd BOOKai
    ```

2.  **Create and Activate Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # .\venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    Create a file named `.env` in the root directory and add the following keys:
    ```dotenv
    GOOGLE_API_KEY="your_gemini_key"
    GROQ_API_KEY="your_groq_key"
    SERPER_API_KEY="your_serper_key"
    HF_TOKEN="your_huggingface_token"
    ```

5.  **Build the Vector Database:**
    Run your ingestion script once to populate the Chroma database.
    ```bash
    python -m scripts.run_ingestion
    ```

6.  **Run the Backend Server:**
    ```bash
    uvicorn main:app --reload
    ```
    The API will be live at `http://127.0.0.1:8000`.

### 2. Frontend Setup

1.  **Go to the UI directory:**
    ```bash
    cd bookai-ui
    ```

2.  **Install Node Dependencies:**
    ```bash
    npm install
    ```

3.  **Start the Frontend:**
    ```bash
    npm run dev
    ```
    The UI will open in your browser at `http://localhost:3000`.

---

## 💻 API Reference

The FastAPI application provides two main endpoints for your recommendation logic.

### 1. `POST /recommend`

The primary hybrid endpoint for requesting book ideas.

| Body Field | Type | Description |
| :--- | :--- | :--- |
| `query` | `string` | The user's natural language request (e.g., "I liked Germs" or "Recommend a book about AI") [cite: user]. |

| Response Field | Type | Description |
| :--- | :--- | :--- |
| `response` | `string` | The final, conversational message from the AI. |
| `recommendations` | `array` | List of books found by the hybrid router [cite: user]. |
| `recommendations[*].score` | `float` | The calculated vector similarity score (Chance of Liking) [cite: user]. |

### 2. `POST /summary`

Initiates the web search and summarization agent for any given book.

| Body Field | Type | Description |
| :--- | :--- | :--- |
| `book_title` | `string` | The exact title to summarize [cite: user]. |

| Response Field | Type | Description |
| :--- | :--- | :--- |
| `summary` | `string` | The single-paragraph, AI-generated summary of the book. |

***
*Project built by Avnish Mishra in 2025.*
 
