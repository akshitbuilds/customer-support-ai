# TechMart Customer Support AI

A multi-agent AI customer support assistant for a fictional electronics retailer (TechMart Electronics), built using Retrieval-Augmented Generation (RAG) and LLM-based intent routing.

## Overview

Unlike a single-purpose chatbot, this system routes each customer query to one or more specialized AI agents — Billing, Technical Support, Product, Complaint, or FAQ — based on LLM-driven intent detection. Each agent retrieves relevant context from a company knowledge base (via FAISS vector search) before generating a response with Groq's Llama 3.3 70B model.

## Features

- **LLM-based intent routing** — detects which department(s) a query belongs to, and can route to multiple agents simultaneously (e.g., a query mixing billing and technical issues gets responses from both, combined into one answer)
- **RAG-powered responses** — all agents (except Complaint) retrieve relevant chunks from company PDFs before answering, grounding responses in actual policy documents rather than the LLM's general knowledge
- **Persistent conversation memory** — every message is saved to SQLite, keyed by session ID, so conversations survive backend restarts
- **Live chat interface** — Streamlit frontend showing which agent(s) handled each response, with color-coded badges
- **Graceful error handling** — backend disconnection, timeouts, and malformed input are all handled without crashing

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | Python, FastAPI, Uvicorn |
| LLM | Groq (Llama 3.3 70B Versatile) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Database | SQLite |
| Frontend | Streamlit |
| PDF Processing | pypdf |

## Project Structure

```
customer_support_ai/
├── backend/
│   ├── agents/
│   │   ├── router.py          # LLM-based intent detection
│   │   └── agents.py          # 5 specialized agents (billing, technical, product, complaint, faq)
│   ├── rag/
│   │   ├── rag_pipeline.py    # PDF loading, chunking, embedding, FAISS retrieval
│   │   └── faiss_index.pkl    # Pre-built vector store
│   ├── database.py            # SQLite conversation memory
│   └── main.py                # FastAPI app, /chat, /history, /sessions, /health endpoints
├── frontend/
│   └── app.py                 # Streamlit chat interface
├── knowledge_base/             # Company policy PDFs (FAQ, refund, shipping, warranty, etc.)
├── .env                        # GROQ_API_KEY (not committed — see setup below)
└── requirements.txt
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/akshitbuilds/customer-support-ai.git
cd customer-support-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at [console.groq.com](https://console.groq.com).

### 4. Build the knowledge base (first-time setup only)
The FAISS index is already built and included (`backend/rag/faiss_index.pkl`). To rebuild it from the PDFs in `knowledge_base/`:
```bash
python backend/rag/rag_pipeline.py
```

### 5. Start the backend
```bash
uvicorn backend.main:app --reload --port 8000
```
Confirm it's running at `http://127.0.0.1:8000/docs`.

### 6. Start the frontend (in a second terminal)
```bash
streamlit run frontend/app.py
```
This opens the chat interface at `http://localhost:8501`.

## Sample Interactions

**Single-agent query:**
> "What is the price of the iPhone 15?" → routed to **Product** agent

**Multi-agent query:**
> "What is refund policy?" → routed to **Billing** and **FAQ** agents, responses combined

**Complaint/escalation:**
> "I am extremely disappointed with your service" → routed to **Complaint** agent, offers escalation contact

## Known Limitations

- Authentication is a lightweight session-name gate rather than full registration/login (documented as a scoping decision — see report)
- Rapid sequential message submission can occasionally cause a display race condition in the frontend (backend always processes correctly; this is a Streamlit session-state timing issue, not a backend/routing bug)
- Deployment to Render's free tier was attempted but blocked by the platform's 512MB memory limit, due to the combined footprint of the embedding model and FAISS index — see report for details

## Author

Akshit — 2nd year CS (AI & Data Science), M S Ramaiah Institute of Technology, Bengaluru
