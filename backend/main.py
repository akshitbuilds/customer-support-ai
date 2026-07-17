# TechMart Customer Support AI - Main FastAPI Server
# This is the central API that connects everything:
# Frontend → main.py → Router → Agents → RAG → LLM → Response
# Also saves every conversation to SQLite for memory.

import os
import sys
import uuid
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from agents.router import detect_intent
from agents.agents import run_agents, combine_responses
from database import init_db, save_message, get_conversation_history, get_all_sessions, create_user, verify_user

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="TechMart Customer Support AI",
    description="Multi-Agent AI Customer Support using RAG and LLMs",
    version="1.0.0"
)

# CORS middleware — allows frontend to talk to backend
# Without this, browser blocks requests from different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
init_db()

# ── Request/Response Models ────────────────────────────────────
# Pydantic models define the structure of API request and response
# FastAPI automatically validates incoming data against these models

class ChatRequest(BaseModel):
    message: str          # Customer's message
    session_id: str = ""  # Empty = new session, filled = existing

class ChatResponse(BaseModel):
    response: str         # AI's response
    session_id: str       # Session ID for continuing conversation
    agents_used: list     # Which agents handled this query
    message_count: int    # Total messages in this session
class AuthRequest(BaseModel):
    username: str
    password: str

class HistoryResponse(BaseModel):
    session_id: str
    history: list

# ── API Endpoints ──────────────────────────────────────────────

@app.get("/")
def root():
    """Health check endpoint — confirms server is running."""
    return {
        "status": "running",
        "message": "TechMart Customer Support AI is active",
        "version": "1.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint — processes customer messages.
    
    Flow:
    1. Generate session_id if new conversation
    2. Save user message to database
    3. Detect intent using router
    4. Run appropriate agent(s)
    5. Combine responses if multiple agents
    6. Save AI response to database
    7. Return response to frontend
    """
    try:
        # Step 1: Handle session
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())  # Generate unique session ID

        # Step 2: Save user message
        save_message(session_id, "user", request.message)

        # Step 3: Detect intent
        agents = detect_intent(request.message)

        # Step 4: Run agents
        responses = run_agents(request.message, agents)

        # Step 5: Combine responses
        final_response = combine_responses(responses)

        # Step 6: Save AI response
        save_message(
            session_id,
            "assistant",
            final_response,
            ",".join(agents)
        )

        # Step 7: Get message count for this session
        history = get_conversation_history(session_id)

        return ChatResponse(
            response=final_response,
            session_id=session_id,
            agents_used=agents,
            message_count=len(history)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    """
    Retrieve conversation history for a session.
    Used by frontend to display previous messages on reload.
    """
    history = get_conversation_history(session_id)
    return HistoryResponse(session_id=session_id, history=history)


@app.get("/sessions")
def get_sessions():
    """
    Get all conversation sessions.
    Useful for analytics — how many users, how active.
    """
    sessions = get_all_sessions()
    return {"total_sessions": len(sessions), "sessions": sessions}


@app.get("/health")
def health_check():
    """Detailed health check — confirms all components are loaded."""
    return {
        "status": "healthy",
        "components": {
            "database": "connected",
            "rag_pipeline": "loaded",
            "agents": "ready",
            "llm": "groq/llama-3.3-70b-versatile"
        }
    }
@app.post("/register")
def register(request: AuthRequest):
    """Register a new user. Fails if username already taken."""
    if len(request.username.strip()) < 3 or len(request.password) < 4:
        raise HTTPException(status_code=400, detail="Username must be 3+ chars, password 4+ chars")
    success = create_user(request.username.strip(), request.password)
    if not success:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"status": "success", "message": "User registered"}

@app.post("/login")
def login(request: AuthRequest):
    """Verify login credentials."""
    if verify_user(request.username.strip(), request.password):
        return {"status": "success", "username": request.username.strip()}
    raise HTTPException(status_code=401, detail="Invalid username or password")