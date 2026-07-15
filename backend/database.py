# Conversation Memory using SQLite
# Stores every message exchanged between user and AI
# so the system can maintain context across a session.
#
# Why SQLite? It runs locally with zero setup.
# No MongoDB account, no cloud, no configuration.
# Perfect for project submission and demos.

import sqlite3
import os
from datetime import datetime

DB_PATH = "backend/conversations.db"

def init_db():
    """
    Create the conversations table if it doesn't exist.
    Run this once when the server starts.
    
    Table structure:
    - id: auto-incrementing primary key
    - session_id: groups messages from same conversation
    - role: 'user' or 'assistant'
    - message: actual text content
    - agent_used: which agent(s) handled this message
    - timestamp: when the message was sent
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            agent_used TEXT DEFAULT 'none',
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized successfully")

def save_message(session_id: str, role: str, message: str, agent_used: str = "none"):
    """Save a single message to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (session_id, role, message, agent_used, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, role, message, agent_used, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_conversation_history(session_id: str, limit: int = 10) -> list:
    """
    Retrieve last N messages for a session.
    Used to give the LLM context of previous messages.
    
    Why limit=10? Sending entire history uses too many tokens.
    Last 10 messages is enough context for most conversations.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, message, agent_used, timestamp
        FROM conversations
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (session_id, limit))
    rows = cursor.fetchall()
    conn.close()
    
    # Reverse so oldest message comes first
    rows.reverse()
    return [
        {
            "role": row[0],
            "message": row[1],
            "agent_used": row[2],
            "timestamp": row[3]
        }
        for row in rows
    ]

def get_all_sessions() -> list:
    """Get all unique session IDs — used for analytics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT session_id, COUNT(*) as message_count, MAX(timestamp) as last_active
        FROM conversations
        GROUP BY session_id
        ORDER BY last_active DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [{"session_id": r[0], "messages": r[1], "last_active": r[2]} for r in rows]

# Test the database
if __name__ == "__main__":
    init_db()
    
    # Test saving messages
    save_message("test_session_1", "user", "I want a refund", "billing")
    save_message("test_session_1", "assistant", "I can help with that refund", "billing")
    save_message("test_session_1", "user", "My order number is 12345", "billing")
    
    # Test retrieving history
    history = get_conversation_history("test_session_1")
    print(f"Retrieved {len(history)} messages:")
    for msg in history:
        print(f"  [{msg['role']}]: {msg['message']}")
    
    print("Database working correctly!")