# Agent Router for TechMart Customer Support
# This module detects customer intent using an LLM
# and routes the query to the correct specialized agent(s).
# A single query can be routed to MULTIPLE agents if needed.

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Intent categories and which agent handles them
INTENT_MAP = {
    "billing": "billing",
    "refund": "billing",
    "payment": "billing",
    "subscription": "billing",
    "invoice": "billing",
    "technical": "technical",
    "login": "technical",
    "password": "technical",
    "error": "technical",
    "installation": "technical",
    "not working": "technical",
    "product": "product",
    "price": "product",
    "features": "product",
    "availability": "product",
    "complaint": "complaint",
    "angry": "complaint",
    "dissatisfied": "complaint",
    "escalate": "complaint",
    "faq": "faq",
    "policy": "faq",
    "contact": "faq",
    "hours": "faq"
}

def detect_intent(user_query: str) -> list:
    """
    Use LLM to detect which agents should handle this query.
    Returns a list of agent names — can be multiple.
    
    Why LLM for intent detection instead of keywords?
    Keywords miss context. "I can't access my paid features" 
    contains no billing keywords but is clearly a billing issue.
    LLM understands meaning, not just words.
    """
    prompt = f"""You are an intent classifier for TechMart Electronics customer support.

Given a customer query, identify which departments should handle it.
Choose from: billing, technical, product, complaint, faq

Rules:
- billing: payment issues, refunds, subscriptions, invoices
- technical: login problems, errors, installation, app not working
- product: product features, pricing, availability, comparisons  
- complaint: customer is angry, wants to escalate, dissatisfied
- faq: general questions, policies, contact info, shipping

Return ONLY a comma-separated list of relevant departments.
Example: "billing,technical" or "faq" or "complaint,billing"

Customer query: {user_query}

Departments:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0
    )

    result = response.choices[0].message.content.strip().lower()
    
    # Parse the comma-separated response into a list
    agents = [a.strip() for a in result.split(",") if a.strip() in 
              ["billing", "technical", "product", "complaint", "faq"]]
    
    # Fallback to faq if nothing detected
    if not agents:
        agents = ["faq"]
    
    print(f"Query: {user_query}")
    print(f"Detected agents: {agents}")
    return agents


# Test the router
if __name__ == "__main__":
    test_queries = [
        "I paid for Premium but can't access it",
        "How do I reset my password?",
        "What is the price of the iPhone 15?",
        "I am very angry with your service",
        "What are your support hours?"
    ]
    
    for query in test_queries:
        agents = detect_intent(query)
        print(f"→ Routed to: {agents}\n")