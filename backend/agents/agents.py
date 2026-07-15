# Specialized AI Agents for TechMart Customer Support
# Each agent is an expert in one domain.
# They use RAG to retrieve relevant company documents
# before generating a response with the LLM.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from dotenv import load_dotenv
from rag.rag_pipeline import load_vector_store, retrieve

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load vector store once at startup — not on every query
# This saves time because loading FAISS index is expensive
vector_store = load_vector_store()

def generate_response(system_prompt: str, user_query: str, context: str) -> str:
    """
    Core LLM call used by all agents.
    Combines system prompt + retrieved context + user query.
    
    Why separate system_prompt per agent?
    Each agent has different expertise and tone.
    Billing agent is formal. Technical agent is step-by-step.
    This is how we specialize behavior without training separate models.
    """
    full_prompt = f"""You are a TechMart Electronics customer support agent.

{system_prompt}

Use the following retrieved information to answer the customer:
{context}

If the information is not in the context, say you will escalate to a human agent.
Always be polite, professional, and specific.
Keep your answer under 150 words.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": user_query}
        ],
        max_tokens=300,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def billing_agent(user_query: str) -> str:
    """
    Handles: payment issues, refunds, subscriptions, invoices.
    Retrieves from: refund_policy.pdf, pricing.pdf
    """
    # Retrieve relevant context from knowledge base
    results = retrieve(user_query, vector_store, top_k=3)
    context = "\n\n".join([r["chunk"] for r in results])

    system_prompt = """You are the Billing specialist at TechMart Electronics.
You handle payment issues, refund requests, subscription problems, and invoices.
Always provide specific refund timelines and process steps from the policy."""

    return generate_response(system_prompt, user_query, context)


def technical_agent(user_query: str) -> str:
    """
    Handles: login issues, errors, installation, app problems.
    Retrieves from: technical_support.pdf
    """
    results = retrieve(user_query, vector_store, top_k=3)
    context = "\n\n".join([r["chunk"] for r in results])

    system_prompt = """You are the Technical Support specialist at TechMart Electronics.
You help with login issues, password resets, app errors, and installation problems.
Always provide clear step-by-step instructions."""

    return generate_response(system_prompt, user_query, context)


def product_agent(user_query: str) -> str:
    """
    Handles: product features, pricing, availability, comparisons.
    Retrieves from: pricing.pdf, faq.pdf
    """
    results = retrieve(user_query, vector_store, top_k=3)
    context = "\n\n".join([r["chunk"] for r in results])

    system_prompt = """You are the Product specialist at TechMart Electronics.
You provide information about products, pricing, features, and availability.
Always mention specific prices and product names from the catalog."""

    return generate_response(system_prompt, user_query, context)


def complaint_agent(user_query: str) -> str:
    """
    Handles: complaints, angry customers, escalations.
    Does NOT use RAG — focuses on empathy and escalation.
    
    Why no RAG here? Complaint handling is about emotional intelligence,
    not document retrieval. The response should be empathetic first.
    """
    system_prompt = """You are the Customer Relations specialist at TechMart Electronics.
You handle complaints and dissatisfied customers with empathy and professionalism.
Always acknowledge their frustration, apologize sincerely, and offer a clear resolution path.
If needed, offer to escalate to a senior manager."""

    context = "Escalation contact: escalations@techmart.com | Priority support: 1-800-TECHMART"
    return generate_response(system_prompt, user_query, context)


def faq_agent(user_query: str) -> str:
    """
    Handles: general questions, policies, contact info, shipping.
    Retrieves from: faq.pdf, shipping_policy.pdf, warranty.pdf
    """
    results = retrieve(user_query, vector_store, top_k=3)
    context = "\n\n".join([r["chunk"] for r in results])

    system_prompt = """You are the General Support specialist at TechMart Electronics.
You answer general questions about policies, shipping, warranty, and company information.
Always provide accurate information from TechMart's official policies."""

    return generate_response(system_prompt, user_query, context)


# Agent dispatcher — maps agent name to function
AGENT_MAP = {
    "billing": billing_agent,
    "technical": technical_agent,
    "product": product_agent,
    "complaint": complaint_agent,
    "faq": faq_agent
}


def run_agents(user_query: str, agent_names: list) -> dict:
    """
    Run all detected agents and collect their responses.
    If multiple agents handle the query, responses are combined.
    
    Returns a dict with agent name and their response.
    """
    responses = {}
    for agent_name in agent_names:
        if agent_name in AGENT_MAP:
            print(f"Running {agent_name} agent...")
            responses[agent_name] = AGENT_MAP[agent_name](user_query)
    return responses


def combine_responses(responses: dict) -> str:
    """
    If multiple agents responded, combine into one final answer.
    Each section is clearly labeled by department.
    """
    if len(responses) == 1:
        return list(responses.values())[0]

    combined = ""
    for agent_name, response in responses.items():
        combined += f"**{agent_name.upper()} TEAM:**\n{response}\n\n"
    return combined.strip()


# Test all agents
if __name__ == "__main__":
    from router import detect_intent

    test_queries = [
        "I want a refund for my order",
        "My app keeps crashing on login",
        "What laptops do you have under 50000?",
        "I am extremely disappointed with your service",
        "What is your return policy?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Customer: {query}")
        agents = detect_intent(query)
        responses = run_agents(query, agents)
        final = combine_responses(responses)
        print(f"Response:\n{final}")