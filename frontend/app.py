"""
TechMart Customer Support AI - Streamlit Frontend
Day 5: Connects to FastAPI backend (/chat endpoint) to provide
a live chat interface with multi-agent routing visibility.
"""

import streamlit as st
import requests

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000/chat"
REQUEST_TIMEOUT = 20  # seconds - generous because multi-agent calls can chain multiple LLM requests

# --- Page setup ---
st.set_page_config(
    page_title="TechMart Customer Support",
    page_icon="🛠️",
    layout="centered"
)

# --- Custom styling ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 4px 8px;
        margin-bottom: 8px;
    }
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛠️ TechMart Customer Support")
st.caption("Ask about billing, technical issues, products, complaints, or general FAQs.")

# --- Agent badge helper ---
# Color-codes each agent so routing is visually distinct at a glance
AGENT_COLORS = {
    "billing": "#F59E0B",     # amber
    "technical": "#3B82F6",   # blue
    "product": "#8B5CF6",     # purple
    "complaint": "#EF4444",   # red
    "faq": "#10B981",         # green
}
def render_agent_badges(agents):
    badges_html = ""
    for agent in agents:
        color = AGENT_COLORS.get(agent.lower(), "#6B7280")  # gray fallback for unknown agents
        badges_html += f'<span class="agent-badge" style="background-color:{color}">{agent.upper()}</span>'
    st.markdown(badges_html, unsafe_allow_html=True)
# --- Minimal login gate ---
# Full auth (registration, password hashing, JWT) is out of scope for this
# project's timeline. This satisfies session identification per Module 1
# while keeping the actual security work honestly scoped in the report.
# --- Login / Register gate ---
if "user_name" not in st.session_state:
    st.session_state.user_name = None

if not st.session_state.user_name:
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            try:
                r = requests.post(
                    "http://127.0.0.1:8000/login",  # update this to your deployed backend URL later
                    json={"username": login_user, "password": login_pass},
                    timeout=10
                )
                if r.status_code == 200:
                    st.session_state.user_name = login_user
                    st.rerun()
                else:
                    st.error(r.json().get("detail", "Login failed"))
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Could not connect to the backend.")

    with tab_register:
        reg_user = st.text_input("Choose a username", key="reg_user")
        reg_pass = st.text_input("Choose a password", type="password", key="reg_pass")
        if st.button("Register"):
            try:
                r = requests.post(
                    "http://127.0.0.1:8000/register",
                    json={"username": reg_user, "password": reg_pass},
                    timeout=10
                )
                if r.status_code == 200:
                    st.success("Registered! Please log in above.")
                else:
                    st.error(r.json().get("detail", "Registration failed"))
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Could not connect to the backend.")

    st.stop()   

# --- Session state initialization ---
# Streamlit reruns the entire script top-to-bottom on every interaction,
# so anything that needs to persist across reruns (chat history, session_id)
# must live in st.session_state, not as a normal local variable.
if "session_id" not in st.session_state:
    st.session_state.session_id = ""  # empty string tells backend to start a new session

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {"role", "content", "agents"}

# --- Sidebar: session info + reset ---
with st.sidebar:
    st.subheader("Session")
    if st.session_state.session_id:
        st.text(f"ID: {st.session_state.session_id[:8]}...")
    else:
        st.text("No active session yet")

    if st.button("Start New Conversation"):
        st.session_state.session_id = ""
        st.session_state.messages = []
        st.rerun()

# --- Display existing chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("agents"):
            render_agent_badges(msg["agents"])

# --- Handle new user input ---
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={
                        "message": user_input,
                        "session_id": st.session_state.session_id
                    },
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()

                ai_reply = data.get("response", "Sorry, I didn't get a response.")
                agents_used = data.get("agents_used", [])

                st.session_state.session_id = data.get("session_id", st.session_state.session_id)

                st.markdown(ai_reply)
                if agents_used:
                    render_agent_badges(agents_used)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_reply,
                    "agents": agents_used
                })
                st.rerun()

            except requests.exceptions.ConnectionError:
                error_msg = "⚠️ Could not connect to the backend. Is uvicorn running on port 8000?"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "agents": []})

            except requests.exceptions.Timeout:
                error_msg = "⚠️ The backend took too long to respond. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "agents": []})

            except requests.exceptions.HTTPError as e:
                error_msg = f"⚠️ Backend returned an error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "agents": []})