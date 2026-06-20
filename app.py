"""
app.py
-------
AreebaBot - A RAG-based personal chatbot about Areeba Amar.

Run locally:
    streamlit run app.py

Deployed on Streamlit Community Cloud (see README.md for the link & steps).
"""

import streamlit as st
from rag_engine import build_or_load_vectorstore, get_conversational_chain
import chat_store

st.set_page_config(page_title="AreebaBot", page_icon="🧩", layout="centered")

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ab-navy: #11182E;
        --ab-navy-soft: #1B2545;
        --ab-amber: #E8A33D;
        --ab-amber-soft: #F7D9A8;
        --ab-paper: #FAF7F2;
        --ab-ink: #1B2545;
        --ab-muted: #6B7280;
    }

    .stApp {
        background: var(--ab-paper);
    }

    header[data-testid="stHeader"] { background: transparent; }

    .ab-hero {
        background: linear-gradient(135deg, var(--ab-navy) 0%, var(--ab-navy-soft) 100%);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 28px;
        color: #F4F1EA;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px -10px rgba(17, 24, 46, 0.45);
    }
    .ab-hero::after {
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(232, 163, 61, 0.35), transparent 70%);
    }
    .ab-hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ab-amber-soft);
        background: rgba(232, 163, 61, 0.12);
        border: 1px solid rgba(232, 163, 61, 0.3);
        padding: 4px 10px;
        border-radius: 999px;
        margin-bottom: 14px;
        font-weight: 600;
    }
    .ab-hero h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
        color: #F4F1EA;
    }
    .ab-hero p {
        margin: 0;
        color: rgba(244, 241, 234, 0.78);
        font-size: 0.95rem;
        max-width: 480px;
        line-height: 1.5;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 4px 6px;
    }
    div[data-testid="stChatMessageAvatarUser"] {
        background-color: var(--ab-navy) !important;
    }
    div[data-testid="stChatMessageAvatarAssistant"] {
        background-color: var(--ab-amber) !important;
    }

    section[data-testid="stSidebar"] {
        background: var(--ab-navy);
    }
    section[data-testid="stSidebar"] * {
        color: #F4F1EA !important;
    }
    section[data-testid="stSidebar"] input {
        color: var(--ab-ink) !important;
        background-color: #F4F1EA !important;
    }
   /* New chat button: amber primary CTA */
    section[data-testid="stSidebar"] button[kind="primary"] {
        background: var(--ab-amber) !important;
        color: var(--ab-navy) !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background: var(--ab-amber-soft) !important;
    }

    /* Recents list: subtle dark secondary buttons */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: rgba(244, 241, 234, 0.06) !important;
        color: #F4F1EA !important;
        font-weight: 400 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        border: 1px solid rgba(244, 241, 234, 0.12) !important;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: rgba(244, 241, 234, 0.12) !important;
        border-color: rgba(244, 241, 234, 0.25) !important;
    }
    section[data-testid="stSidebar"] a {
        color: var(--ab-amber-soft) !important;
        text-decoration: underline;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(244, 241, 234, 0.15);
    }

    div[data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
    }

    /* Saved chat list buttons (secondary style - not the amber CTA) */
    .ab-history-btn button {
        background: transparent !important;
        color: #F4F1EA !important;
        font-weight: 400 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        border: 1px solid rgba(244, 241, 234, 0.12) !important;
    }
    .ab-history-btn button:hover {
        background: rgba(244, 241, 234, 0.08) !important;
    }
    .ab-history-btn-active button {
        background: rgba(232, 163, 61, 0.18) !important;
        border: 1px solid rgba(232, 163, 61, 0.4) !important;
        color: var(--ab-amber-soft) !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="ab-hero">
        <div class="ab-hero-tag">● Online · RAG-powered</div>
        <h1> 💻AreebaBot</h1>
        <p>I know everything on Areeba Amar's CV — her skills, education, projects,
        and certifications. Ask me anything about her background.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "current_session_id" not in st.session_state:
    existing = chat_store.list_sessions()
    if existing:
        st.session_state.current_session_id = existing[0][0]
    else:
        st.session_state.current_session_id = chat_store.create_session()

if "messages" not in st.session_state:
    st.session_state.messages = chat_store.get_messages(st.session_state.current_session_id)

if "vectorstore" not in st.session_state:
    with st.spinner("📚 Loading knowledge base (one-time setup)..."):
        st.session_state.vectorstore = build_or_load_vectorstore()

if "chain" not in st.session_state:
    st.session_state.chain = None

# ---------------------------------------------------------------------------
# Sidebar - New chat, saved chat history, settings
# ---------------------------------------------------------------------------
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True, key="new_chat_btn", type="primary"):
        st.session_state.current_session_id = chat_store.create_session()
        st.session_state.messages = []
        st.session_state.chain = None
        st.rerun()

    st.markdown("#### Recents")
    sessions = chat_store.list_sessions()
    if not sessions:
        st.caption("No saved chats yet.")
    for session_id, title, _ in sessions:
        is_active = session_id == st.session_state.current_session_id
        label = f"💬 {title}" if is_active else title
        open_key = f"open_{session_id}"
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=open_key, use_container_width=True, type="secondary"):
                st.session_state.current_session_id = session_id
                st.session_state.messages = chat_store.get_messages(session_id)
                st.session_state.chain = None
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{session_id}", type="secondary"):
                chat_store.delete_session(session_id)
                if session_id == st.session_state.current_session_id:
                    remaining = chat_store.list_sessions()
                    if remaining:
                        st.session_state.current_session_id = remaining[0][0]
                        st.session_state.messages = chat_store.get_messages(remaining[0][0])
                    else:
                        st.session_state.current_session_id = chat_store.create_session()
                        st.session_state.messages = []
                    st.session_state.chain = None
                st.rerun()

    st.divider()
    st.markdown("### ⚙️ Settings")
    st.markdown(
        "AreebaBot uses **Groq** (free LLM API) to generate answers.\n\n"
        "1. Get a free key at [console.groq.com](https://console.groq.com/keys)\n"
        "2. Paste it below 👇"
    )
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.divider()
    st.markdown(
        "**About AreebaBot**\n\n"
        "A RAG (Retrieval-Augmented Generation) chatbot built with "
        "LangChain + FAISS + Groq, using Areeba's CV as its knowledge base. "
        "Chats are saved locally so you can reopen them anytime."
    )

# ---------------------------------------------------------------------------
# Empty state - suggested questions (only show before the first message)
# ---------------------------------------------------------------------------
SUGGESTIONS = [
    "What programming languages does Areeba know?",
    "Tell me about her academic projects",
    "What certifications does she have?",
    "Where is she from?",
]

clicked_suggestion = None
if not st.session_state.messages:
    st.markdown("##### Try asking")
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                clicked_suggestion = suggestion

# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "💻" if msg["role"] == "user" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_question = st.chat_input("Type your question about Areeba...") or clicked_suggestion

if user_question:
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar first.")
        st.stop()

    # Build the conversational chain (only once per session / per key change)
    if st.session_state.chain is None:
        with st.spinner("🔧 Setting up AreebaBot..."):
            st.session_state.chain = get_conversational_chain(
                st.session_state.vectorstore, groq_api_key
            )

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_question)

    # Generate and show bot response
    with st.chat_message("assistant", avatar="🧩"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.chain.invoke({"question": user_question})
                answer = result["answer"]
            except Exception as e:
                answer = f"⚠️ Something went wrong: {e}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Persist this session to disk so it shows up in "Recents" after reload
    chat_store.save_messages(st.session_state.current_session_id, st.session_state.messages)

    st.rerun()