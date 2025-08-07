import streamlit as st
import os
import uuid
from dotenv import load_dotenv
import asyncio
import sys
import pickle
from rag import rag_pipeline

# --- Environment Setup ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GITHUB_TOKEN"] = os.getenv("GITHUB_TOKEN")

# --- Directory Setup ---
UPLOAD_DIR = "uploads"
SESSION_DIR = "chat_history"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)

# --- CSS for scrollable chat window ---
st.markdown("""
    <style>
        .chat-message {
            margin-bottom: 1.5rem;
        }
        .chat-role {
            font-weight: bold;
            margin-bottom: 0.3rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Cache the pipeline ---
@st.cache_resource(show_spinner="Setting up RAG pipeline...")
def setup_rag_pipeline(file_path):
    return rag_pipeline(file_path)  # returns (graph, memory)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_graph" not in st.session_state:
    st.session_state.rag_graph = None

if "memory" not in st.session_state:
    st.session_state.memory = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- Load Previous Chat Session ---
session_file = os.path.join(SESSION_DIR, f"{st.session_state.session_id}.pkl")
if os.path.exists(session_file):
    with open(session_file, "rb") as f:
        st.session_state.messages = pickle.load(f)

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("1. Upload your Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        temp_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.session_state.rag_graph, st.session_state.memory = setup_rag_pipeline(temp_path)
        st.success("Document processed! You can now ask questions.")
        os.remove(temp_path)

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I'm ready! Ask me anything about **{uploaded_file.name}**."
        })

# --- Chat UI ---
st.title("Chat with your PDF using LangGraph")

chat_box = st.container()
with chat_box:
    st.markdown('<div>', unsafe_allow_html=True)

    for message in st.session_state.messages:
        role_label = "🧑 You" if message["role"] == "user" else "🤖 Assistant"
        st.markdown(f'''
            <div class="chat-message">
                <div class="chat-role">{role_label}</div>
                <div>{message["content"]}</div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- Chat Input & RAG Interaction ---
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})


    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.rag_graph:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_graph.invoke(
                    {"messages": [{"type": "human", "content": prompt}]},
                    config={"configurable": {"thread_id": st.session_state.session_id}},
                )
                messages = result.get("messages", [])
                if messages and hasattr(messages[-1], "content"):
                    answer = messages[-1].content
                else:
                    answer = "Sorry, I couldn't find an answer."

                st.markdown(answer)

        # Store assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})


    else:
        st.warning("Please upload a document first.")

# --- Save Chat History (pkl + txt) ---
with open(session_file, "wb") as f:
    pickle.dump(st.session_state.messages, f)

text_file = os.path.join(SESSION_DIR, f"chat_history_{st.session_state.session_id}.txt")
with open(text_file, "w") as f:
    for message in st.session_state.messages:
        f.write(f"{message['role'].capitalize()}: {message['content']}\n\n")
