# ---------- SAFE PATH SETUP ----------
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------- IMPORTS ----------
import streamlit as st
from src.rag_core import RAGCore
from src.memory import ConversationMemory

import base64

import os





    



# ---------- CONFIG ----------
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")

# ---------- SESSION INIT ----------
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "show_meta" not in st.session_state:
    st.session_state.show_meta = True


# ---------- AUTO CREATE FIRST CHAT ----------
if not st.session_state.chats:
    chat_id = "Chat 1"

    st.session_state.chats[chat_id] = {
        "messages": [],
        "memory": ConversationMemory(),
        "current_style": {
            "tone": ["professional"],
            "depth": 3,
            "language": "English",
            "bullet_summary": False,
            "format": "step"
        }
    }

    st.session_state.current_chat = chat_id


# ---------- HELPERS ----------
def generate_chat_title(text, max_len=40):
    text = text.strip().replace("\n", " ")
    return text[:max_len] + ("..." if len(text) > max_len else "")


def make_unique_title(base_title):
    existing = st.session_state.chats.keys()

    if base_title not in existing:
        return base_title

    i = 2
    while f"{base_title} ({i})" in existing:
        i += 1

    return f"{base_title} ({i})"


def create_new_chat():
    chat_id = f"Chat {len(st.session_state.chats) + 1}"

    st.session_state.chats[chat_id] = {
        "messages": [],
        "memory": ConversationMemory(),
        "current_style": {
            "tone": ["professional"],
            "depth": 3,
            "language": "English",
            "bullet_summary": False,
            "format": "step"
        }
    }

    st.session_state.current_chat = chat_id


# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("💬 Chats")

    if st.button("+ New Chat"):
        create_new_chat()

    st.markdown("### Select Chat")

    for chat_id in list(st.session_state.chats.keys()):
        if st.button(chat_id):
            st.session_state.current_chat = chat_id

    st.markdown("---")

    # ---------- STYLE CONTROLS ----------
    if st.session_state.current_chat:
        chat = st.session_state.chats[st.session_state.current_chat]

        st.markdown("### ⚙️ Response Style")

        tone = st.selectbox(
            "Tone",
            ["professional", "casual", "eli5", "analogy", "interview"],
            index=["professional", "casual", "eli5", "analogy", "interview"].index(
                chat["current_style"]["tone"][0]
            )
        )

        depth = st.slider("Depth Level", 0, 5, chat["current_style"]["depth"])

        language = st.selectbox(
            "Language",
            ["English", "Hinglish"],
            index=["English", "Hinglish"].index(chat["current_style"]["language"])
        )

        format_type = st.selectbox(
            "Response Format",
            ["normal", "bullet", "step", "structured"],
            index=["normal", "bullet", "step", "structured"].index(
                chat["current_style"]["format"]
            )
        )

        bullet_summary = st.checkbox(
            "Add Bullet Summary",
            value=chat["current_style"]["bullet_summary"]
        )

        show_meta = st.checkbox(
            "Show Confidence & Sources",
            value=st.session_state.show_meta
        )

        # update style (applies to future messages)
        chat["current_style"] = {
            "tone": [tone],
            "depth": depth,
            "language": language,
            "bullet_summary": bullet_summary,
            "format": format_type
        }

        st.session_state.show_meta = show_meta




# ---------- MAIN ----------
st.title("💬SUNDAY Chats")

chat = st.session_state.chats[st.session_state.current_chat]


# ---------- CACHE RAG ----------
@st.cache_resource
def get_rag(_memory):
    return RAGCore(memory=_memory)


rag = get_rag(chat["memory"])


# ---------- DISPLAY CHAT ----------
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- USER INPUT ----------
user_input = st.chat_input("Ask something...")

if user_input:
    # display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # store user message
    chat["messages"].append({
        "role": "user",
        "content": user_input
    })

    # ---------- AUTO RENAME (FIRST MESSAGE ONLY) ----------
    if len(chat["messages"]) == 1:
        base_title = generate_chat_title(user_input)
        new_title = make_unique_title(base_title)

        st.session_state.chats[new_title] = st.session_state.chats.pop(
            st.session_state.current_chat
        )
        st.session_state.current_chat = new_title
        chat = st.session_state.chats[new_title]  # update reference

    # ---------- GENERATE RESPONSE ----------
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            result = rag.query(
                user_input,
                style_config=chat["current_style"]
            )

            styled_answer = result["answer"]
            raw_answer = result["raw_answer"]
            confidence = result["confidence"]
            verdict = result["verdict"]
            sources = result["sources"]

            # show styled answer
            st.markdown(styled_answer)

            # metadata
            if st.session_state.show_meta:
                st.markdown("---")
                st.markdown(f"**Confidence:** {confidence} ({verdict})")

                if sources:
                    st.markdown("**Sources:**")
                    for s in sources:
                        st.markdown(f"- {s}")

    # ---------- STORE CLEAN ANSWER ----------
    chat["messages"].append({
        "role": "assistant",
        "content": raw_answer
    })


