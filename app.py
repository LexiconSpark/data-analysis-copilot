import os
import uuid
from datetime import datetime, timezone

import openai
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Constants — edit these to change model behaviour.
# ---------------------------------------------------------------------------
MODEL = "gpt-4o"
TEMPERATURE = 0.7
SYSTEM_PROMPT = "You are a helpful assistant."

# ---------------------------------------------------------------------------
# API key (T005)
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="GPT-4o Chatbot", layout="centered")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found — add it to your .env file.")
    st.stop()


# ---------------------------------------------------------------------------
# LangChain chain (T006)
# Cached once per app process. The history getter always returns the live
# session-state object so memory accumulates across Streamlit reruns.
# ---------------------------------------------------------------------------
@st.cache_resource
def build_chain() -> RunnableWithMessageHistory:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
    )
    chain = prompt | llm | StrOutputParser()

    def get_session_history(session_id: str) -> ChatMessageHistory:  # noqa: ARG001
        return st.session_state.history

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


# ---------------------------------------------------------------------------
# Session state initialisation (T007)
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = ChatMessageHistory()
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False


# ---------------------------------------------------------------------------
# Session reset — New Chat (T016)
# ---------------------------------------------------------------------------
def reset_session() -> None:
    st.session_state.messages = []
    st.session_state.history = ChatMessageHistory()
    st.session_state.is_loading = False


# ---------------------------------------------------------------------------
# Conversation rendering (T008 + T009)
# ---------------------------------------------------------------------------
def render_conversation() -> None:
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("Hello! How can I help you today?")
        return
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# LLM invocation (T011) — T014/T015: history getter returns live session
# object; session_id config ensures RunnableWithMessageHistory accumulates
# history across multiple turns.
# ---------------------------------------------------------------------------
def get_ai_response(prompt: str) -> str:
    return build_chain().invoke(
        {"input": prompt},
        config={"configurable": {"session_id": st.session_state.session_id}},
    )


# ---------------------------------------------------------------------------
# Error formatting (T012)
# ---------------------------------------------------------------------------
def format_error(exc: Exception) -> str:
    if isinstance(exc, openai.AuthenticationError):
        return "API key is invalid or missing. Check your .env file."
    if isinstance(exc, openai.RateLimitError):
        return "Rate limit reached. Please wait a moment and try again."
    if isinstance(exc, openai.APITimeoutError):
        return "Request timed out. Please try again."
    return "Something went wrong. Please try again."


# ---------------------------------------------------------------------------
# Main app body
# ---------------------------------------------------------------------------
init_session_state()

# Header with New Chat button (T017)
col1, col2 = st.columns([4, 1])
col1.title("GPT-4o Chatbot")
col2.button("New Chat", on_click=reset_session)

# ---------------------------------------------------------------------------
# Loading state (T010 + T013): render conversation so far (including the
# newly appended user message), make the API call with spinner, then rerun
# to show the completed exchange with input re-enabled.
# ---------------------------------------------------------------------------
if st.session_state.is_loading:
    last_user_msg = next(
        m["content"]
        for m in reversed(st.session_state.messages)
        if m["role"] == "user"
    )

    render_conversation()

    response = None
    error_msg = None
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = get_ai_response(last_user_msg)
            except Exception as exc:
                error_msg = format_error(exc)

    if error_msg:
        st.error(error_msg)
    else:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    st.session_state.is_loading = False
    st.rerun()

# ---------------------------------------------------------------------------
# Normal state (T008 + T009 + T013): render full conversation and show
# the input. Input is disabled while is_loading is True, but in loading
# state we never reach this code (st.rerun() above cuts execution).
# ---------------------------------------------------------------------------
render_conversation()

if prompt := st.chat_input("Type a message…"):
    if prompt.strip():
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt.strip(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        st.session_state.is_loading = True
        st.rerun()
