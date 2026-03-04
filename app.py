import os
import uuid
from datetime import datetime, timezone

import openai
import pandas as pd
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
LANGSMITH_PROJECT = "data_analysis_copilot"

MAX_CSV_ROWS = 10_000

DEFAULT_CSV_DATA = {
    "A": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "B": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
    "C": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
}

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
            ("system", "{system_context}"),
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
# CSV context builder
# ---------------------------------------------------------------------------
def build_system_context() -> str:
    csv_text = st.session_state.df.to_csv(index=False)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "The user has provided the following CSV data. "
        "Use it when answering questions:\n\n"
        f"```csv\n{csv_text}\n```"
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
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(DEFAULT_CSV_DATA)
    if "csv_truncated" not in st.session_state:
        st.session_state.csv_truncated = False


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
        {"input": prompt, "system_context": build_system_context()},
        config={
            "configurable": {"session_id": st.session_state.session_id},
            "run_name": "csv-chat",
            "tags": ["streamlit", "csv-context"],
            "metadata": {
                "session_id": st.session_state.session_id,
                "csv_rows": len(st.session_state.df),
                "csv_cols": len(st.session_state.df.columns),
            },
        },
    )


# ---------------------------------------------------------------------------
# Sidebar: CSV data display and upload
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.header("CSV Data")
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                if len(df) > MAX_CSV_ROWS:
                    df = df.iloc[:MAX_CSV_ROWS]
                    st.session_state.csv_truncated = True
                else:
                    st.session_state.csv_truncated = False
                st.session_state.df = df
            except Exception:
                st.error("Could not parse the uploaded file. Please upload a valid CSV.")

        if st.button("Reset to default data"):
            st.session_state.df = pd.DataFrame(DEFAULT_CSV_DATA)
            st.session_state.csv_truncated = False

        if st.session_state.csv_truncated:
            st.warning(f"CSV truncated to {MAX_CSV_ROWS:,} rows.")

        st.dataframe(st.session_state.df, use_container_width=True)
        rows, cols = st.session_state.df.shape
        st.caption(f"{rows:,} rows × {cols} columns")

        tracing_on = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        st.caption(f"Tracing: {'✓ LangSmith' if tracing_on else 'off'}")


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
render_sidebar()

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
