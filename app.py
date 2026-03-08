import os
import uuid
from datetime import datetime, timezone

import matplotlib
import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "gpt-4o"
TEMPERATURE = 0.7

# T005 — data analyst persona
SYSTEM_PROMPT = (
    "You are an expert data analyst assistant. "
    "When the user asks a data analysis question:\n"
    "1. Provide a clear chat_reply summarizing your approach and key findings.\n"
    "2. Provide plan_steps: a numbered list of concrete analysis steps.\n"
    "3. Provide code: complete, executable Python code using pandas and matplotlib "
    "that implements the plan. Store figures in a variable named `fig` and "
    "tabular results in a variable named `result`.\n\n"
    "If the question is purely conversational (not a data analysis request), "
    "return an empty list for plan_steps and an empty string for code."
)

LANGSMITH_PROJECT = "data_analysis_copilot"
MAX_CSV_ROWS = 10_000

DEFAULT_CSV_DATA = {
    "A": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "B": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
    "C": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
}


# ---------------------------------------------------------------------------
# T002 — Structured output model
# ---------------------------------------------------------------------------
class AnalysisResponse(BaseModel):
    chat_reply: str = Field(
        description="Natural language summary of the analysis or conversational answer"
    )
    plan_steps: list[str] = Field(
        default_factory=list,
        description="Ordered analysis steps; empty list for conversational queries",
    )
    code: str = Field(
        default="",
        description="Complete executable Python code; empty string for conversational queries",
    )


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# T004 / T026 — wide layout + updated title
st.set_page_config(page_title="Data Analyst Copilot", layout="wide")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found — add it to your .env file.")
    st.stop()


# ---------------------------------------------------------------------------
# T006 — LangChain chain with structured output
# Cached once per app process. History is managed manually so that
# with_structured_output() works cleanly without RunnableWithMessageHistory.
# ---------------------------------------------------------------------------
@st.cache_resource
def build_chain():
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
    return prompt | llm.with_structured_output(AnalysisResponse)


# ---------------------------------------------------------------------------
# CSV context builder
# ---------------------------------------------------------------------------
def build_system_context() -> str:
    csv_text = st.session_state.df.to_csv(index=False)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "The user's CSV data:\n\n"
        f"```csv\n{csv_text}\n```"
    )


# ---------------------------------------------------------------------------
# T003 — Session state initialisation
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
    # 002-analyst-copilot-ui additions
    if "plan_steps" not in st.session_state:
        st.session_state.plan_steps = []
    if "plan_code" not in st.session_state:
        st.session_state.plan_code = ""
    if "templates" not in st.session_state:
        st.session_state.templates = []
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None


# ---------------------------------------------------------------------------
# T010 — Session reset (New Chat)
# Preserves df and templates; clears conversation and analysis state.
# ---------------------------------------------------------------------------
def reset_session() -> None:
    st.session_state.messages = []
    st.session_state.history = ChatMessageHistory()
    st.session_state.is_loading = False
    st.session_state.plan_steps = []
    st.session_state.plan_code = ""
    st.session_state.analysis_result = None


# ---------------------------------------------------------------------------
# Error formatting
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
# T008 — LLM invocation with structured output
# Manually manages ChatMessageHistory so history accumulates correctly
# across Streamlit reruns without RunnableWithMessageHistory complications.
# ---------------------------------------------------------------------------
def get_ai_response(prompt: str) -> str:
    response: AnalysisResponse = build_chain().invoke(
        {
            "input": prompt,
            "system_context": build_system_context(),
            "history": st.session_state.history.messages,
        },
        config={
            "run_name": "csv-chat",
            "tags": ["streamlit", "csv-context"],
            "metadata": {
                "session_id": st.session_state.session_id,
                "csv_rows": len(st.session_state.df),
                "csv_cols": len(st.session_state.df.columns),
            },
        },
    )
    # Update history with the raw prompt and the chat-facing reply
    st.session_state.history.add_user_message(prompt)
    st.session_state.history.add_ai_message(response.chat_reply)
    # Propagate structured parts to session state
    st.session_state.plan_steps = response.plan_steps
    st.session_state.plan_code = response.code
    return response.chat_reply


# ---------------------------------------------------------------------------
# T022 — Code execution with isolated namespace
# ---------------------------------------------------------------------------
def execute_code(code: str, df: pd.DataFrame) -> dict:
    namespace: dict = {"df": df, "pd": pd, "plt": plt}
    try:
        exec(code, namespace)  # noqa: S102
        # Prefer explicitly assigned `fig`; fall back to any open figure
        fig = namespace.get("fig") or (
            plt.gcf() if plt.get_fignums() else None
        )
        if fig is not None:
            result = {"output_type": "figure", "content": fig, "error": None}
            plt.close("all")
            return result
        result_val = namespace.get("result")
        if isinstance(result_val, pd.DataFrame):
            return {"output_type": "dataframe", "content": result_val, "error": None}
        if result_val is not None:
            return {"output_type": "text", "content": str(result_val), "error": None}
        return {"output_type": "none", "content": None, "error": None}
    except Exception as exc:
        plt.close("all")
        return {"output_type": "none", "content": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# T007, T009, T010, T011 — Chat panel (top-left)
# ---------------------------------------------------------------------------
def render_chat_panel(col) -> None:
    with col:
        hdr_col, btn_col = st.columns([5, 1])
        hdr_col.subheader("Chat")
        btn_col.button("New Chat", on_click=reset_session)

        chat_container = st.container(height=400)
        with chat_container:
            if not st.session_state.messages:
                with st.chat_message("assistant"):
                    st.markdown("Hello! Ask me about your CSV data.")
            else:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            if st.session_state.is_loading:
                last_user_msg = next(
                    m["content"]
                    for m in reversed(st.session_state.messages)
                    if m["role"] == "user"
                )
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

        if prompt := st.chat_input("Ask about your data…"):
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


# ---------------------------------------------------------------------------
# T015–T021, T024 — Tabs panel (top-right): Template | Plan | Code
# ---------------------------------------------------------------------------
def render_tabs_panel(col) -> None:
    with col:
        st.subheader("Analysis")
        tab_template, tab_plan, tab_code = st.tabs(["Template", "Plan", "Code"])

        # T018, T019, T020 — Template tab
        with tab_template:
            if st.session_state.templates:
                for i, tmpl in enumerate(st.session_state.templates):
                    tc1, tc2 = st.columns([4, 1])
                    tc1.markdown(
                        f"**{tmpl['name']}** — {tmpl['created_at'][:10]}"
                    )
                    if tc2.button("Load", key=f"load_tmpl_{i}"):
                        st.session_state.plan_steps = tmpl["plan_steps"]
                        st.session_state.plan_code = tmpl["code"]
                        st.rerun()
            else:
                st.caption("No saved templates yet.")

            st.divider()
            tmpl_name = st.text_input(
                "Template name",
                key="tmpl_name_input",
                placeholder="e.g. Filter & Plot Temperature",
            )
            save_disabled = not bool(st.session_state.plan_steps)
            if st.button("Save as Template", disabled=save_disabled):
                st.session_state.templates.append(
                    {
                        "name": tmpl_name.strip() or "Untitled",
                        "plan_steps": list(st.session_state.plan_steps),
                        "code": st.session_state.plan_code,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                st.rerun()

        # T016, T024 — Plan tab + Execute Plan wiring
        with tab_plan:
            if st.session_state.plan_steps:
                for i, step in enumerate(st.session_state.plan_steps, 1):
                    st.write(f"{i}. {step}")
                if st.button("Execute Plan"):
                    result = execute_code(
                        st.session_state.plan_code, st.session_state.df
                    )
                    st.session_state.analysis_result = result
                    st.rerun()
            else:
                st.info("Send a data analysis question to generate a plan.")

        # T017 — Code tab
        with tab_code:
            if st.session_state.plan_code:
                st.code(st.session_state.plan_code, language="python")
            else:
                st.info("Code will appear here after your first analysis.")


# ---------------------------------------------------------------------------
# T012, T013, T014 — CSV panel (bottom-left)
# Replaces render_sidebar(); no sidebar references remain.
# ---------------------------------------------------------------------------
def render_csv_panel(col) -> None:
    with col:
        st.subheader("User Data Set")
        uploaded = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            label_visibility="collapsed",
        )
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
        rows, cols_count = st.session_state.df.shape
        st.caption(f"{rows:,} rows × {cols_count} columns")


# ---------------------------------------------------------------------------
# T023, T025 — Results panel (bottom-right)
# ---------------------------------------------------------------------------
def render_results_panel(col) -> None:
    with col:
        st.subheader("Analysis Results")
        result = st.session_state.analysis_result
        if result is None:
            st.info("Analysis results will appear here after you execute a plan.")
        elif result["output_type"] == "figure":
            st.pyplot(result["content"])
        elif result["output_type"] == "dataframe":
            st.dataframe(result["content"], use_container_width=True)
        elif result["output_type"] == "text":
            st.markdown(result["content"])
        elif result.get("error"):
            st.error(f"Code execution error: {result['error']}")
        else:
            st.info("Analysis results will appear here after you execute a plan.")


# ---------------------------------------------------------------------------
# T004, T011, T014, T021, T025 — Main app: 4-quadrant layout
# ---------------------------------------------------------------------------
init_session_state()

top_left, top_right = st.columns([1, 1])
bottom_left, bottom_right = st.columns([1, 1])

render_chat_panel(top_left)
render_tabs_panel(top_right)
render_csv_panel(bottom_left)
render_results_panel(bottom_right)
