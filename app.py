import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict

import matplotlib
import numpy as np
import openai
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
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
# LangGraph State Schema (T006)
# ---------------------------------------------------------------------------
class AnalysisState(TypedDict):
    """Complete state for LangGraph analysis workflow."""

    # Inputs
    user_query: str  # User's natural language analysis request
    df_csv: str  # Serialized CSV text of active DataFrame

    # Plan Generation
    plan_steps: list[str]  # Ordered list of analysis steps
    plan_approved: Optional[bool]  # None=awaiting, True=approved, False=rejected

    # Code Execution
    code: str  # Generated or rewritten Python code
    execution_result: Optional[dict]  # {"output_type": str, "content": Any, "error": Optional[str]}
    retry_count: int  # Number of code rewrite attempts (0–3, reserved for Session 2)

    # Report
    report_summary: str  # Natural language summary of analysis
    execution_blocks: list[dict]  # List of {"step": str, "code": str, "result": dict}

    # Control
    cancelled: bool  # Whether user cancelled (reserved for Session 2)
    error: Optional[str]  # Unrecoverable error message
    plan_revised: bool  # True after node_revise_plan has run (prevents infinite revise loop)

    # Session 2 (Reserved)
    templates: list[dict]  # Session-scoped saved templates


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
# Utility functions: Artifacts & Templates (T002, T003)
# ---------------------------------------------------------------------------
def clear_artifacts() -> None:
    """Clear artifacts folder (charts from previous runs)."""
    artifacts_dir = Path(__file__).parent / "artifacts"
    if artifacts_dir.exists():
        for file in artifacts_dir.glob("*"):
            try:
                file.unlink()
            except Exception:
                pass  # Gracefully skip if file is locked


def save_template(name: str, plan_steps: list, code: str) -> str:
    """Save analysis template as JSON. Returns template_id."""
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    template_id = str(uuid.uuid4())
    template_data = {
        "name": name or "Untitled",
        "plan_steps": plan_steps,
        "code": code,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(templates_dir / f"{template_id}.json", "w") as f:
        json.dump(template_data, f)
    return template_id


def load_templates() -> list:
    """Load all templates from JSON files."""
    templates_dir = Path(__file__).parent / "templates"
    templates = []
    if templates_dir.exists():
        for json_file in sorted(templates_dir.glob("*.json"), reverse=True):
            try:
                with open(json_file) as f:
                    template = json.load(f)
                    template["id"] = json_file.stem  # Add file name as ID
                    templates.append(template)
            except Exception:
                pass  # Gracefully skip corrupted files
    return templates


def load_template(template_id: str) -> Optional[dict]:
    """Load a specific template by ID."""
    try:
        template_file = Path(__file__).parent / "templates" / f"{template_id}.json"
        if template_file.exists():
            with open(template_file) as f:
                return json.load(f)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# T008 — Session state initialisation (Updated for LangGraph)
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    # Create runtime directories (T002, T003)
    (Path(__file__).parent / "artifacts").mkdir(exist_ok=True)
    (Path(__file__).parent / "templates").mkdir(exist_ok=True)

    # Only clear artifacts once at session start, not on every rerun
    if "artifacts_cleared" not in st.session_state:
        clear_artifacts()
        st.session_state.artifacts_cleared = True

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

    # LangGraph checkpointer initialization (T008)
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = st.session_state.session_id
    if "graph_run_state" not in st.session_state:
        st.session_state.graph_run_state = None

    # 002-analyst-copilot-ui additions
    if "plan_steps" not in st.session_state:
        st.session_state.plan_steps = []
    if "plan_code" not in st.session_state:
        st.session_state.plan_code = ""
    if "templates" not in st.session_state:
        st.session_state.templates = []
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    # Active tab for UI navigation
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "templates"  # Default to Templates tab

    # Session 2: Plan approval with interrupt/resume
    if "awaiting_approval" not in st.session_state:
        st.session_state.awaiting_approval = False
    if "awaiting_approval_thread_id" not in st.session_state:
        st.session_state.awaiting_approval_thread_id = None
    if "plan_steps_pending" not in st.session_state:
        st.session_state.plan_steps_pending = []  # Steps shown in Plan tab during approval


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
    st.session_state.awaiting_approval = False
    st.session_state.awaiting_approval_thread_id = None
    st.session_state.plan_steps_pending = []


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
    # Import scipy and scikit-learn for code execution
    import scipy  # noqa: F401
    from sklearn import preprocessing, metrics, ensemble, decomposition  # noqa: F401

    namespace: dict = {
        "df": df,
        "pd": pd,
        "plt": plt,
        "np": np,
        "sns": sns,
        "scipy": scipy,
        "preprocessing": preprocessing,
        "metrics": metrics,
        "ensemble": ensemble,
        "decomposition": decomposition,
    }
    try:
        exec(code, namespace)  # noqa: S102
        # Prefer explicitly assigned `fig`; fall back to any open figure
        fig = namespace.get("fig") or (
            plt.gcf() if plt.get_fignums() else None
        )
        if fig is not None:
            # Save figure to file instead of storing in state (msgpack can't serialize Figure objects)
            artifacts_dir = Path(__file__).parent / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            fig_path = artifacts_dir / "result_figure.png"
            fig.savefig(fig_path, dpi=100, bbox_inches="tight")
            plt.close("all")
            return {"output_type": "figure", "content": str(fig_path), "error": None}
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
# LangGraph: Node functions (T010, T011, T012, T013, T014)
# ---------------------------------------------------------------------------
def node_generate_plan(state: AnalysisState) -> AnalysisState:
    """Generate analysis plan using LLM (T010)."""
    if state.get("cancelled"):
        return state
    try:
        # Build prompt for plan generation
        prompt_text = f"""Generate a numbered step-by-step analysis plan for this question:

{state['user_query']}

The user's data has columns: {', '.join(st.session_state.df.columns)}

Provide 3-5 specific, concrete analysis steps. Format as a numbered list (1. Step name, 2. Step name, etc.).
Each step should be executable and specific."""

        response = build_chain().invoke(
            {
                "input": prompt_text,
                "system_context": "You are a data analysis expert. Generate a detailed step-by-step analysis plan.",
                "history": [],
            }
        )
        # Extract plan_steps from response
        if hasattr(response, "plan_steps"):
            state["plan_steps"] = response.plan_steps
        else:
            # Fallback: parse response text
            state["plan_steps"] = [
                line.strip()
                for line in response.chat_reply.split("\n")
                if line.strip() and line[0].isdigit()
            ]
    except Exception as e:
        state["error"] = f"Plan generation failed: {str(e)}"
    return state


def node_await_approval(state: AnalysisState) -> Command:
    """Interrupt for plan approval (Session 2)."""
    if state.get("cancelled"):
        return Command(update={}, goto=END)
    resume_value = interrupt({"plan_steps": state["plan_steps"]})
    if resume_value.get("action") == "approve":
        new_steps = resume_value.get("plan_steps", state["plan_steps"])
        return Command(update={"plan_approved": True, "plan_steps": new_steps}, goto="generate_code")
    else:  # cancel
        return Command(update={"plan_approved": False, "cancelled": True}, goto=END)


def node_generate_code(state: AnalysisState) -> AnalysisState:
    """Generate executable Python code for the plan (T012)."""
    if state.get("cancelled"):
        return state
    try:
        retry_count = state.get("retry_count", 0)
        retry_context = ""
        if retry_count > 0 and state.get("execution_result", {}).get("error"):
            prev_error = state["execution_result"]["error"]
            retry_context = f"""
IMPORTANT - Previous attempt failed: {prev_error}
Previous code:
```python
{state.get('code', '')}
```
Fix this error in the rewritten code.
"""
        plan_text = "\n".join(
            f"{i + 1}. {step}" for i, step in enumerate(state["plan_steps"])
        )
        prompt_text = f"""Generate complete, executable Python code to implement this analysis plan:

Plan:
{plan_text}

Data Info:
{state['df_csv'][:500]}...

IMPORTANT: Available libraries in the execution environment:
- pandas as pd (DataFrame operations)
- numpy as np (numerical operations)
- matplotlib.pyplot as plt (plotting)
- seaborn as sns (statistical plotting)
- scipy (scientific computing)
- scikit-learn (machine learning){retry_context}

Generate Python code that:
- ONLY uses the libraries listed above
- Do NOT use pd.np (deprecated) - use np directly instead
- Stores any figure in variable named `fig`
- Stores results in variable named `result` (DataFrame or text)
- Is complete and executable
- No imports needed (all libraries pre-imported)"""

        response = build_chain().invoke(
            {
                "input": prompt_text,
                "system_context": "You are a Python data analysis expert. Generate production-quality code that works with the available libraries. NEVER use deprecated patterns like pd.np.",
                "history": [],
            }
        )
        if hasattr(response, "code"):
            state["code"] = response.code
        else:
            state["code"] = response.chat_reply
    except Exception as e:
        state["error"] = f"Code generation failed: {str(e)}"
    return state


def node_execute_code(state: AnalysisState) -> Command:
    """Execute generated code and route to retry, revise, or render (T013)."""
    if state.get("cancelled"):
        return Command(update={}, goto=END)
    try:
        clear_artifacts()
        result = execute_code(state["code"], st.session_state.df)
    except Exception as e:
        result = {"output_type": "none", "content": None, "error": str(e)}

    has_error = bool(result.get("error"))
    new_retry_count = state.get("retry_count", 0) + (1 if has_error else 0)

    if not has_error:
        return Command(update={"execution_result": result}, goto="render_report")
    if new_retry_count < 3:
        return Command(update={"execution_result": result, "retry_count": new_retry_count}, goto="generate_code")
    if not state.get("plan_revised"):
        return Command(update={"execution_result": result, "retry_count": new_retry_count}, goto="revise_plan")
    return Command(update={"execution_result": result, "retry_count": new_retry_count,
                            "error": f"Code failed after plan revision: {result['error']}"}, goto="render_report")


def node_revise_plan(state: AnalysisState) -> Command:
    """Revise plan after code failure and restart code generation (Session 2)."""
    if state.get("cancelled"):
        return Command(update={}, goto=END)
    try:
        prev_error = (state.get("execution_result") or {}).get("error", "Unknown error")
        original_plan = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["plan_steps"]))
        response = build_chain().invoke({
            "input": f"This plan failed with error '{prev_error}':\n{original_plan}\nRevise it to use a simpler approach.",
            "system_context": "You are a data analysis expert. Revise the plan to avoid the error.",
            "history": [],
        })
        new_steps = response.plan_steps if (hasattr(response, "plan_steps") and response.plan_steps) else [
            l.strip() for l in response.chat_reply.split("\n") if l.strip() and l[0].isdigit()
        ]
        st.session_state.plan_steps = new_steps  # Update UI immediately
        return Command(update={"plan_steps": new_steps, "plan_revised": True, "retry_count": 0,
                                "error": None, "execution_result": None}, goto="generate_code")
    except Exception as e:
        return Command(update={"plan_revised": True, "error": f"Plan revision failed: {e}"}, goto="render_report")


def node_render_report(state: AnalysisState) -> AnalysisState:
    """Generate report summary and execution blocks (T014)."""
    if state.get("cancelled"):
        return state
    try:
        # Generate summary
        summary_prompt = f"""Summarize the results of this analysis in 1-2 sentences:

Question: {state['user_query']}
Results: {str(state['execution_result'].get('content', 'No output'))[:500]}"""

        summary_response = build_chain().invoke(
            {
                "input": summary_prompt,
                "system_context": "Summarize analysis results concisely.",
                "history": [],
            }
        )
        state["report_summary"] = (
            summary_response.chat_reply
            if hasattr(summary_response, "chat_reply")
            else str(summary_response)
        )

        # Build execution blocks
        state["execution_blocks"] = [
            {
                "step": (
                    " ".join(state["plan_steps"])
                    if state["plan_steps"]
                    else "Analysis"
                ),
                "code": state["code"],
                "result": state["execution_result"],
            }
        ]
    except Exception as e:
        state["error"] = f"Report generation failed: {str(e)}"
    return state


def build_analysis_graph():
    """Build LangGraph state graph (T007). NOT cached - must use session checkpointer."""
    graph = StateGraph(AnalysisState)

    # Add nodes
    graph.add_node("generate_plan", node_generate_plan)
    graph.add_node("await_approval", node_await_approval)
    graph.add_node("generate_code", node_generate_code)
    graph.add_node("execute_code", node_execute_code)
    graph.add_node("revise_plan", node_revise_plan)
    graph.add_node("render_report", node_render_report)

    # Add edges (straight-line flow with Command + goto routing)
    graph.add_edge(START, "generate_plan")
    graph.add_edge("generate_plan", "await_approval")
    # await_approval → generate_code or END (via Command)
    graph.add_edge("generate_code", "execute_code")
    # execute_code → generate_code | revise_plan | render_report (via Command)
    # revise_plan → generate_code (via Command)
    graph.add_edge("render_report", END)

    # Compile with checkpointer (per-session)
    return graph.compile(checkpointer=st.session_state.checkpointer)


# ---------------------------------------------------------------------------
# T015, T016, T017 — UI Panels (Chat, Tabs, CSV)
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
                    with st.spinner("Running analysis…"):
                        try:
                            # Invoke LangGraph analysis workflow (T015)
                            graph = build_analysis_graph()
                            run_thread_id = str(uuid.uuid4())
                            st.session_state.awaiting_approval_thread_id = run_thread_id
                            config = {"configurable": {"thread_id": run_thread_id}}

                            initial_state = {
                                "user_query": last_user_msg,
                                "df_csv": st.session_state.df.to_csv(index=False),
                                "plan_steps": [],
                                "plan_approved": None,
                                "code": "",
                                "execution_result": None,
                                "retry_count": 0,
                                "report_summary": "",
                                "execution_blocks": [],
                                "cancelled": False,
                                "error": None,
                                "plan_revised": False,
                                "templates": [],
                            }

                            result = graph.invoke(initial_state, config=config)

                            # Detect interrupt: graph.get_state(config).next is non-empty when paused
                            graph_snapshot = graph.get_state(config)
                            is_interrupted = bool(graph_snapshot.next)

                            if is_interrupted:
                                st.session_state.awaiting_approval = True
                                st.session_state.plan_steps = result.get("plan_steps", [])
                                st.session_state.plan_steps_pending = result.get("plan_steps", [])
                                response = None  # Don't append message yet; wait for approval
                            else:
                                st.session_state.graph_run_state = result
                                st.session_state.plan_steps = result.get("plan_steps", [])
                                st.session_state.plan_code = result.get("code", "")
                                st.session_state.analysis_result = result.get("execution_result")
                                st.session_state.awaiting_approval = False
                                st.session_state.awaiting_approval_thread_id = None
                                if result.get("cancelled"):
                                    response = "Analysis cancelled."
                                else:
                                    st.session_state.active_tab = "code"
                                    response = result.get("report_summary") or "Analysis complete. See Code tab for details."
                        except Exception as exc:
                            error_msg = format_error(exc)

                if error_msg:
                    st.error(error_msg)
                elif response:  # Only append if not waiting for approval
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                st.session_state.is_loading = False
                st.rerun()

        if st.session_state.awaiting_approval:
            st.warning("Plan ready for review. Open the **Plan tab** → to approve or cancel.")

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
        tab_template, tab_plan, tab_code = st.tabs(["Templates", "Plan", "Code"])

        # T023 — Template tab with load functionality (wire save button in Session 2)
        with tab_template:
            # Load templates from JSON files
            templates = load_templates()
            if templates:
                for tmpl in templates:
                    tc1, tc2 = st.columns([4, 1])
                    tc1.markdown(
                        f"**{tmpl['name']}** — {tmpl['created_at'][:10]}"
                    )
                    if tc2.button("Load", key=f"load_tmpl_{tmpl['id']}"):
                        st.session_state.plan_steps = tmpl["plan_steps"]
                        st.session_state.plan_code = tmpl["code"]
                        st.rerun()
            else:
                st.caption("No saved templates yet.")

            st.divider()
            # Save as Template button (T020, not wired yet per Session 1 spec)
            tmpl_name = st.text_input(
                "Template name",
                key="tmpl_name_input",
                placeholder="e.g. Correlation Analysis",
            )
            save_disabled = not bool(st.session_state.plan_steps)
            if st.button("Save as Template", disabled=save_disabled):
                # Save to JSON (T020)
                save_template(
                    tmpl_name.strip() or "Untitled",
                    st.session_state.plan_steps,
                    st.session_state.plan_code,
                )
                st.success("Template saved!")
                st.rerun()

        # T016 — Plan tab (approval mode in Session 2, view mode otherwise)
        with tab_plan:
            if st.session_state.awaiting_approval:
                # APPROVAL MODE: editable steps + Approve/Cancel buttons
                st.info("Review and edit the plan, then approve or cancel.")
                plan_steps_pending = st.session_state.plan_steps_pending or st.session_state.plan_steps

                # Editable text inputs for each step
                edited_steps = []
                for i, step in enumerate(plan_steps_pending):
                    edited_step = st.text_input(
                        f"Step {i + 1}",
                        value=step,
                        key=f"plan_step_edit_{i}",
                    )
                    edited_steps.append(edited_step)

                st.divider()

                # Save as Template section
                st.markdown("#### Save as Template")
                template_name = st.text_input(
                    "Template name",
                    key="tmpl_name_approval",
                    placeholder="e.g. Correlation Analysis",
                )
                if st.button("Save as Template", key="save_tmpl_approval"):
                    save_template(
                        template_name.strip() or "Untitled",
                        edited_steps,
                        st.session_state.plan_code,
                    )
                    st.success("Template saved!")

                st.divider()

                # Approve and Cancel buttons
                col_approve, col_cancel = st.columns(2)
                with col_approve:
                    if st.button("✓ Approve Plan", type="primary", use_container_width=True):
                        # Resume graph with edited steps
                        try:
                            graph = build_analysis_graph()
                            config = {"configurable": {"thread_id": st.session_state.awaiting_approval_thread_id}}
                            result = graph.invoke(
                                Command(resume={"action": "approve", "plan_steps": edited_steps}),
                                config=config,
                            )
                            st.session_state.graph_run_state = result
                            st.session_state.plan_steps = result.get("plan_steps", [])
                            st.session_state.plan_code = result.get("code", "")
                            st.session_state.analysis_result = result.get("execution_result")
                            st.session_state.awaiting_approval = False
                            st.session_state.awaiting_approval_thread_id = None

                            if result.get("cancelled"):
                                response = "Analysis cancelled."
                            else:
                                st.session_state.active_tab = "code"
                                response = result.get("report_summary") or "Analysis complete. See Code tab for details."

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error resuming analysis: {str(e)}")

                with col_cancel:
                    if st.button("✗ Cancel", use_container_width=True):
                        st.session_state.awaiting_approval = False
                        st.session_state.awaiting_approval_thread_id = None
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Analysis cancelled.",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        st.rerun()

            elif st.session_state.plan_steps:
                # VIEW MODE: read-only plan display
                st.markdown("### Analysis Plan")
                for i, step in enumerate(st.session_state.plan_steps, 1):
                    st.write(f"{i}. {step}")

                st.divider()

                # Save as Template section in view mode
                st.markdown("#### Save as Template")
                template_name = st.text_input(
                    "Template name",
                    key="tmpl_name_view",
                    placeholder="e.g. Correlation Analysis",
                )
                if st.button("Save as Template", key="save_tmpl_view"):
                    save_template(
                        template_name.strip() or "Untitled",
                        st.session_state.plan_steps,
                        st.session_state.plan_code,
                    )
                    st.success("Template saved!")

            else:
                # EMPTY STATE
                st.info("Send a data analysis question in the chat to generate a plan.")

        # T016, T017 — Code tab with execution blocks and report (always visible)
        with tab_code:
            if st.session_state.graph_run_state:
                state = st.session_state.graph_run_state

                # Display report summary at top
                if state.get("report_summary"):
                    st.markdown("### Report Summary")
                    st.markdown(state["report_summary"])
                    st.divider()
                else:
                    st.warning("No report summary available")

                # Display execution blocks (always visible, not collapsed)
                if state.get("execution_blocks"):
                    st.markdown("### Execution Details")
                    for i, block in enumerate(state["execution_blocks"], 1):
                        st.markdown(f"**Step {i}: {block.get('step', 'Analysis')}**")
                        # Code (always visible)
                        code_str = block.get("code", "# No code")
                        if code_str:
                            st.code(code_str, language="python")
                        # Result
                        result = block.get("result", {})
                        if result:
                            if result.get("error"):
                                st.error(f"Error: {result['error']}")
                            elif result.get("output_type") == "figure":
                                # Figure is saved as file path, display using st.image
                                fig_path = result.get("content")
                                if fig_path and Path(fig_path).exists():
                                    try:
                                        st.image(fig_path)
                                    except Exception as e:
                                        st.error(f"Could not display figure: {e}")
                                else:
                                    st.warning(f"Figure file not found: {fig_path}")
                            elif result.get("output_type") == "dataframe":
                                st.dataframe(
                                    result["content"], use_container_width=True
                                )
                            elif result.get("output_type") == "text":
                                st.markdown(result["content"])
                            elif result.get("output_type") == "none":
                                st.info("(No output)")
                        st.divider()
                else:
                    st.warning("No execution blocks available")

                # Error display
                if state.get("error"):
                    st.error(f"Analysis Error: {state['error']}")
            else:
                st.info("Code and results will appear here after running an analysis.")


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
            # Figure is saved as file path, display using st.image
            fig_path = result.get("content")
            if fig_path and Path(fig_path).exists():
                try:
                    st.image(fig_path)
                except Exception as e:
                    st.error(f"Could not display figure: {e}")
            else:
                st.warning(f"Figure file not found: {fig_path}")
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
