# Research: AI Data Analyst Copilot — Core UI Layout

**Branch**: `002-analyst-copilot-ui` | **Phase**: 0 | **Date**: 2026-03-05

---

## Decision 1: Streamlit Layout Engine

**Decision**: Use `st.set_page_config(layout="wide")` with two `st.columns([1, 1])` calls — one for each row — to produce a stable 2-column, 2-row grid.

**Rationale**: Streamlit's `st.columns()` is the canonical way to create multi-column layouts. Calling it twice (once for the top row, once for the bottom row) produces four distinct rendering regions. The `layout="wide"` page config uses the full viewport width, which is required for all four quadrants to be usable at 1080p without horizontal scrolling.

**Implementation pattern**:
```python
st.set_page_config(page_title="Data Analyst Copilot", layout="wide")

top_left, top_right = st.columns([1, 1])
bottom_left, bottom_right = st.columns([1, 1])

with top_left:
    render_chat_panel()

with top_right:
    render_tabs_panel()

with bottom_left:
    render_csv_panel()

with bottom_right:
    render_results_panel()
```

**Alternatives considered**:
- `st.sidebar` for CSV — used in `001-gpt4o-chatbot` but does not fit the 4-quadrant design
- Custom CSS grid — overly complex; Streamlit `st.columns` is stable and officially supported
- `streamlit-extras` layout components — adds a third-party dependency for no benefit over native columns

---

## Decision 2: Structured LLM Output (Plan + Code Extraction)

**Decision**: Use `ChatOpenAI.with_structured_output()` with a Pydantic v2 `BaseModel` to obtain a typed `AnalysisResponse` containing `chat_reply`, `plan_steps`, and `code` from a single LLM call.

**Rationale**: The UI requires the AI response to be decomposed into three distinct parts (chat reply for the chat panel, plan steps for the Plan tab, code for the Code tab). `with_structured_output()` is LangChain's recommended approach for this (introduced in `langchain-openai>=0.1.0`). It uses OpenAI's function-calling / tool-use API under the hood, which reliably returns structured JSON matching the Pydantic schema. This is preferable to post-processing a free-text response with regex or JSON extraction.

**Pydantic model**:
```python
from pydantic import BaseModel, Field

class AnalysisResponse(BaseModel):
    chat_reply: str = Field(description="Natural language summary of the analysis")
    plan_steps: list[str] = Field(default_factory=list, description="Numbered analysis steps")
    code: str = Field(default="", description="Complete executable Python code")
```

**Chain construction**:
```python
llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)
chain = prompt | llm.with_structured_output(AnalysisResponse)
```

**Alternatives considered**:
- Free-text response + JSON extraction — fragile; depends on consistent model formatting
- Two separate LLM calls (one for chat, one for code) — doubles latency and cost
- LangChain `PydanticOutputParser` — deprecated in favor of `with_structured_output()`

---

## Decision 3: Scrollable Chat Container

**Decision**: Use `st.container(height=400)` to constrain the chat message area to a fixed height with overflow scrolling.

**Rationale**: Without a height-constrained container, the chat history grows and pushes the bottom rows off-screen — violating FR-011 (all quadrants visible without scrolling). `st.container(height=N)` was introduced in Streamlit 1.31.0 and is the official mechanism for fixed-height scrollable content regions. The `requirements.txt` version floor must be raised from `>=1.28.0` to `>=1.31.0` to guarantee this API.

**Implementation**:
```python
with top_left:
    chat_container = st.container(height=400)
    with chat_container:
        render_conversation()

    # Input stays outside the scrollable container (pinned below it)
    if prompt := st.chat_input("Ask about your data…"):
        handle_user_input(prompt)
```

**Alternatives considered**:
- CSS `overflow-y: scroll` via `st.markdown` with unsafe HTML — fragile, breaks on Streamlit updates
- `streamlit-chat` or other third-party widgets — adds dependency; native solution is sufficient

---

## Decision 4: Template Persistence

**Decision**: Store templates in `st.session_state.templates` as a list of dicts. Templates are session-scoped only (lost on page reload).

**Rationale**: The spec's Assumptions section explicitly states "Template persistence is session-based for MVP; cross-session storage is a future enhancement." This keeps the implementation dependency-free (no file I/O, no database). Each template stores `name`, `plan_steps`, and `code`.

**Template schema**:
```python
{
    "name": "Filter & Plot Temperature",
    "plan_steps": ["Filter rows...", "Choose plot type...", "Generate plot..."],
    "code": "import pandas as pd\n...",
    "created_at": "2026-03-05T02:44:00"
}
```

**Alternatives considered**:
- JSON file on disk — adds file I/O; premature for MVP scope
- SQLite via `sqlite3` stdlib — unnecessary complexity for session-scoped data

---

## Decision 5: Code Display

**Decision**: Use `st.code(code_string, language="python")` for syntax-highlighted code display in the Code tab.

**Rationale**: Streamlit's built-in `st.code()` renders syntax-highlighted code blocks with a copy button. No additional dependency. Supports Python highlighting natively.

**Alternatives considered**:
- `st.markdown` with fenced code blocks — no copy button; syntax highlighting depends on theme
- `streamlit-ace` editor — adds a dependency; read-only display doesn't need a full editor

---

## Decision 6: Execute Plan (Code Execution)

**Decision**: Run the AI-generated code via Python `exec()` with an isolated namespace containing the current `pd.DataFrame`. Capture `matplotlib.figure.Figure` objects and `pd.DataFrame` results from the namespace; render them in the bottom-right results panel using `st.pyplot()` or `st.dataframe()`.

**Rationale**: `exec()` with a controlled namespace (`{"df": st.session_state.df, "pd": pd, "plt": plt}`) is the simplest way to execute AI-generated pandas/matplotlib code inside a Streamlit app. The namespace isolation prevents overwriting global state. Output capture by inspecting `namespace["fig"]` or `namespace["result"]` is a well-established pattern for code-execution sandboxes at this scale.

**Security note**: Code execution is limited to a local development tool for trusted engineers — not a multi-user or public deployment. No sandboxing beyond namespace isolation is required for MVP scope.

**Implementation**:
```python
def execute_code(code: str, df: pd.DataFrame) -> dict:
    import matplotlib.pyplot as plt
    namespace = {"df": df, "pd": pd, "plt": plt}
    exec(code, namespace)
    return namespace
```

**Alternatives considered**:
- Subprocess execution — adds overhead and IPC complexity
- Docker sandbox — vastly over-engineered for a local single-user tool
- LangChain `PythonREPLTool` — designed for agents, not for inline Streamlit execution

---

## Decision 7: Results Panel (MVP)

**Decision**: Render a `st.info("Analysis results will appear here after you execute a plan.")` placeholder in the bottom-right quadrant for MVP. When code has been executed, render captured figures and DataFrames there.

**Rationale**: The spec explicitly designates the bottom-right panel as a placeholder for future analysis results. The placeholder satisfies FR-010 and SC-003 while keeping implementation minimal. The Execute Plan flow (Decision 6) will populate this panel dynamically.

**Alternatives considered**:
- Empty container — provides no guidance to the user about the panel's purpose
- Pre-rendered sample chart — misleading; doesn't reflect the user's actual data

---

## Decision 8: System Prompt Update

**Decision**: Replace the generic `"You are a helpful assistant."` system prompt with a data analyst persona that explicitly instructs the model to populate `plan_steps` and `code` fields for analysis questions, and to leave them empty for conversational queries.

**Updated prompt**:
```
You are an expert data analyst assistant. When the user asks a data analysis question:
1. Provide a clear chat_reply summarizing your approach and key findings.
2. Provide plan_steps: a numbered list of concrete analysis steps.
3. Provide code: complete, executable Python code using pandas and matplotlib that implements the plan.

If the question is purely conversational (not a data analysis request), return an empty list for plan_steps and an empty string for code.

The user's CSV data is provided below.
```

**Alternatives considered**:
- Keep the generic prompt and rely on the Pydantic schema alone — risks empty `plan_steps` on analysis queries
- Separate system prompts for conversational vs. analysis mode — requires intent classification step
