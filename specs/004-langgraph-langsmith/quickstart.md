# Quick Start: LangGraph + LangSmith Integration

**Date**: 2026-03-08
**Status**: Development guide for Session 1

This guide outlines the structure and key components of the implementation.

---

## Architecture Overview

```
Streamlit UI
    ↓
Chat Panel (top-left)
    ↓ user submits query
    ↓
Graph Entrypoint: invoke(state)
    ↓
LangGraph State Graph
├── generate_plan (LLM)
├── auto_approve_plan
├── generate_code (LLM)
├── execute_code (sandbox)
├── render_report
└── return final state
    ↓
Streamlit Session State (persisted via checkpointer)
    ↓
Code Tab: render execution blocks
```

---

## File Structure

```
app.py
├── Constants (MODEL, SYSTEM_PROMPT, etc.)
├── Pydantic models (AnalysisResponse, etc.)
├── Utility functions
│   ├── build_system_context()
│   ├── build_chain() — LLM for plan/code generation
│   ├── execute_code() — sandbox execution
│   ├── save_template() — JSON to templates/
│   ├── load_template() — JSON from templates/
│   └── clear_artifacts() — cleanup charts folder
│
├── LangGraph Components
│   ├── AnalysisState TypedDict
│   ├── Graph node functions
│   │   ├── node_generate_plan()
│   │   ├── node_auto_approve()
│   │   ├── node_generate_code()
│   │   ├── node_execute_code()
│   │   └── node_render_report()
│   ├── build_graph() → StateGraph compiled
│   └── Checkpointer initialization
│
├── Streamlit Session State Init
│   ├── init_session_state()
│   ├── Session vars: graph_run, checkpointer, current_result, etc.
│   └── Directory setup: artifacts/, templates/
│
└── Streamlit UI Panels
    ├── render_chat_panel()
    ├── render_tabs_panel()
    ├── render_csv_panel()
    └── render_results_panel()
```

---

## Key Components

### 1. LangGraph State Graph

**What**: Orchestrates the analysis workflow (plan → code → execute → report)

**Where**: Defined in `app.py` after imports

**Scope**: Session 1 = straight-line flow (no branching)

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Define state
class AnalysisState(TypedDict):
    user_query: str
    df_csv: str
    plan_steps: list[str]
    plan_approved: Optional[bool]
    code: str
    execution_result: Optional[dict]
    retry_count: int
    report_summary: str
    execution_blocks: list[dict]
    cancelled: bool
    error: Optional[str]
    templates: list[dict]

# Define nodes
def node_generate_plan(state: AnalysisState) -> AnalysisState:
    # LLM generates plan_steps
    return state

def node_auto_approve(state: AnalysisState) -> AnalysisState:
    # Session 1: auto-approve; Session 2: interrupt for user response
    state["plan_approved"] = True
    return state

# ... more nodes ...

# Build graph
def build_analysis_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("generate_plan", node_generate_plan)
    graph.add_node("auto_approve", node_auto_approve)
    # ... add edges with .add_edge() ...

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
```

### 2. Checkpointer (One Per Session)

**What**: Persists graph state across Streamlit reruns

**Where**: Created in `init_session_state()` once per session

```python
def init_session_state():
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = st.session_state.session_id
    # ... initialize other state vars ...
```

### 3. Code Execution Sandbox

**What**: Runs generated code with DataFrame available; captures output

**Where**: `execute_code()` function (existing, may extend)

**Key fields in state**:
- **code**: Generated Python code string
- **df_csv**: CSV text passed to the code's namespace
- **execution_result**: `{"output_type": "...", "content": ..., "error": None or str}`

```python
def execute_code(code: str, df: pd.DataFrame) -> dict:
    namespace = {"df": df, "pd": pd, "plt": plt}
    try:
        exec(code, namespace)  # noqa: S102
        # Capture outputs: fig (matplotlib), result (DataFrame/text)
        fig = namespace.get("fig")
        result_val = namespace.get("result")
        # ... format and return ...
    except Exception as exc:
        return {"output_type": "none", "content": None, "error": str(exc)}
```

### 4. Templates: JSON Storage

**What**: Session-scoped saved analysis templates

**Where**: `templates/` folder at app root

**Functions**:
- `save_template(name, plan_steps, code)` → JSON file
- `load_template(template_id)` → dict

```python
def save_template(name: str, plan_steps: list, code: str) -> None:
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    template_id = str(uuid.uuid4())
    template_data = {
        "name": name,
        "plan_steps": plan_steps,
        "code": code,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(templates_dir / f"{template_id}.json", "w") as f:
        json.dump(template_data, f)

def load_templates() -> list[dict]:
    templates_dir = Path("templates")
    templates = []
    if templates_dir.exists():
        for json_file in templates_dir.glob("*.json"):
            with open(json_file) as f:
                templates.append(json.load(f))
    return templates
```

### 5. UI Tabs & Tab Switching

**Requirement**: Templates tab (default), Plan tab (auto-activate on approval), Code tab (auto-activate on completion)

**Where**: `render_tabs_panel()` function (extend existing)

**Key state variables**:
- `st.session_state.active_tab`: "templates" | "plan" | "code"
- Tabs rendered as: `st.tabs(["Templates", "Plan", "Code"])`

```python
def render_tabs_panel(col) -> None:
    with col:
        st.subheader("Analysis")

        # Get active tab from session state
        active_idx = {"templates": 0, "plan": 1, "code": 2}.get(
            st.session_state.get("active_tab", "templates"), 0
        )

        tab_templates, tab_plan, tab_code = st.tabs(["Templates", "Plan", "Code"])

        # Update session state based on which tab is selected
        # (Note: Streamlit tabs don't support initial active tab directly;
        # manual tab routing via state vars and selectbox workaround may be needed)

        with tab_templates:
            # Render templates list + load buttons
            # Add save button (wired to session 2)

        with tab_plan:
            # Render plan_steps
            # If awaiting approval: show Approve/Reject/Cancel buttons
            # If approved: show "Save as Template" button

        with tab_code:
            # Render execution_blocks in order
            # Block: step description + syntax-highlighted code + result
            # Always visible (no expanders)
```

### 6. Artifacts Folder (Charts)

**What**: Temporary folder for generated matplotlib figures

**Where**: `artifacts/` folder at app root

**Lifecycle**:
- Created at app startup
- Cleared at the start of each graph run
- Images saved by code execution (e.g., `fig.savefig("artifacts/chart.png")`)

```python
def clear_artifacts() -> None:
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    for file in artifacts_dir.glob("*"):
        file.unlink()
```

---

## Implementation Steps (Session 1)

1. **Update requirements.txt**: Add `langgraph==0.3.18`
2. **Define AnalysisState TypedDict** in app.py (from data-model.md)
3. **Implement graph nodes** (generate_plan, auto_approve, generate_code, execute_code, render_report)
4. **Build and compile graph** with MemorySaver checkpointer
5. **Update session state init** to create checkpointer and directories
6. **Extend UI panels**:
   - Chat: wire graph invocation to query submission
   - Plan tab: auto-activate, show plan_steps
   - Code tab: auto-activate, render execution_blocks
   - Templates tab: list saved templates, load buttons
7. **Implement template save/load logic** (JSON files, not wired to UI button yet)
8. **Test straight-line flow**: query → plan → execute → report

---

## Testing (Post-Implementation)

### Unit Tests
- `test_execute_code()`: Sandbox code execution with DataFrame
- `test_generate_plan()`: LLM plan generation (mock if needed)
- `test_template_save_load()`: JSON file I/O

### Integration Tests
- `test_graph_happy_path()`: Full workflow from query to report
- `test_code_with_error()`: Error handling in execution

### Manual Tests
- Upload CSV, submit query, verify plan tab activates
- Approve plan, verify code tab activates and shows results
- Save template, load template in new query

---

## Environment Variables

Required (existing from CLAUDE.md):
```
OPENAI_API_KEY=sk-...
```

Optional (for LangSmith tracing):
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls_...
LANGCHAIN_PROJECT=data_analysis_copilot
```

---

## Common Pitfalls

1. **Checkpointer scope**: Create once per session; reuse across reruns with same thread_id
2. **State mutation**: LangGraph expects nodes to return updated state; don't mutate in place
3. **Streamlit reruns**: Graph invocation must happen in the chat callback (when query submitted), not in rendering code
4. **Tab switching**: Streamlit tabs don't natively support programmatic switching; use state vars + selectbox workaround or custom JS
5. **CSV serialization**: Ensure df_csv is text (use `.to_csv(index=False)`); don't pass pickle/binary

---

## References

- **LangGraph docs**: https://langchain-ai.github.io/langgraph/
- **Streamlit State Management**: https://docs.streamlit.io/develop/concepts/design/session-state
- **Feature Spec**: [spec.md](spec.md)
- **Data Model**: [data-model.md](data-model.md)
