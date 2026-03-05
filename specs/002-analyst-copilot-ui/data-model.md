# Data Model: AI Data Analyst Copilot — Core UI Layout

**Branch**: `002-analyst-copilot-ui` | **Date**: 2026-03-05

---

## Overview

All data is held in-memory for the duration of a single browser session via `st.session_state`. There is no persistent storage. This feature extends the `001-gpt4o-chatbot` data model with three new entities: `AnalysisResponse`, `AnalysisTemplate`, and `AnalysisResult`.

---

## Existing Entities (unchanged from 001-gpt4o-chatbot)

- **Message** — single conversation turn (`role`, `content`, `timestamp`)
- **ConversationSession** — full session state (`session_id`, `messages`, `history`, `is_loading`)
- **CSVDataset** — active tabular dataset (`df`, `csv_truncated`)
- **AppConfig** — module-level constants (`MODEL`, `TEMPERATURE`, `SYSTEM_PROMPT`)

See [001-gpt4o-chatbot/data-model.md](../001-gpt4o-chatbot/data-model.md) for full definitions.

---

## Entity: AnalysisResponse

The structured output returned by the LLM for every AI invocation. Contains three parts: a chat reply for display, ordered plan steps for the Plan tab, and executable code for the Code tab.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `chat_reply` | `str` | Non-empty | Natural language summary of the analysis or conversational answer |
| `plan_steps` | `list[str]` | May be empty | Ordered analysis steps; empty when query is conversational |
| `code` | `str` | May be empty string | Complete executable Python code; empty when query is conversational |

**Pydantic definition**:
```python
from pydantic import BaseModel, Field

class AnalysisResponse(BaseModel):
    chat_reply: str = Field(description="Natural language summary")
    plan_steps: list[str] = Field(default_factory=list, description="Analysis steps")
    code: str = Field(default="", description="Executable Python code")
```

**Validation rules**:
- `chat_reply` must be non-empty (the LLM always provides a reply)
- `plan_steps` is empty `[]` for conversational messages (no analysis required)
- `code` is `""` for conversational messages

**Session state storage**:
```python
st.session_state.plan_steps  # list[str] — updated after each AI response
st.session_state.plan_code   # str — updated after each AI response
```

**State transitions**:
```
[Empty / previous plan] → (user sends analysis query) → [New plan_steps + code]
[Empty / previous plan] → (user sends conversational query) → [plan_steps=[], code=""]
                        → (user clicks "New Chat") → [plan_steps=[], code=""]
```

---

## Entity: AnalysisTemplate

A saved snapshot of a plan+code pair, named by the user. Used to re-apply previous analyses to new datasets.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `name` | `str` | Non-empty, max 100 chars | User-provided display name |
| `plan_steps` | `list[str]` | Non-empty | The analysis steps at save time |
| `code` | `str` | Non-empty | The code at save time |
| `created_at` | `str` | ISO 8601 datetime string | When the template was saved |

**Validation rules**:
- `name` must be non-empty (user must provide a name before saving)
- Cannot save a template when `plan_steps` is empty (nothing to save)
- Duplicate names are allowed (user may save multiple templates with the same name)

**Session state storage**:
```python
st.session_state.templates  # list[dict] — append-only during session
```

**Initialization**:
```python
if "templates" not in st.session_state:
    st.session_state.templates = []
```

**State transitions**:
```
[Empty list] → (user saves template) → [1 template]
             → (user saves again)    → [2 templates]
             → (New Chat clicked)    → [templates preserved — CSV and templates survive reset]
             → (page reload)         → [Empty list]
```

**Note**: Template list survives "New Chat" (same as CSV dataset — only the conversation is cleared on reset).

---

## Entity: AnalysisResult

The captured output of executing AI-generated code. Holds one or more renderable artifacts.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `output_type` | `str` | One of `"figure"`, `"dataframe"`, `"text"`, `"none"` | Category of the output |
| `content` | `Any` | Type depends on `output_type` | `matplotlib.Figure`, `pd.DataFrame`, `str`, or `None` |
| `error` | `str \| None` | Optional | Error message if code execution failed |

**Session state storage**:
```python
st.session_state.analysis_result  # AnalysisResult dict or None
```

**Initialization**:
```python
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
```

**State transitions**:
```
[None / placeholder] → (Execute Plan succeeds, figure) → [output_type="figure", content=Figure]
                     → (Execute Plan succeeds, df)     → [output_type="dataframe", content=DataFrame]
                     → (Execute Plan fails)             → [output_type="none", error="..."]
                     → (New Chat clicked)               → [None]
```

---

## Session State Schema (complete, this feature)

```python
# Inherited from 001-gpt4o-chatbot
st.session_state.messages         # list[dict] — display messages
st.session_state.history          # ChatMessageHistory — LangChain history
st.session_state.session_id       # str — UUID for LangChain config
st.session_state.is_loading       # bool — response in flight
st.session_state.df               # pd.DataFrame — active dataset
st.session_state.csv_truncated    # bool — truncation flag

# New in 002-analyst-copilot-ui
st.session_state.plan_steps       # list[str] — current analysis plan steps
st.session_state.plan_code        # str — current generated code
st.session_state.templates        # list[dict] — saved templates
st.session_state.analysis_result  # dict | None — result of last code execution
```

---

## Runtime Data Flow

```
User types analysis query
       │
       ▼
RunnableWithMessageHistory.invoke(
    {"input": query, "system_context": build_system_context()}
)  →  AnalysisResponse (structured)
       │
       ├──► chat_reply → append to st.session_state.messages (role="assistant")
       │                  → renders in top-left chat container
       │
       ├──► plan_steps → st.session_state.plan_steps
       │                  → renders in top-right Plan tab
       │
       └──► code       → st.session_state.plan_code
                          → renders in top-right Code tab

User clicks "Execute Plan"
       │
       ▼
exec(st.session_state.plan_code, {"df": st.session_state.df, "pd": pd, "plt": plt})
       │
       ├──► captures namespace["fig"] or namespace["result"]
       └──► stores in st.session_state.analysis_result
                          → renders in bottom-right results panel
```
