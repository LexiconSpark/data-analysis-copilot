# Implementation Plan: AI Data Analyst Copilot — Core UI Layout

**Branch**: `002-analyst-copilot-ui` | **Date**: 2026-03-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-analyst-copilot-ui/spec.md`

## Summary

Refactor `app.py` from a centered single-column chat + sidebar layout to a 2-column, 2-row Streamlit layout (wide mode). The four quadrants are: chat interface (top-left), Plan/Code/Template tabs (top-right), CSV table with upload (bottom-left), and analysis results placeholder (bottom-right). The LLM chain is upgraded to return structured output (chat response + analysis plan + code) using `ChatOpenAI.with_structured_output()`.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: streamlit>=1.31.0, langchain>=0.3.0, langchain-openai>=0.1.0, langchain-community>=0.3.0, openai>=1.0.0, pandas>=2.0.0, pydantic>=2.0.0
**Storage**: Session state only (in-memory, no persistence)
**Testing**: pytest
**Target Platform**: Local web browser (Streamlit dev server)
**Project Type**: web-app (Streamlit single-file)
**Performance Goals**: AI response with plan + code in < 30 seconds
**Constraints**: All 4 quadrants visible on 1920×1080 without outer-page scrolling; no new external storage dependencies
**Scale/Scope**: Single-user local tool; single `app.py` file

## Constitution Check

*No `constitution.md` found in `.specify/memory/` — constitution not yet configured. Gates skipped.*

## Project Structure

### Documentation (this feature)

```text
specs/002-analyst-copilot-ui/
├── plan.md              # This file
├── research.md          # Phase 0 decisions
├── data-model.md        # Phase 1 entities
├── quickstart.md        # Phase 1 setup guide
├── contracts/           # Phase 1 UI contracts
│   └── ui-contract.md
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
app.py                  # Single-file Streamlit app (modified in place)
requirements.txt        # Updated: streamlit>=1.31.0, pydantic>=2.0.0
config.toml             # Existing model/temperature config (unchanged)
.env                    # API key (unchanged)
```

**Structure Decision**: Single-file app (`app.py`) — same as the existing `001-gpt4o-chatbot` implementation. The app is scoped tightly enough that splitting into modules would be premature. All changes are in-place modifications to `app.py` and a `requirements.txt` version bump.

## Phase 0: Research Decisions

See [research.md](research.md) for full rationale. Summary of key decisions:

| # | Decision | Choice |
|---|----------|--------|
| 1 | Layout engine | `st.columns([1, 1])` × 2 rows, `layout="wide"` |
| 2 | Structured AI output | `ChatOpenAI.with_structured_output()` + Pydantic |
| 3 | Chat scroll container | `st.container(height=400)` (Streamlit ≥1.31.0) |
| 4 | Template storage | `st.session_state.templates` list (session-scoped) |
| 5 | Code display | `st.code(code, language="python")` |
| 6 | Results panel | Placeholder `st.info()` message for MVP |

## Phase 1: Design

### Data Model

See [data-model.md](data-model.md). New entities added on top of `001-gpt4o-chatbot` baseline:

- **AnalysisResponse** — structured LLM output (chat reply + plan steps + code)
- **AnalysisTemplate** — saved plan+code pair with a user-provided name
- **AnalysisResult** — placeholder result container (output_type, content)

### UI Contracts

See [contracts/ui-contract.md](contracts/ui-contract.md). Four quadrant contracts:

- **ChatQuadrant** — scrollable conversation with input at bottom
- **TabsQuadrant** — Template / Plan / Code tabs; Plan and Code populate after AI response
- **CSVQuadrant** — dataframe display with upload control and row counter
- **ResultsQuadrant** — placeholder info box; will host charts/tables in future

### Implementation Breakdown

#### 1. Layout Refactor (`app.py`)

```python
st.set_page_config(page_title="Data Analyst Copilot", layout="wide")

top_left, top_right = st.columns([1, 1])
bottom_left, bottom_right = st.columns([1, 1])
```

- Remove `st.sidebar` — CSV moves to `bottom_left`
- Move chat rendering into `top_left` with `st.container(height=400)`
- Add tabs in `top_right`: `st.tabs(["Template", "Plan", "Code"])`
- Bottom-right gets a `st.info("Analysis results will appear here…")` placeholder

#### 2. Structured Output Chain

Replace `StrOutputParser()` with a Pydantic-based structured output:

```python
from pydantic import BaseModel

class AnalysisResponse(BaseModel):
    chat_reply: str
    plan_steps: list[str]
    code: str

llm_structured = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)
structured_chain = prompt | llm_structured.with_structured_output(AnalysisResponse)
```

The system prompt is updated to instruct the model to always emit `plan_steps` (numbered steps) and `code` (executable Python). If the query is conversational (no analysis needed), `plan_steps` and `code` are empty lists/strings.

#### 3. Session State Additions

```python
st.session_state.plan_steps   # list[str] — current plan steps
st.session_state.plan_code    # str — current generated code
st.session_state.templates    # list[dict] — {"name": str, "steps": list, "code": str}
```

#### 4. Tab Rendering

- **Template tab**: List saved templates; "Save current plan as template" input + button
- **Plan tab**: `st.write` each numbered step from `st.session_state.plan_steps`
- **Code tab**: `st.code(st.session_state.plan_code, language="python")`

#### 5. Execute Plan Button

In the Plan tab, add an "Execute Plan" button. When clicked, run the stored code via `exec()` with a sandboxed namespace and capture `matplotlib` figures or `pandas` DataFrames to display in the bottom-right results panel.

#### 6. CSV Quadrant

Move `render_sidebar()` logic to a `render_csv_panel(col)` function, rendered inside `bottom_left`.

### Updated System Prompt

```
You are an expert data analyst assistant. When the user asks a data analysis question:
1. Provide a clear chat_reply summarizing your approach and key findings.
2. Provide plan_steps: a numbered list of concrete analysis steps.
3. Provide code: complete, executable Python code using pandas/matplotlib that implements the plan.

If the question is conversational (not data analysis), return an empty list for plan_steps and an empty string for code.
```

## Quickstart

See [quickstart.md](quickstart.md).

## Verification

1. `pip install -r requirements.txt` — ensure streamlit>=1.31.0 and pydantic>=2.0.0 install cleanly
2. `streamlit run app.py` — confirm wide layout renders 4 quadrants without scrolling at 1080p
3. Send "plot the relationship between A and B" — verify Plan tab populates with steps, Code tab shows Python code
4. Upload a CSV — verify bottom-left table updates to new data
5. Click "Save as Template" — verify template appears in Template tab list on next interaction
6. Send a conversational message ("What is today?") — verify plan_steps is empty and only chat_reply renders
