# Implementation Plan: GPT-4o Chatbot + CSV Context + LangSmith Tracing

**Branch**: `001-gpt4o-chatbot` | **Date**: 2026-03-04 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-gpt4o-chatbot/spec.md`

## Summary

A single-file Streamlit app (`app.py`) wrapping a LangChain LCEL chain backed by OpenAI GPT-4o. Features: multi-turn conversation memory, CSV data as LLM context (default dataset pre-loaded; user can upload), CSV display in sidebar, and LangSmith tracing via environment variables. Configuration via module-level constants; secrets via `.env`.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: streamlit>=1.28.0, langchain>=0.3.0, langchain-openai>=0.1.0, langchain-community>=0.3.0, openai>=1.0.0, python-dotenv>=1.0.0, pandas>=2.0.0, langsmith>=0.1.0
**Storage**: In-memory session state only (no persistence)
**Testing**: pytest (unit), manual smoke test
**Target Platform**: Local / Streamlit Community Cloud
**Project Type**: Web application (single-page Streamlit app)
**Performance Goals**: AI response within 10s for typical messages; CSV upload confirmation within 5s for files ≤5MB
**Constraints**: Single user per session; no auth; CSV capped at 10,000 rows; no code execution engine
**Scale/Scope**: Single-user session-scoped; one active CSV at a time

## Constitution Check

No project constitution defined. No gates to evaluate.

## Project Structure

### Documentation (this feature)

```text
specs/001-gpt4o-chatbot/
├── plan.md              # This file
├── research.md          # Architectural decisions (11 decisions documented)
├── data-model.md        # Entities: Message, ConversationSession, CSVDataset, AppConfig
├── quickstart.md        # Setup and run instructions
├── contracts/
│   └── ui-contract.md   # UI component contracts and session state schema
└── tasks.md             # Implementation tasks (T001–T022 complete; T022 manual)
```

### Source Code (repository root)

```text
app.py                   # Single-file Streamlit application (all logic here)
requirements.txt         # Python dependencies
.env                     # API key + LangSmith keys (git-ignored)
.env.example             # Template for .env
.gitignore
```

**Structure Decision**: Single-file layout. The app is ≤270 lines with a clear, flat function structure.

## Key Architectural Decisions

### 1. LangChain Memory — `RunnableWithMessageHistory`

`ConversationBufferMemory` is deprecated in LangChain 0.3+. The modern pattern uses `RunnableWithMessageHistory` with `ChatMessageHistory` stored in `st.session_state` so history survives Streamlit reruns.

### 2. CSV Context Injection — Runtime Template Variable

The prompt template uses `("system", "{system_context}")`. `build_system_context()` assembles `SYSTEM_PROMPT + CSV data` from `st.session_state.df` and passes it at every `.invoke()` call. This keeps `@st.cache_resource` intact while allowing per-invoke CSV context.

### 3. CSV Display — Sidebar

`st.sidebar` holds the file uploader, dataframe preview, "Reset to default data" button, row/column metadata, and tracing status.

### 4. Default CSV — Always Pre-Loaded

`DEFAULT_CSV_DATA` (columns A, B, C; 10 rows each) is loaded into session state on startup. No empty-state handling needed.

### 5. Config — Module-Level Constants

Model, temperature, system prompt, and LangSmith project name are module-level constants at the top of `app.py`. Secrets via `.env` + `python-dotenv`.

### 6. LangSmith Tracing — Environment Variables + Enhanced Metadata

LangChain auto-traces to LangSmith when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set — no SDK wrappers needed. Each `.invoke()` also passes `run_name`, `tags`, and `metadata` (session ID, CSV shape) for searchable, annotated traces.

### 7. Loading State — Two-Rerun Pattern

Streamlit has no async. Input is blocked during LLM calls via `st.session_state.is_loading`. User submits → `is_loading=True` → `st.rerun()` → loading branch makes API call → `is_loading=False` → `st.rerun()`.

## Data Flow

```
User types message
       │
       ▼
Validate: non-empty
       │
       ▼
Append to st.session_state.messages (role="user")
Set st.session_state.is_loading = True → st.rerun()
       │
       ▼ (loading branch)
build_system_context()
  → SYSTEM_PROMPT + st.session_state.df.to_csv()
       │
       ▼
RunnableWithMessageHistory.invoke(
    {"input": user_msg, "system_context": system_context},
    config={
        "configurable": {"session_id": ...},
        "run_name": "csv-chat",
        "tags": ["streamlit", "csv-context"],
        "metadata": {"session_id": ..., "csv_rows": N, "csv_cols": M},
    }
)  ← trace emitted to LangSmith if LANGSMITH_TRACING=true
       │
       ▼
Append response to st.session_state.messages (role="assistant")
Set st.session_state.is_loading = False → st.rerun()
       │
       ▼ (normal branch)
render_conversation() + st.chat_input()
```

## Session State Schema

| Key | Type | Default | Reset on "New Chat" |
|-----|------|---------|---------------------|
| `messages` | `list[dict]` | `[]` | Yes |
| `history` | `ChatMessageHistory` | new instance | Yes |
| `session_id` | `str` (UUID) | auto-generated | No |
| `is_loading` | `bool` | `False` | Yes |
| `df` | `pd.DataFrame` | `DEFAULT_CSV_DATA` | No |
| `csv_truncated` | `bool` | `False` | No |
