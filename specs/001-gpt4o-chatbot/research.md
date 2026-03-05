# Research: GPT-4o Conversational Chatbot

**Branch**: `001-gpt4o-chatbot` | **Phase**: 0 | **Date**: 2026-03-03

---

## Decision 1: LangChain Memory Architecture

**Decision**: Use `RunnableWithMessageHistory` with `InMemoryChatMessageHistory`, paired with `st.session_state` for Streamlit re-run persistence.

**Rationale**: `ConversationBufferMemory` is deprecated as of LangChain 0.3.0. The modern LCEL-based approach (`RunnableWithMessageHistory`) is the recommended replacement, integrates cleanly with `ChatOpenAI`, and is actively maintained. For a simple Streamlit app, `InMemoryChatMessageHistory` provides session-scoped storage with no external dependencies.

**Implementation pattern**:
```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Store history object in st.session_state to survive Streamlit reruns
if "history" not in st.session_state:
    st.session_state.history = ChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(chain, lambda _: st.session_state.history)
response = chain_with_history.invoke({"input": user_message}, config={"configurable": {"session_id": "default"}})
```

**Alternatives considered**:
- `ConversationBufferMemory` — deprecated in LangChain 0.3.0, raises deprecation warning
- `ConversationBufferWindowMemory` — deprecated; windowing can be added manually if needed
- Manual message list — viable but bypasses LangChain's history abstraction

---

## Decision 2: Streamlit Chat UI Pattern

**Decision**: Use `st.chat_input()` + `st.chat_message()` (available since Streamlit v1.24.0).

**Rationale**: These are the native Streamlit chat primitives, purpose-built for conversational UIs. They handle avatar rendering, message threading, and input focus automatically. Available in all current Streamlit versions (v1.24+).

**Implementation pattern**:
```python
# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capture input (blocks until user submits)
if prompt := st.chat_input("Type a message..."):
    # append, invoke, append response
```

**Alternatives considered**:
- `st.text_input` + `st.button` — works but produces a form-style UX, not a conversational feel
- Third-party chat components — unnecessary complexity for a simple interface

---

## Decision 3: LangChain + OpenAI Integration

**Decision**: Use `ChatOpenAI` from `langchain-openai` package with model `"gpt-4o"`.

**Rationale**: `langchain-openai` is the officially maintained integration package. `ChatOpenAI` supports all GPT-4o variants, streaming, and temperature control. The model name `"gpt-4o"` maps to the latest stable GPT-4o release.

**Key class**:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=config["llm"]["model"], temperature=config["llm"]["temperature"])
```

**Required package**: `langchain-openai>=0.1.0`

**Alternatives considered**:
- Direct `openai` SDK — bypasses LangChain's chain/memory abstractions entirely
- `langchain-community` OpenAI wrapper — legacy; `langchain-openai` is the canonical choice

---

## Decision 4: Config File Format

**Decision**: Use a `config.toml` file at the project root for model name and temperature. Use `.env` for the API key.

**Rationale**: TOML is human-readable, type-safe, and natively supported in Python 3.11+ via `tomllib`. It clearly separates non-secret configuration (model, temperature) from secrets (API key). This matches how Streamlit itself uses TOML for its own config (`.streamlit/config.toml`).

**Config structure**:
```toml
[llm]
model = "gpt-4o"
temperature = 0.7
```

**Loading pattern**:
```python
import tomllib  # Python 3.11+ stdlib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

model = config["llm"]["model"]
temperature = config["llm"]["temperature"]
```

**API key** stays in `.env` (never committed):
```
OPENAI_API_KEY=sk-...
```

**Alternatives considered**:
- YAML — requires `pyyaml` dependency; whitespace-sensitive; no advantage over TOML here
- Single `.env` for everything — poor fit for structured/typed config (temperature as float)
- `pyproject.toml` — conflates app config with packaging metadata; better kept separate

---

## Decision 5: Project Structure

**Decision**: Single-file application (`app.py`) at the project root, with `config.toml` alongside it.

**Rationale**: The chatbot is a focused, single-responsibility app (~200–300 lines). A single-file structure is simpler to deploy (especially to Streamlit Community Cloud), easier to understand, and avoids premature abstraction. Module splitting is appropriate only when the app exceeds ~500 lines or gains distinct subsystems.

**Final structure**:
```
/
├── app.py              # Streamlit app entry point
├── config.toml         # Model name, temperature (committed)
├── requirements.txt    # Pinned dependencies
├── .env                # API key (git-ignored)
└── .gitignore
```

**Alternatives considered**:
- Modular structure (`modules/llm_config.py`, `modules/chat_handler.py`) — over-engineered for scope
- `pyproject.toml` for dependency management — adds complexity without benefit for a deployable app

---

## Decision 6: Python Dependency Management

**Decision**: Use `requirements.txt` with pinned versions.

**Rationale**: Streamlit Community Cloud expects `requirements.txt` in the repository root. It's the simplest approach for a deployable single-project app.

**Minimum dependencies**:
```
streamlit>=1.28.0
langchain>=0.3.0
langchain-openai>=0.1.0
langchain-community>=0.3.0
openai>=1.0.0
python-dotenv>=1.0.0
```

**Alternatives considered**:
- Poetry — adds overhead; `requirements.txt` is sufficient for this scope
- `pyproject.toml` alone — doesn't generate a lock file without additional tooling

---

## Decision 8: CSV Context Injection Pattern

**Decision**: Change the LangChain prompt template's system slot from a hardcoded string to `{system_context}` and pass the assembled string at every `.invoke()` call.

**Rationale**: The chain is `@st.cache_resource` (built once per app process). Baking CSV data into the cached chain would require a chain rebuild on every upload, losing `@st.cache_resource` benefits and resetting the `RunnableWithMessageHistory` history object. Passing `system_context` as a runtime template variable keeps the chain stateless and cached while letting CSV data flow in per-invoke.

**Implementation**:
```python
# Prompt template
("system", "{system_context}")

# At invoke time
chain.invoke({"input": prompt, "system_context": build_system_context()})

# Context builder reads from session state
def build_system_context() -> str:
    csv_text = st.session_state.df.to_csv(index=False)
    return f"{SYSTEM_PROMPT}\n\nCSV data:\n```csv\n{csv_text}\n```"
```

**Alternatives considered**:
- Rebuild chain on CSV upload — drops `@st.cache_resource`, resets history
- Prepend CSV as a human message — pollutes conversation thread and history

---

## Decision 9: CSV Display + Upload Location

**Decision**: Sidebar (`st.sidebar`) with `st.file_uploader`, `st.dataframe`, and a "Reset to default data" button.

**Rationale**: `layout="centered"` gives the chat a fixed-width column; placing a dataframe above the chat would push the input down. The sidebar is the established Streamlit pattern for auxiliary data panels, keeping the chat uncluttered.

---

## Decision 10: Default CSV Data

**Decision**: Define `DEFAULT_CSV_DATA` as a module-level dict constant. Convert to `pd.DataFrame` during `init_session_state()`. Always pre-loaded at startup.

**Rationale**: The user always has data context from message one. No empty-state handling needed. Users can override with upload or revert with the "Reset" button.

---

## Decision 11: LangSmith Tracing

**Decision**: Enable tracing via environment variables only (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`). Enhance trace quality by passing `run_name`, `tags`, and `metadata` in the LangChain `.invoke()` config.

**Rationale**: LangChain 0.3+ automatically emits traces to LangSmith when the env vars are present — no SDK calls or code wrappers are required. Tracing is therefore fully opt-in: setting the vars enables it, omitting them disables it, with zero impact on the app's behaviour either way. Adding `metadata` (session_id, CSV shape) and `tags` costs nothing and makes traces searchable in the LangSmith UI.

**Environment variables** (added to `.env.example`):
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__your-key-here
LANGSMITH_PROJECT=data-analysis-copilot
```

**Code addition** (in `get_ai_response()`):
```python
config={
    "configurable": {"session_id": st.session_state.session_id},
    "run_name": "csv-chat",
    "tags": ["streamlit", "csv-context"],
    "metadata": {
        "session_id": st.session_state.session_id,
        "csv_rows": len(st.session_state.df),
        "csv_cols": len(st.session_state.df.columns),
    },
}
```

**Package**: `langsmith>=0.1.0` added to `requirements.txt` (it is a transitive dependency of LangChain but made explicit).

**Alternatives considered**:
- Manual `@traceable` decorator — not needed; LCEL chain is already auto-traced
- Programmatic `langsmith.Client()` — adds complexity with no benefit over env var approach

---

## Decision 7: Error Handling Strategy

**Decision**: Catch API exceptions in Streamlit and display `st.error()` with a retry option; disable the input during response generation.

**Common failure modes to handle**:
| Error | Cause | User-facing message |
|-------|-------|---------------------|
| `AuthenticationError` | Invalid API key | "API key invalid. Check your .env file." |
| `RateLimitError` | Quota exceeded | "Rate limited. Please wait and try again." |
| `APITimeoutError` | Slow network | "Request timed out. Please retry." |
| General `Exception` | Unexpected failure | "Something went wrong. Please try again." |

**Input locking**: Use `st.session_state.is_loading` flag to disable `st.chat_input` while a response is pending (Streamlit re-renders the widget as disabled when `disabled=True` is passed).
