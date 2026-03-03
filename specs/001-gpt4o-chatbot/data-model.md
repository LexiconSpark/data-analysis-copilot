# Data Model: GPT-4o Conversational Chatbot

**Branch**: `001-gpt4o-chatbot` | **Date**: 2026-03-03

---

## Overview

All data is held in-memory for the duration of a single browser session. There is no persistent storage. On page reload or "New Chat," all session data is cleared.

---

## Entity: Message

A single turn in the conversation — either a user submission or an AI response.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `role` | `str` | One of `"user"`, `"assistant"` | Who sent the message |
| `content` | `str` | Non-empty, non-whitespace | The text of the message |
| `timestamp` | `datetime` | Auto-set on creation | When the message was added to the session |

**Validation rules**:
- `content` must not be empty or whitespace-only (FR-008)
- `role` must be either `"user"` or `"assistant"`

**Storage**: Stored as dictionaries in `st.session_state.messages` (a `list[dict]`) for Streamlit display rendering. The same history is also tracked by LangChain's `InMemoryChatMessageHistory` object stored in `st.session_state.history`.

**Example**:
```python
{"role": "user", "content": "What is the capital of France?", "timestamp": "2026-03-03T10:00:00"}
{"role": "assistant", "content": "The capital of France is Paris.", "timestamp": "2026-03-03T10:00:02"}
```

---

## Entity: ConversationSession

The full ordered sequence of messages exchanged since the interface was loaded or last reset.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `session_id` | `str` | UUID, auto-generated | Unique identifier for the LangChain history store |
| `messages` | `list[Message]` | Ordered, append-only | All messages in the current session |
| `history` | `ChatMessageHistory` | LangChain object | LangChain-native history used by `RunnableWithMessageHistory` |
| `is_loading` | `bool` | Default `False` | Whether a response is currently being generated |
| `created_at` | `datetime` | Set on session init | When the session was started |

**State transitions**:
```
[Empty] → (user sends message) → [Has messages, loading=True]
         → (response received) → [Has messages, loading=False]
         → (user sends again)  → [Has messages, loading=True] → ...
         → (user clicks "New Chat") → [Empty]
```

**Storage**: All fields live in `st.session_state`. Cleared on page reload or "New Chat" action.

**Initialization** (on first load):
```python
if "messages" not in st.session_state:
    st.session_state.messages = []          # display list
    st.session_state.history = ChatMessageHistory()  # LangChain history
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.is_loading = False
    st.session_state.created_at = datetime.utcnow()
```

---

## Entity: AppConfig

Application-level configuration defined as module-level constants at the top of `app.py`. Not mutable at runtime.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `MODEL` | `str` | Non-empty string | OpenAI model name (e.g., `"gpt-4o"`) |
| `TEMPERATURE` | `float` | 0.0 – 2.0 inclusive | Sampling temperature for the LLM |
| `SYSTEM_PROMPT` | `str` | Non-empty string | Fixed system message prepended to every conversation |

**Source**: Module-level constants at the top of `app.py` (edit the file to change values):
```python
MODEL = "gpt-4o"
TEMPERATURE = 0.7
SYSTEM_PROMPT = "You are a helpful assistant."
```

**Validation rules**:
- `MODEL` must be a non-empty string matching a valid OpenAI model name
- `TEMPERATURE` must be a float in range [0.0, 2.0]
- `SYSTEM_PROMPT` must be a non-empty string

---

## Runtime Data Flow

```
User types message
       │
       ▼
Validate: non-empty content
       │
       ▼
Append to st.session_state.messages (role="user")
Append to st.session_state.history (LangChain)
Set st.session_state.is_loading = True
       │
       ▼
RunnableWithMessageHistory.invoke(
    {"input": content},
    config={"configurable": {"session_id": session_id}}
)
       │
       ▼
Response text received
       │
       ▼
Append to st.session_state.messages (role="assistant")
LangChain history already updated internally
Set st.session_state.is_loading = False
       │
       ▼
Streamlit re-renders UI with updated messages
```

---

## Data Boundaries

| Boundary | What is stored | What is NOT stored |
|----------|---------------|-------------------|
| In-session | All messages, LangChain history, loading state | Nothing persisted to disk or database |
| Between sessions | Nothing | All conversation history is lost on reload |
| `app.py` constants | Model name, temperature, system prompt | API keys, secrets |
| .env | OPENAI_API_KEY | Any conversation data |
