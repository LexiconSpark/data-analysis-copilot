# UI Contract: GPT-4o Conversational Chatbot

**Branch**: `001-gpt4o-chatbot` | **Date**: 2026-03-03

This document defines the Streamlit UI contract: what components exist, what state they read/write, and what behavior they guarantee. It serves as the interface contract between the UI layer and the LangChain/LLM layer.

---

## Page Layout

```
┌───────────────────────────────────────────┐
│  GPT-4o Chatbot            [New Chat]     │  ← st.title + st.button
├───────────────────────────────────────────┤
│                                           │
│  🤖  Hello! How can I help you today?    │  ← st.chat_message("assistant")
│                                           │
│  👤  What is the capital of France?      │  ← st.chat_message("user")
│                                           │
│  🤖  The capital of France is Paris.     │  ← st.chat_message("assistant")
│                                           │
│           (scrollable area)              │
├───────────────────────────────────────────┤
│  [Type a message...               ] [→]  │  ← st.chat_input
└───────────────────────────────────────────┘
```

---

## Component Contracts

### 1. Chat Input — `st.chat_input`

| Property | Value |
|----------|-------|
| Placeholder | `"Type a message…"` |
| Disabled when | `st.session_state.is_loading == True` |
| On submit | Triggers message send flow (see State Transitions below) |
| Validation | Empty or whitespace-only input is never submitted |

**Guarantee**: When `is_loading` is `True`, the input widget renders as disabled and submission is blocked.

---

### 2. Message Thread — `st.chat_message`

| Property | Value |
|----------|-------|
| User role label | `"user"` |
| Assistant role label | `"assistant"` |
| Content rendering | `st.markdown()` (supports bold, code blocks, lists) |
| Order | Chronological, top-to-bottom |
| Auto-scroll | Streamlit auto-scrolls to bottom on new message |

**Guarantee**: All messages in `st.session_state.messages` are rendered in order on every re-run. Messages are never removed except on "New Chat."

---

### 3. Loading Indicator

| Property | Value |
|----------|-------|
| Shown when | `st.session_state.is_loading == True` |
| Component | `st.chat_message("assistant")` containing `st.markdown("▌")` or `st.spinner()` |
| Cleared when | Response received; `is_loading` set to `False` |

**Guarantee**: A visible indicator is always present while a response is being generated (FR-005).

---

### 4. New Chat Button — `st.button`

| Property | Value |
|----------|-------|
| Label | `"New Chat"` |
| Placement | Top-right of page (sidebar or header) |
| On click | Clears `st.session_state.messages`, `st.session_state.history`, resets `is_loading` to `False` |
| Availability | Always visible, including during loading |

**Guarantee**: Clicking "New Chat" resets all session state. The next message will have no prior context (FR-007).

---

### 5. Error Display — `st.error`

| Property | Value |
|----------|-------|
| Shown when | LLM call raises an exception |
| Content | User-friendly message (no raw stack traces) |
| Retry guidance | Message instructs user to try again |
| Position | Inline, below the last user message |

**Guarantee**: No raw Python exceptions are surfaced to the user. All errors produce a human-readable `st.error()` message (FR-006).

---

## Session State Schema

The full `st.session_state` schema consumed and mutated by the UI:

```python
{
    "messages": list[dict],           # [{"role": str, "content": str, "timestamp": str}, ...]
    "history": ChatMessageHistory,    # LangChain history object (not rendered directly)
    "session_id": str,                # UUID, stable within a session
    "is_loading": bool,               # True while awaiting LLM response
    "created_at": datetime,           # Session start time
}
```

---

## Message Send Flow

Describes the sequence of state changes and renders triggered by a user submitting a message:

```
1. User submits message via st.chat_input
2. Validate: content is non-empty/non-whitespace → if invalid, do nothing
3. Append {"role": "user", "content": prompt} to st.session_state.messages
4. Append user message to st.session_state.history
5. Set st.session_state.is_loading = True
6. Streamlit re-runs → UI renders user message + loading indicator
7. Invoke RunnableWithMessageHistory with {input: prompt}
   ├── On success: response text received
   │   ├── Append {"role": "assistant", "content": response} to st.session_state.messages
   │   └── Set st.session_state.is_loading = False
   └── On failure: exception caught
       ├── Display st.error(user_friendly_message)
       └── Set st.session_state.is_loading = False
8. Streamlit re-runs → UI renders response (or error)
```

---

## New Chat Flow

```
1. User clicks "New Chat" button
2. st.session_state.messages = []
3. st.session_state.history = ChatMessageHistory()  (fresh instance)
4. st.session_state.is_loading = False
5. Streamlit re-runs → conversation thread is empty
6. Welcome message or empty placeholder shown
```

---

## Config Integration Contract

`MODEL`, `TEMPERATURE`, and `SYSTEM_PROMPT` are module-level constants defined at the top of `app.py`. The LangChain chain is built once per app process via `@st.cache_resource`:

```python
# Constants read directly from module scope — no config file needed
MODEL = "gpt-4o"
TEMPERATURE = 0.7
SYSTEM_PROMPT = "You are a helpful assistant."

@st.cache_resource
def build_chain() -> RunnableWithMessageHistory:
    # Constructs and caches the full LangChain chain using the constants above
    ...
```

**Guarantee**: Constants are fixed at app startup. `build_chain()` is called once per app process and reused across all user interactions and Streamlit reruns.

---

## Accessibility & Responsiveness

- **SC-003 (first-use without instructions)**: The input placeholder and button labels are self-explanatory
- **SC-004 (mobile to desktop)**: Streamlit's default layout is responsive; no custom CSS required for MVP
- Markdown rendering supports code blocks and lists for rich AI responses
