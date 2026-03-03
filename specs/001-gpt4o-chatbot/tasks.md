# Tasks: GPT-4o Conversational Chatbot

**Input**: Design documents from `specs/001-gpt4o-chatbot/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/ui-contract.md, research.md
**Branch**: `001-gpt4o-chatbot`

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each increment.

> **Note**: No tests were requested in the spec — test tasks are omitted. The spec's Independent Test for each story serves as the manual validation criterion.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (no mutual dependency)
- **[US1/US2/US3]**: Maps task to its user story

---

## Phase 1: Setup

**Purpose**: Create all project files and declare dependencies before any code is written.

- [x] T001 Create `requirements.txt` at project root listing: `streamlit>=1.28.0`, `langchain>=0.3.0`, `langchain-openai>=0.1.0`, `langchain-community>=0.3.0`, `openai>=1.0.0`, `python-dotenv>=1.0.0`
- [x] T002 [P] Create `.gitignore` at project root excluding `.env`, `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`
- [x] T003 [P] Create `.env.example` at project root with the single line `OPENAI_API_KEY=sk-your-key-here` as an onboarding hint

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure — API wiring, LangChain chain construction, and session state schema. Must be complete before any user story can be implemented.

**⚠️ CRITICAL**: All user story phases depend on this phase being complete.

- [x] T004 Create scaffold of `app.py` at project root: add all imports (`streamlit`, `uuid`, `datetime`, `os`, `dotenv`, `langchain_openai`, `langchain_core`, `langchain_community`), call `st.set_page_config(page_title="GPT-4o Chatbot", layout="centered")`, and declare the constants block at the top: `MODEL = "gpt-4o"`, `TEMPERATURE = 0.7`, `SYSTEM_PROMPT = "You are a helpful assistant."`
- [x] T005 Add API key loading in `app.py` immediately after constants: call `load_dotenv()`, read `OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")`, and if missing display `st.error("OPENAI_API_KEY not found — add it to your .env file.")` then call `st.stop()`
- [x] T006 Add `build_chain()` function decorated with `@st.cache_resource` in `app.py` — construct `ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), MessagesPlaceholder("history"), ("human", "{input}")])`, pipe into `ChatOpenAI(model=MODEL, temperature=TEMPERATURE, openai_api_key=OPENAI_API_KEY)` then `StrOutputParser()`, and wrap with `RunnableWithMessageHistory(chain, lambda _: st.session_state.history, input_messages_key="input", history_messages_key="history")`
- [x] T007 Add `init_session_state()` function in `app.py` that guards each key with `if "key" not in st.session_state`: set `messages = []` (display list), `history = ChatMessageHistory()` (LangChain object), `session_id = str(uuid.uuid4())`, `is_loading = False`; call `init_session_state()` at the top of the app body before any rendering

**Checkpoint**: `app.py` launches without error (`streamlit run app.py`), shows the page title, and handles a missing API key gracefully.

---

## Phase 3: User Story 1 — Send a Message and Receive a Response (Priority: P1) 🎯 MVP

**Goal**: A user types a message, sees a loading indicator, and receives an AI-generated response rendered in a scrollable conversation thread. Empty input is rejected. API errors are caught and shown as friendly messages.

**Independent Test**: Run `streamlit run app.py`, type "What is the capital of France?", press Enter, and verify an accurate response appears below the user's message.

### Implementation

- [x] T008 [US1] Add `render_conversation()` function in `app.py` — iterate `st.session_state.messages`; for each dict render `st.chat_message(msg["role"])` containing `st.markdown(msg["content"])`; call this function in the main app body before the chat input
- [x] T009 [US1] Add empty-state welcome in `app.py` inside `render_conversation()` — when `st.session_state.messages` is empty, render `st.chat_message("assistant")` containing `st.markdown("Hello! How can I help you today?")`
- [x] T010 [US1] Add loading indicator in `app.py` after `render_conversation()` — if `st.session_state.is_loading` is `True`, render `st.chat_message("assistant")` containing `st.spinner("Thinking…")`
- [x] T011 [US1] Add `get_ai_response(prompt: str) -> str` function in `app.py` — call `build_chain().invoke({"input": prompt}, config={"configurable": {"session_id": st.session_state.session_id}})` and return the response string; let exceptions propagate to the caller
- [x] T012 [US1] Add `format_error(exc: Exception) -> str` function in `app.py` — map `openai.AuthenticationError` → `"API key is invalid or missing. Check your .env file."`, `openai.RateLimitError` → `"Rate limit reached. Please wait a moment and try again."`, `openai.APITimeoutError` → `"Request timed out. Please try again."`, any other exception → `"Something went wrong. Please try again."`; return the mapped string
- [x] T013 [US1] Add main chat interaction loop in `app.py` (after render calls): capture `prompt = st.chat_input("Type a message…", disabled=st.session_state.is_loading)`; if `prompt` is truthy and `prompt.strip()` is non-empty: append `{"role": "user", "content": prompt.strip(), "timestamp": datetime.utcnow().isoformat()}` to `st.session_state.messages`, set `st.session_state.is_loading = True`, call `st.rerun()`; on rerun when `is_loading` is `True`: wrap `get_ai_response()` in try/except, on success append assistant message dict and set `is_loading = False`, on exception call `st.error(format_error(exc))` and set `is_loading = False`

**Checkpoint**: US1 is fully functional. Single-turn send/receive works with loading state and error handling. Run the independent test above to validate before proceeding.

---

## Phase 4: User Story 2 — Multi-Turn Conversation with Memory (Priority: P2)

**Goal**: Each AI response is informed by the full prior conversation history of the current session.

**Independent Test**: Send "My name is Alex." then ask "What is my name?" — the chatbot must answer "Alex" without being reminded.

### Implementation

- [x] T014 [US2] Verify the history getter in `build_chain()` in `app.py` — confirm the lambda passed to `RunnableWithMessageHistory` returns `st.session_state.history` (the live `ChatMessageHistory` object from session state, not a copy); `RunnableWithMessageHistory` will append both user and assistant messages to it automatically after each invoke
- [x] T015 [US2] Confirm `get_ai_response()` in `app.py` passes `{"configurable": {"session_id": st.session_state.session_id}}` as the `config` argument to `invoke()` — this is required for `RunnableWithMessageHistory` to look up the correct history object; verify by running the independent test (send name → ask name → confirm reply)

**Checkpoint**: US2 verified. The chain accumulates history in `st.session_state.history` across Streamlit reruns and passes it to GPT-4o on every call.

---

## Phase 5: User Story 3 — Start a New Conversation (Priority: P3)

**Goal**: A user can clear all conversation history and start fresh.

**Independent Test**: After a multi-turn conversation about "Paris," click "New Chat," then ask "What city were we discussing?" — the chatbot must have no memory of Paris.

### Implementation

- [x] T016 [US3] Add `reset_session()` function in `app.py` — set `st.session_state.messages = []`, reassign `st.session_state.history = ChatMessageHistory()` (a fresh instance clears all LangChain history), set `st.session_state.is_loading = False`
- [x] T017 [US3] Add "New Chat" button in `app.py` header — use `col1, col2 = st.columns([4, 1])`, render `col1.title("GPT-4o Chatbot")` and `col2.button("New Chat", on_click=reset_session)`; this replaces any bare `st.title()` call from T004

**Checkpoint**: US3 functional. Clicking "New Chat" clears all messages and history; the next message starts with no prior context.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Align documentation with the clarified design (no `config.toml`) and run final end-to-end validation.

- [x] T018 Update summary and Architecture Overview in `specs/001-gpt4o-chatbot/plan.md` — replace `config.toml` references with "constants at top of `app.py`"; remove `config.toml` from the source code tree diagram; update the Key Implementation Decisions table row for "Config format"
- [x] T019 [P] Update AppConfig entity in `specs/001-gpt4o-chatbot/data-model.md` — replace `config.toml` loading pattern with "MODULE-LEVEL constants (`MODEL`, `TEMPERATURE`, `SYSTEM_PROMPT`) defined at top of `app.py`"; remove TOML code examples
- [x] T020 [P] Update Config Integration Contract section in `specs/001-gpt4o-chatbot/contracts/ui-contract.md` — replace `load_config()` / `tomllib` pattern with "constants read directly from module scope; `@st.cache_resource` applies only to `build_chain()`"
- [x] T021 Update setup instructions in `specs/001-gpt4o-chatbot/quickstart.md` — remove Step 5 "Review or adjust the model config" TOML instructions; replace with "Edit the constants `MODEL`, `TEMPERATURE`, and `SYSTEM_PROMPT` at the top of `app.py` before starting the application"
- [ ] T022 Final end-to-end smoke test — follow `quickstart.md` setup steps from scratch, then run all three user story independent tests in sequence: (1) "What is the capital of France?" → accurate response, (2) "My name is Alex" → "What is my name?" → "Alex", (3) multi-turn about Paris → New Chat → ask about city → no memory

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Requires Phase 1 complete; **blocks all user story phases**
- **Phase 3 (US1)**: Requires Phase 2 complete
- **Phase 4 (US2)**: Requires Phase 3 complete — verifies memory already wired in Phase 2
- **Phase 5 (US3)**: Requires Phase 3 complete — can run in parallel with Phase 4
- **Phase 6 (Polish)**: Requires Phases 3, 4, and 5 complete

### Task-Level Dependencies Within Phases

| Task | Depends On |
|------|-----------|
| T005 | T004 |
| T006 | T005 |
| T007 | T004 |
| T008, T009, T010 | T007 (can run in parallel with each other) |
| T011, T012 | T006 (can run in parallel with each other) |
| T013 | T008, T009, T010, T011, T012 |
| T014, T015 | T013 |
| T016 | T007 |
| T017 | T016 |
| T018, T019, T020, T021 | T017 (can run in parallel with each other) |
| T022 | T018, T019, T020, T021 |

### Parallel Opportunities

```bash
# Phase 1 (after T001):
T002 (.gitignore) || T003 (.env.example)

# Phase 2 (sequential — each unlocks the next):
T004 → T005 → T006
T007 (can overlap with T005 since T007 only needs T004)

# Phase 3 (after Phase 2):
T008 || T009 || T010   # rendering concerns, no shared state writes
T011 || T012           # pure functions, no shared state

# Phase 5 (US3, after Phase 3):
T016 → T017            # sequential within story
# Phase 4 and Phase 5 can run concurrently:
(T014, T015) || (T016, T017)

# Phase 6 (after all stories):
T018 || T019 || T020 || T021
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Run the US1 independent test
5. Ship or demo this increment

### Incremental Delivery

| Phase | Delivers |
|-------|---------|
| 1 + 2 | App launches, API connected, session ready |
| 3 (US1) | ✅ Working chatbot — single turn, error handling |
| 4 (US2) | ✅ Verified memory — multi-turn contextual dialogue |
| 5 (US3) | ✅ New Chat — full reset capability |
| 6 (Polish) | ✅ Docs aligned, smoke test passed |

---

## Notes

- All source code lives in a single `app.py` — no subdirectories needed
- `MODEL`, `TEMPERATURE`, and `SYSTEM_PROMPT` are module-level constants in `app.py` (no `config.toml`)
- `st.session_state.messages` (display list) and `st.session_state.history` (LangChain object) serve different purposes — do not conflate
- `@st.cache_resource` on `build_chain()` means the chain is initialized once per app process, not per rerun
- `RunnableWithMessageHistory` automatically appends both user and assistant messages to the history object — no manual append needed
- Commit after each phase checkpoint to preserve working increments
