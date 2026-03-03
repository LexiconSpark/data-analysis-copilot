# Implementation Plan: GPT-4o Conversational Chatbot

**Branch**: `001-gpt4o-chatbot` | **Date**: 2026-03-03 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-gpt4o-chatbot/spec.md`

## Summary

Build a browser-based conversational chatbot using Streamlit for the UI and LangChain for the AI orchestration layer. The chatbot uses OpenAI GPT-4o as its language model, maintains full conversation history within a browser session using LangChain's `RunnableWithMessageHistory` pattern, and exposes `MODEL`, `TEMPERATURE`, and `SYSTEM_PROMPT` as named constants at the top of `app.py` for easy adjustment.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: streamlit>=1.28.0, langchain>=0.3.0, langchain-openai>=0.1.0, langchain-community>=0.3.0, openai>=1.0.0, python-dotenv>=1.0.0
**Storage**: In-memory only — `st.session_state` for Streamlit re-run persistence; session cleared on page reload
**Testing**: pytest (unit tests for config loading and message formatting logic)
**Target Platform**: Local browser; deployable to Streamlit Community Cloud
**Project Type**: Web application (Streamlit single-page app)
**Performance Goals**: AI response delivered within 10 seconds for typical messages (SC-001)
**Constraints**: Session-scoped memory only (no cross-session persistence); no authentication; English language
**Scale/Scope**: Single user per browser session; no concurrent user management required

## Constitution Check

*No project constitution exists. No gates to enforce.*

Post-design re-check: N/A (no constitution).

## Project Structure

### Documentation (this feature)

```text
specs/001-gpt4o-chatbot/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── ui-contract.md
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
/
├── app.py              # Streamlit entry point — all UI and chat logic (MODEL, TEMPERATURE, SYSTEM_PROMPT constants at top)
├── requirements.txt    # Pinned Python dependencies
├── .env                # OPENAI_API_KEY (git-ignored, never committed)
├── .env.example        # Template for .env (committed)
└── .gitignore          # Excludes .env, .venv, __pycache__, etc.
```

**Structure Decision**: Single-file layout. The chatbot is a focused, ~200–300 line application. A monolithic `app.py` is appropriate for this scope and simplifies Streamlit Cloud deployment. Module splitting is deferred until the app exceeds ~500 lines or gains distinct subsystems.

## Architecture Overview

```
┌─────────────────────────────────────┐
│           Streamlit UI              │
│  st.chat_message  st.chat_input     │
│  st.session_state (message display) │
└──────────────┬──────────────────────┘
               │ user message
               ▼
┌─────────────────────────────────────┐
│      LangChain Chain (LCEL)         │
│  ChatPromptTemplate                 │
│  → ChatOpenAI (gpt-4o)             │
│  → StrOutputParser                  │
└──────────────┬──────────────────────┘
               │ wrapped by
               ▼
┌─────────────────────────────────────┐
│   RunnableWithMessageHistory        │
│   InMemoryChatMessageHistory        │
│   stored in st.session_state.history│
└─────────────────────────────────────┘
               │ reads constants from
               ▼
┌─────────────────────────────────────┐
│   app.py (module-level constants)   │
│   MODEL, TEMPERATURE, SYSTEM_PROMPT │
└─────────────────────────────────────┘
```

## Key Implementation Decisions

| Concern | Decision | Rationale |
|---------|----------|-----------|
| Memory | `RunnableWithMessageHistory` + `InMemoryChatMessageHistory` | `ConversationBufferMemory` deprecated in LangChain 0.3.0; LCEL approach is current standard |
| History persistence across reruns | Store history object in `st.session_state.history` | Survives Streamlit's re-run cycle; cleared on page reload (session-scoped per spec) |
| UI components | `st.chat_input()` + `st.chat_message()` | Native Streamlit chat primitives (v1.24+); purpose-built for conversational UIs |
| LLM integration | `ChatOpenAI` from `langchain-openai` | Officially maintained; supports all GPT-4o variants and temperature control |
| Config format | Module-level constants in `app.py` | Simplest approach; "specified in the file" per user requirement; no extra dependency or file |
| API key storage | `.env` + `python-dotenv` | Standard practice; never committed to version control |
| Input locking | `st.session_state.is_loading` flag | Prevents concurrent submissions while response is generating |
| Error handling | `st.error()` with retry guidance | User-friendly messages for auth errors, rate limits, and timeouts |
| Dependency management | `requirements.txt` | Required by Streamlit Community Cloud |

## Phase 1 Artifacts

- [data-model.md](data-model.md) — Entity definitions for Message, ConversationSession, AppConfig
- [contracts/ui-contract.md](contracts/ui-contract.md) — Streamlit UI component contracts and state schema
- [quickstart.md](quickstart.md) — Setup and run instructions

## Next Steps

Run `/speckit.tasks` to generate the implementation task list.
