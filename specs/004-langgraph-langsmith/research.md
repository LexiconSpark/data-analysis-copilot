# Research: LangGraph + LangSmith Integration — Phase 0

**Date**: 2026-03-08
**Status**: Complete (no NEEDS CLARIFICATION markers in spec)

## Research Summary

No clarifications were needed. The feature spec from Session 1 provides clear direction on all technical decisions. This document confirms key choices and their rationales.

---

## Decision 1: LangGraph as Orchestration Layer

**Decision**: Use LangGraph (pinned v0.3.18) as the sole orchestration engine for analysis workflows.

**Rationale**:
- **Plan approval control flow**: Requires interrupt/resume mechanism to halt graph at approval step and resume conditionally. LangGraph's interrupt() is perfect fit.
- **Code execution retry loop** (Session 2): Command + goto routing allows imperative retry logic without conditional edge branching.
- **Session state checkpointing**: LangGraph's checkpointer keeps thread-scoped state across Streamlit reruns.
- **Observability**: LangGraph tracing integrates natively with LangSmith via environment variables.

**Alternatives considered**:
- Direct LangChain chain: No interrupt/resume; retry requires complex LCEL patterns
- Orchestration hand-rolled in Streamlit: No checkpointing; manual state serialization
- BullMQ / Celery async tasks: Over-engineered for single-session Streamlit app; would need distributed state

**Verdict**: LangGraph is the only framework providing interrupt/resume + checkpointing needed for this UX.

---

## Decision 2: Command + goto Routing (No Conditionals)

**Decision**: Use LangGraph's Command + goto pattern exclusively; no conditional edges.

**Rationale**:
- **Testability**: Imperative goto calls are easier to trace and test than conditional branching.
- **Session 1 simplicity**: Straight-line flow (plan → auto-approve → execute → report); no branching yet.
- **Session 2 readiness**: Retry logic (rewrite_code → execute_code loop) fits naturally into goto pattern.
- **Code clarity**: `goto("step_name")` is more readable than `if condition → edge_name`.

**Alternatives considered**:
- Conditional edges (if/else on state): Would work; more verbose for simple flows
- State machine pattern: Over-engineered; goto is simpler

**Verdict**: Command + goto satisfies spec requirement FR-002.

---

## Decision 3: In-Memory MemorySaver Checkpointer

**Decision**: Use LangGraph's `MemorySaver` (in-memory, thread-scoped) for checkpointing.

**Rationale**:
- **Session scope**: Streamlit session == browser session; per-session checkpointer is perfect.
- **No persistence needed**: Session 1 templates are session-scoped JSON files; no cross-session graph state.
- **Simplicity**: No database, no file I/O overhead.
- **Default behavior**: LangGraph documentation recommends MemorySaver for single-threaded, session-scoped workflows.

**Alternatives considered**:
- PostgreSQL checkpointer: Over-engineered for MVP; would need DB schema, migrations, cleanup logic
- File-based checkpointer: Unnecessary for session-scoped state; would add I/O

**Verdict**: MemorySaver is sufficient for Session 1 and scalable to Session 2.

---

## Decision 4: JSON File Templates (Local Storage)

**Decision**: Save/load analysis templates as JSON files in a local `templates/` folder.

**Rationale**:
- **Session 1 MVP**: Templates are session-scoped (Session 2 may add persistence); simple JSON matches current data model.
- **No new dependencies**: JSON is built-in; no database required.
- **Portability**: Template files can later be moved to a database without changing the template interface.

**Alternatives considered**:
- Database storage: Over-engineered for Session 1; cross-session persistence is deferred.
- Session state only: Would lose templates on session reset; JSON allows manual export/import.

**Verdict**: JSON files in `templates/` folder satisfy FR-012 (save/load) without over-engineering.

---

## Decision 5: All Code in app.py (No New Source Files)

**Decision**: All graph definition, state schema, UI logic, and helpers remain in `app.py`.

**Rationale**:
- **Requirement FR-018**: Explicit constraint from feature spec.
- **Single-file Streamlit pattern**: Standard deployment model; no build step.
- **Session 1 scope**: Current code base is 415 lines; adding graph + UI changes fits in one file.
- **Session 2 migration**: Refactoring into modules is deferred post-implementation.

**Alternatives considered**:
- Extract graph to `graph.py`: Violates FR-018; would require imports at startup
- Extract state to `state.py`: Violates FR-018; LangGraph TypedDict is lightweight

**Verdict**: Single file satisfies requirement and current project structure.

---

## Decision 6: No Async Execution

**Decision**: Graph runs synchronously within Streamlit's `st.rerun()` lifecycle.

**Rationale**:
- **Streamlit model**: `st.rerun()` is blocking; async would not integrate cleanly.
- **Session 1 scope**: Plan generation (<15s) and code execution (<30s) are acceptable latencies for sync.
- **Simplicity**: Async adds debugging complexity; not needed for MVP.

**Alternatives considered**:
- Async with `asyncio`: Would require wrapping in `st.session_state` lifecycle; over-complicated
- Background jobs: Over-engineered; Streamlit Cloud / Spaces don't natively support background tasks

**Verdict**: Synchronous execution matches Streamlit's single-threaded model.

---

## Decision 7: Templates Folder + Artifacts Folder

**Decision**: Create two runtime folders at app startup:
- `templates/`: JSON files for saved templates
- `artifacts/`: Temporary chart images (cleared per run)

**Rationale**:
- **Templates folder**: Mirrors app design (save/load from UI).
- **Artifacts folder**: Isolates generated images from source; cleanup avoids disk bloat.
- **Startup initialization**: Folders created if missing; no manual setup required.

**Alternatives considered**:
- Store templates in Streamlit secrets: Secrets are read-only config; not suitable for user-created data
- Store artifacts in session state as base64: Would bloat session state; image files are cleaner

**Verdict**: Folder structure matches UX expectations and simplifies file management.

---

## Technical Validations

**LangGraph 0.3.18 Compatibility**:
- Requires: Python 3.9+ ✓ (project uses 3.11+)
- Requires: `langchain>=0.3.0` ✓ (already in requirements.txt)
- No breaking changes from 0.3.17 → 0.3.18 affecting interrupt/resume or checkpointing ✓

**Streamlit Rerun Model**:
- Graph state persists across reruns via checkpointer (thread_id = session_id) ✓
- `st.session_state` holds current tab selection and interrupt response ✓
- No conflicts between Streamlit and LangGraph state models ✓

**LangSmith Tracing**:
- Activated via `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` ✓
- LangGraph automatically records spans for each node when enabled ✓
- No code changes needed to enable/disable (env vars only) ✓

---

## Conclusion

All technical decisions are validated. No clarifications were needed. The straight-line graph architecture satisfies Session 1 requirements and pre-designs for Session 2 (retry loop, cancel, cross-session templates).

**Ready for Phase 1: Design** → `data-model.md`, `contracts/`, `quickstart.md`
