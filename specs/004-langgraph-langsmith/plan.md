# Implementation Plan: LangGraph + LangSmith Integration

**Branch**: `004-langgraph-langsmith` | **Date**: 2026-03-08 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-langgraph-langsmith/spec.md`

**Note**: Step 1 (Session 1) — Straight-line happy path. Auto-approve plans, no retry logic yet. Full state schema designed for Session 2 compatibility.

## Summary

Wire LangGraph orchestration and LangSmith observability into an existing Streamlit data analysis app. Session 1 implements a straight-line graph: generate plan → auto-approve → execute code (record errors, no retry) → render report. UI: four-quadrant layout with fixed heights and borders. Templates tab is default; auto-switch to Code tab after run. Code tab shows execution blocks (syntax-highlighted code + markdown results, always visible). Template save/load via JSON (not wired to UI button yet). State schema pre-designed with `retry_count` and `cancelled` fields for Session 2 (retry loop) compatibility.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: streamlit>=1.31.0, langchain>=0.3.0, langchain-openai>=0.1.0, langchain-community>=0.3.0, **langgraph==0.3.18 (to add)**, python-dotenv>=1.0.0, pandas>=2.0.0, matplotlib>=3.7.0, pydantic>=2.0.0, langsmith>=0.1.0
**Storage**: Session state only (Streamlit); templates as JSON files in local `templates/` folder; artifacts (charts) in `artifacts/` folder (cleared per run)
**Testing**: pytest (from CLAUDE.md); test coverage for graph nodes and code execution
**Target Platform**: Web browser via Streamlit
**Project Type**: Streamlit web application (single-file deployment, no new source files)
**Performance Goals**: Plan generation <15s, code execution <30s, template load <5s (from spec Success Criteria)
**Constraints**: All code in `app.py` only; no new source files; Code tab reports always visible (no collapsed sections); graph synchronous (no async)
**Scale/Scope**: Single Streamlit app, up to 10k CSV rows, one session per user (browser), one graph run at a time

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Status**: No constitution file exists yet in `.specify/memory/`. Proceeding with reasonable defaults from CLAUDE.md:
- Python 3.11+ ✓
- Use streamlit (existing) ✓
- Use langchain (existing) ✓
- No new source files (FR-018) ✓
- pytest for testing ✓
- Session state for storage ✓

No violations detected. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
# Single-file Streamlit app (all code in app.py)
app.py                    # Main Streamlit application with LangGraph + UI logic
requirements.txt          # Add langgraph==0.3.18
.env                      # Existing: OPENAI_API_KEY, LANGSMITH_* (env vars)

# New runtime directories (created at app startup)
artifacts/                # Temporary folder for generated chart images (cleared per run)
templates/                # JSON files for saved analysis templates

# Spec documentation
specs/004-langgraph-langsmith/
├── spec.md
├── plan.md                # This file
├── research.md            # Phase 0 output
├── data-model.md          # Phase 1 output
├── quickstart.md          # Phase 1 output
└── checklists/
```

**Structure Decision**: Single-file Streamlit app (FR-018: no new source files). Graph definition, state schema, and UI logic all in `app.py`. Runtime artifacts (`artifacts/`, `templates/`) created at startup. No test files in Session 1 (tests deferred to post-implementation validation).

## Complexity Tracking

No violations. Single-file Streamlit app satisfies all requirements.

---

## Phase 0: Research

**Status**: Complete (no NEEDS CLARIFICATION markers in spec)

**Key Decisions Confirmed**:
- LangGraph for orchestration (vs. direct chain): Required for interrupt/resume plan approval and complex control flow (retry, cancel)
- Command + goto routing: Simplest pattern for imperative graph (plan → approve → execute → report)
- In-memory MemorySaver checkpointer: Sufficient for session-scoped graphs; no persistence layer needed
- JSON templates in local folder: MVP solution; cross-session persistence deferred to Session 2
- All code in app.py: Constraint from FR-018; inline graph definition, state, and UI logic

**Output**: ✓ [research.md](research.md) — Complete

---

## Phase 1: Design

**Prerequisites**: Phase 0 complete

### 1.1 Data Model

**Output**: ✓ [data-model.md](data-model.md) — Complete

Entities defined:
- `AnalysisState`: Full LangGraph state schema (11 fields: user_query, df_csv, plan_steps, plan_approved, code, execution_result, retry_count, report_summary, execution_blocks, cancelled, error, templates)
- `ExecutionBlock`: Code + result for Code tab report
- `AnalysisTemplate`: Saved template structure (JSON format)

### 1.2 Interface Contracts

**Output**: ✓ [contracts/internal.md](contracts/internal.md) — Complete

Project type: Streamlit web application (internal only, no external APIs in Session 1). External integrations documented: OpenAI API, LangSmith (optional).

### 1.3 Quickstart

**Output**: ✓ [quickstart.md](quickstart.md) — Complete

Includes architecture overview, file structure, key components, implementation steps, testing guide, environment variables, and common pitfalls.

### 1.4 Agent Context Update

**Status**: ✓ Complete

CLAUDE.md updated with new technologies: `langgraph==0.3.18`, storage model, and project type.

---

## Re-evaluation: Constitution Check (Post-Phase 1)

**Status**: No violations detected post-design.

Confirmed:
- All code in `app.py` only (FR-018) ✓
- Python 3.11+ ✓
- Streamlit + LangChain + LangGraph stack ✓
- Session state storage, no new persistence layer ✓
- Single-file deployment model ✓
