# Interface Contract: Internal Streamlit Application

**Date**: 2026-03-08
**Status**: Session 1 (no external APIs)

## Overview

The data-analysis-copilot is a single-user Streamlit web application. Session 1 has no external APIs or system boundaries. All interfaces are internal to the Streamlit session.

---

## Internal Session Interfaces (Not External APIs)

### 1. LangGraph → Streamlit Session State

**Direction**: Graph writes state; UI reads state

**Data Type**: `AnalysisState` TypedDict

**Guarantee**: State persists across Streamlit reruns via MemorySaver checkpointer (thread_id = session_id)

### 2. UI → Graph Invocation

**Direction**: Streamlit calls graph.invoke()

**Input**: User query (str) + CSV (pd.DataFrame)

**Output**: Updated AnalysisState (plan_steps, code, execution_result, execution_blocks)

### 3. Code Execution Sandbox

**Direction**: Generated code runs in isolated namespace

**Inputs**: Code (str), DataFrame (df), imports (pd, plt)

**Outputs**: fig (matplotlib), result (DataFrame/text), or error (str)

---

## External Integrations (Session 1)

### OpenAI API (Existing)

**Used for**: Plan generation, code generation, report summary

**Authentication**: `OPENAI_API_KEY` environment variable

**No contract changes**: Session 1 uses existing LangChain integration (build_chain function)

### LangSmith (Optional)

**Used for**: Tracing graph nodes and LLM calls

**Authentication**: `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` environment variables

**Activation**: Via `LANGCHAIN_TRACING_V2=true` only; no programmatic SDK calls

**No new interface**: LangGraph automatically records spans; no custom tracing code needed

---

## Session 2 (Reserved)

External APIs may be added in Session 2:
- Database (cross-session template persistence)
- Custom LLM service (if OpenAI replaced)
- Webhook (for async job results)

Contracts for these will be defined when Session 2 spec is created.

---

## Conclusion

Session 1 is self-contained. All interfaces are internal to the Streamlit application. No breaking changes needed for Session 2 template persistence (templates will be serialized the same way, just stored in DB instead of local JSON).
