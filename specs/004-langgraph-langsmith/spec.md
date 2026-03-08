# Feature Specification: LangGraph + LangSmith Integration — AI Data Analyst Co-Pilot

**Feature Branch**: `004-langgraph-langsmith`
**Created**: 2026-03-08
**Status**: Draft
**Input**: User description: "Wiring LangGraph (orchestration) and LangSmith (observability) into an existing Streamlit data analysis app. LangGraph with Command+goto routing, one checkpointer per session, interrupt/resume for plan approval, code retry up to 3× with LLM rewrite, clean cancel, save/load templates."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — End-to-End Guided Analysis (Priority: P1)

A user uploads a CSV, describes what they want to analyze, sees an AI-generated step-by-step plan appear in the Plan tab, approves it, and receives a rendered report in the Code tab with a summary and all execution blocks always visible.

**Why this priority**: This is the entire core loop — plan → approve → execute → report. Without it the product has no value.

**Independent Test**: Can be fully tested by uploading any CSV, submitting an analysis query, approving the auto-generated plan, and verifying a rendered report appears in the Code tab with summary and execution blocks.

**Acceptance Scenarios**:

1. **Given** a CSV is loaded and the user submits an analysis query, **When** the graph runs, **Then** the Plan tab activates automatically with numbered steps and Approve / Reject / Cancel controls.
2. **Given** the plan is displayed with approval controls, **When** the user clicks Approve, **Then** code executes and the Code tab activates showing a report summary followed by all execution blocks.
3. **Given** the report is shown, **When** the user views the Code tab, **Then** all execution blocks are permanently visible — no collapsed or expandable sections.

---

### User Story 2 — Code Execution Retry with LLM Rewrite (Priority: P2)

Generated code fails to execute. The graph automatically rewrites the code using the LLM and retries up to 3 times without user intervention. If all retries fail, the graph falls back to revising the plan itself and presents the revised plan to the user for re-approval.

**Why this priority**: Silent code failures would make the tool unusable for real analysis. Automated recovery is essential.

**Independent Test**: Can be tested by crafting a query likely to produce broken code, verifying that the system retries automatically, and confirming fallback to plan revision after 3 failures.

**Acceptance Scenarios**:

1. **Given** code execution fails on the first attempt and retry_count < 3, **When** the graph processes the failure, **Then** the LLM rewrites the code and execution is retried automatically with no user action required.
2. **Given** code execution fails 3 consecutive times, **When** max retries are exhausted, **Then** the graph falls back to revising the plan and the user sees an updated plan with re-approval controls.
3. **Given** a code rewrite succeeds on retry attempt 2, **When** execution completes, **Then** the report renders normally in the Code tab and the retry count is not surfaced to the user.

---

### User Story 3 — Cancel at Any Point (Priority: P2)

A user can cancel an in-progress analysis at any point — during plan generation, while awaiting approval, or during code execution — and the graph exits cleanly with no partial results displayed.

**Why this priority**: Users must always be in control; an uninterruptible long-running process would be unacceptable.

**Independent Test**: Can be tested by clicking Cancel at each distinct graph stage and verifying the UI returns to idle state cleanly each time.

**Acceptance Scenarios**:

1. **Given** the graph is awaiting plan approval, **When** the user clicks Cancel, **Then** the graph exits and the Plan tab shows a neutral idle message.
2. **Given** the graph is executing code, **When** the user clicks Cancel, **Then** execution stops and the UI returns to the idle state with no partial results shown.
3. **Given** any cancellation occurs, **When** the graph exits, **Then** no partial analysis results or error states are left visible in any tab.

---

### User Story 4 — Save and Load Analysis Templates (Priority: P3)

A user saves an approved plan as a named template from the Plan tab. In a future session, they load the template from the Templates tab and re-execute the analysis on new data without re-entering the original query.

**Why this priority**: Reusability of recurring analyses reduces repeated effort and adds durable value beyond single-use queries.

**Independent Test**: Can be tested by saving a template after plan approval, resetting the chat session, loading the template, and verifying the plan steps are restored and executable.

**Acceptance Scenarios**:

1. **Given** a plan has been approved, **When** the user enters a name and clicks "Save as Template" on the Plan tab, **Then** the template appears in the Templates tab list.
2. **Given** a saved template exists in the Templates tab, **When** the user clicks Load, **Then** the plan steps populate the Plan tab and the user can execute without querying the LLM again.

---

### User Story 5 — LangSmith Observability (Priority: P4)

Every graph run is automatically traced in LangSmith at the node level. Traces are enabled by environment variables only. If tracing credentials are absent, the app runs without errors.

**Why this priority**: Observability supports debugging, performance monitoring, and future cost attribution — all important for a team tool.

**Independent Test**: Can be verified by setting LANGSMITH_API_KEY and LANGSMITH_PROJECT, running any analysis, and checking the LangSmith project dashboard for node-level spans.

**Acceptance Scenarios**:

1. **Given** LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY, and LANGCHAIN_PROJECT are set, **When** any graph run completes, **Then** a trace appears in LangSmith with at least one span per graph node.
2. **Given** LangSmith environment variables are absent, **When** any graph run completes, **Then** the app operates correctly with no tracing-related errors surfaced to the user.

---

### Edge Cases

- What happens when a Streamlit rerun fires while the graph is mid-execution?
- How does the checkpointer preserve graph state across rapid Streamlit reruns?
- What if the user submits a new query while a previous graph run is still active?
- What if code execution produces no output (no `fig` variable, no `result` variable)?
- What if the CSV is replaced or reset while a graph run is in progress?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The app MUST use a LangGraph state graph as the sole orchestration layer for all AI-driven analysis steps.
- **FR-002**: The graph MUST use the Command + goto routing pattern exclusively; conditional edges are prohibited.
- **FR-003**: One graph checkpointer MUST be created per Streamlit browser session and reused across all Streamlit reruns within that session.
- **FR-004**: Plan approval MUST use LangGraph's interrupt/resume mechanism; the graph halts execution at the approval step and only resumes when the user explicitly approves or rejects.
- **FR-005**: LangSmith tracing MUST be activated via environment variables only (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT); no programmatic SDK calls to enable tracing.
- **FR-006**: The graph MUST automatically retry failed code execution up to 3 times, rewriting the code using the LLM on each retry, before any user notification.
- **FR-007**: After 3 consecutive code execution failures, the graph MUST fall back to revising the analysis plan and presenting the revised plan to the user for re-approval.
- **FR-008**: The user MUST be able to cancel the graph at any stage (plan generation, approval pending, code execution) with a single Cancel action; the graph MUST exit cleanly with no residual error state.
- **FR-009**: The tabs panel MUST display three tabs in this order: Templates (default active on load), Plan, Code.
- **FR-010**: The Plan tab MUST activate automatically when the graph interrupts waiting for plan approval.
- **FR-011**: The Plan tab MUST display Approve, Reject, and Cancel controls when approval is pending.
- **FR-012**: The Plan tab MUST include a "Save as Template" button available after a plan has been approved.
- **FR-013**: The Code tab MUST activate automatically after a graph run completes successfully.
- **FR-014**: The Code tab MUST display a report summary paragraph followed by all execution blocks; all blocks MUST be permanently visible with no collapsed or expandable sections.
- **FR-015**: All four quadrants MUST have a fixed height and visible borders.
- **FR-016**: The complete AnalysisState schema MUST be fully defined in this session (Session 1) to support Session 2 additions without any schema changes.
- **FR-017**: The `langgraph` dependency MUST be pinned to version 0.3.18 in requirements.txt.
- **FR-018**: All new code MUST be added to the existing `app.py` file only; no new source files may be created.
- **FR-019**: The existing 4-quadrant layout (chat top-left, analysis tabs top-right, CSV bottom-left, results bottom-right) MUST be preserved.

### Key Entities *(include if feature involves data)*

- **AnalysisState**: The LangGraph TypedDict state schema. Fields defined in Session 1:
  - `user_query` (str): The user's natural language analysis request
  - `df_csv` (str): Serialized CSV text of the active dataset passed into the graph
  - `plan_steps` (list[str]): Ordered list of AI-generated analysis steps
  - `plan_approved` (bool | None): None = awaiting, True = approved, False = rejected
  - `code` (str): Current generated or LLM-rewritten Python code
  - `execution_result` (dict | None): Output from code execution — type, content, error
  - `retry_count` (int): Number of code rewrite attempts made (0–3)
  - `report_summary` (str): Natural language summary written after successful execution
  - `execution_blocks` (list[dict]): Ordered list of step+code+result blocks for the Code tab report
  - `cancelled` (bool): Whether the user cancelled the run
  - `error` (str | None): Unrecoverable error message if graph cannot continue
  - `templates` (list[dict]): Session-scoped saved templates (populated in Session 2; schema slot reserved here)

- **GraphRun**: A single invocation of the LangGraph graph. Has a `thread_id` matching the Streamlit `session_id`, a checkpointer bound to the session, and a lifecycle from plan generation to report completion or cancellation.

- **ExecutionBlock**: One entry in the Code tab report. Contains a step description, the code that was executed, and the result (figure, dataframe, or text).

- **AnalysisTemplate**: A saved plan — name, plan_steps, code, created_at — that can be reloaded and re-executed on new data without re-querying the LLM.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users see the Plan tab with numbered steps and approval controls within 15 seconds of submitting an analysis request on a standard internet connection.
- **SC-002**: After plan approval, the Code tab activates and shows a complete rendered report within 30 seconds for a typical CSV dataset (≤ 10,000 rows).
- **SC-003**: Failed code is retried and rewritten with zero user interaction required — 0 manual steps needed for retry.
- **SC-004**: Users can cancel an in-progress analysis with a single click and the UI returns to idle state within 2 seconds.
- **SC-005**: Every graph run produces a LangSmith trace with at least one span per graph node when LangSmith environment variables are configured.
- **SC-006**: A saved template can be loaded and the plan restored in under 5 seconds with 0 re-entry of the original analysis query.
- **SC-007**: The app runs correctly with `langgraph==0.3.18` alongside existing dependencies with no version conflicts.

## Assumptions

- LangGraph 0.3.18 is compatible with the existing `langchain>=0.3.0`, `langchain-openai>=0.1.0`, and Python 3.11+ environment.
- Session 2 will only add new optional fields to AnalysisState and new graph nodes; no existing fields will be renamed or removed.
- The graph checkpointer uses LangGraph's in-memory MemorySaver; no persistent storage is needed for Session 1.
- "Save as Template" in Session 1 stores templates in Streamlit session state only; cross-session persistence is deferred to Session 2.
- Code execution uses the same isolated `exec` namespace as the existing `execute_code` function (`df`, `pd`, `plt` pre-injected).
- The graph runs synchronously within Streamlit's execution model; async execution patterns are out of scope for Session 1.
- Fixed-height quadrants with visible borders will be implemented via Streamlit's `st.container(height=...)` and CSS border styling.
- The existing `AnalysisResponse` Pydantic model may be reused inside graph nodes for structured LLM output.
- "Reject" during approval presents the revised plan for re-approval (same flow as post-retry fallback).
