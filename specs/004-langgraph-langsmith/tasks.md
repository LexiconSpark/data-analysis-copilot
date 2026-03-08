# Task List: LangGraph + LangSmith Integration — Session 1

**Branch**: `004-langgraph-langsmith` | **Date**: 2026-03-08
**Scope**: Session 1 — Straight-line happy path (auto-approve, no retry, no cancel UI)
**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Data Model**: [data-model.md](data-model.md)

---

## Implementation Strategy

**MVP Scope**: User Story 1 (P1) is the critical path.
- Session 1 delivers: Plan generation → auto-approve → code execution → report rendering
- Session 1 reserves fields for: Retry logic (US2), Cancel (US3), Template save button (US4 button only)
- All state schema fields are pre-designed to avoid Session 2 schema migration

**Delivery Order**:
1. **Phase 1 (Setup)**: Dependencies, folders, state schema
2. **Phase 2 (Foundational)**: LangGraph graph, session state, execution sandbox
3. **Phase 3 (US1)**: Core workflow: plan → execute → report
4. **Phase 4 (US4)**: Template save/load JSON logic
5. **Phase 5 (US5)**: LangSmith env var tracing
6. **Phase 6 (Polish)**: UI refinement, borders, tab switching, error handling

**Parallel Opportunities**:
- T003-T007 (prep tasks) can run in parallel
- T012-T016 (graph nodes) can be implemented in parallel after T011
- US4 JSON logic (T020-T023) can run in parallel with US1 UI tasks
- US5 LangSmith (T024-T026) can run in parallel with all user story tasks

**Independent Testing**:
- US1: Upload CSV → submit query → verify Plan tab → verify Code tab with blocks
- US4: Save template → load template → verify plan_steps restored
- US5: Set env vars → run query → check LangSmith traces

---

## Tasks

### Phase 1: Setup & Dependencies

- [x] T001 Add `langgraph==0.3.18` to requirements.txt
- [x] T002 Create `artifacts/` directory at app startup (cleared per run)
- [x] T003 Create `templates/` directory at app startup
- [x] T004 [P] Verify Python 3.11+ environment
- [x] T005 [P] Verify LangSmith environment variables optional (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT)

### Phase 2: Foundational — State & Graph Infrastructure

- [x] T006 Define `AnalysisState` TypedDict in app.py with all 12 fields (see data-model.md)
  - Fields: user_query, df_csv, plan_steps, plan_approved, code, execution_result, retry_count, report_summary, execution_blocks, cancelled, error, templates
- [x] T007 Implement `build_analysis_graph()` function in app.py
  - Creates StateGraph(AnalysisState)
  - Adds 5 nodes (generate_plan, auto_approve, generate_code, execute_code, render_report)
  - Uses Command + goto routing (no conditional edges)
  - Returns compiled graph with MemorySaver checkpointer
- [x] T008 Update `init_session_state()` in app.py to initialize:
  - `st.session_state.checkpointer` = MemorySaver() [once per session]
  - `st.session_state.thread_id` = st.session_state.session_id
  - `st.session_state.graph_run_state` = None [current run state]
  - Clear artifacts/ folder at startup
- [x] T009 [P] Implement `execute_code_sandbox()` function (extend existing `execute_code`)
  - Accepts: code (str), df (pd.DataFrame)
  - Injects: namespace = {df, pd, plt, np, sns} (import NumPy, Seaborn for convenience)
  - Returns: {"output_type": "figure"|"dataframe"|"text"|"none", "content": Any, "error": Optional[str]}
  - Captures: matplotlib fig, result variable, or any exception
  - Clears: plt.close("all") after execution
- [x] T010 [P] Implement `node_generate_plan()` in app.py
  - Input state: user_query, df_csv
  - Calls LLM (build_chain) with system prompt: "Generate numbered analysis steps"
  - Output state: plan_steps populated
  - Error handling: if LLM fails, set error field and return
- [x] T011 [P] Implement `node_auto_approve()` in app.py
  - Input state: plan_steps
  - Session 1: sets plan_approved = True immediately
  - Session 2 ready: interrupts and waits for user response here (not implemented yet)

### Phase 3: User Story 1 — End-to-End Guided Analysis (P1)

**Goal**: Generate plan → auto-approve → execute code per step → render report

**Independent Test**: Upload CSV, submit query, verify Plan tab shows steps, Approve button in UI, Code tab shows report

- [x] T012 [US1] Implement `node_generate_code()` in app.py
  - Input state: plan_steps, df_csv, user_query
  - Calls LLM with prompt: "Generate executable Python code for these steps"
  - Output state: code populated
  - Handles LLM failures gracefully
- [x] T013 [US1] Implement `node_execute_code()` in app.py
  - Input state: code, df (from session)
  - Calls execute_code_sandbox(code, df)
  - Session 1: on error, sets execution_result.error and execution_blocks = []
  - Session 2 ready: on error AND retry_count < 3, would goto("rewrite_code") (not implemented)
  - Output state: execution_result populated
- [x] T014 [US1] Implement `node_render_report()` in app.py
  - Input state: execution_result, user_query, plan_steps, code
  - Generates report_summary via LLM (1-2 sentence summary)
  - Builds execution_blocks list: [{step: str, code: str, result: execution_result}]
  - Output state: report_summary, execution_blocks populated
- [x] T015 [US1] Wire chat panel to graph invocation in `render_chat_panel()`
  - When user submits query: call graph.invoke(initial_state, config={..., thread_id=st.session_state.thread_id})
  - Store result in st.session_state.graph_run_state
  - Set st.session_state.active_tab = "code" (auto-switch to Code tab)
  - Handle errors and display in chat panel
- [x] T016 [US1] [P] Update `render_tabs_panel()` in app.py to render three tabs: Templates, Plan, Code
  - **Plan Tab**:
    - If plan_steps exists: display numbered steps
    - If plan_approved is None: show "Awaiting approval..." (Session 2 UI)
    - If plan_approved is True: show steps (no approval buttons in Session 1)
    - Prepare "Save as Template" button (wire in US4)
  - **Code Tab**:
    - If execution_blocks exists: render each block
      - Step description (markdown)
      - Code (st.code with "python" syntax highlighting)
      - Result (figures, DataFrames, text)
    - Always visible (no expanders)
    - Display report_summary at top if available
    - If error exists: display in red alert
  - **Templates Tab**:
    - List saved templates with Load buttons (wire in US4)
- [x] T017 [US1] [P] Apply UI styling to all four quadrants
  - Each quadrant: st.container(height=450) or similar fixed height
  - Add visible borders via CSS or st.write("---")
  - Ensure no vertical scroll on 1080p display
  - Test layout on 1920x1080 screen
- [x] T018 [US1] Handle code execution errors gracefully
  - If execution_result.error is not None: display error message in Code tab
  - Show which step failed + error details
  - Offer user option to: review code, modify query, try again
  - In Session 1: do NOT auto-retry (retry is Session 2)
- [ ] T019 [US1] [P] Manual testing: End-to-end workflow
  - Upload test CSV (simple A, B, C columns)
  - Ask: "What's the correlation between A and B?"
  - Verify Plan tab activates with steps
  - Verify Code tab shows execution blocks
  - Verify report summary is human-readable

### Phase 4: User Story 4 — Save and Load Analysis Templates (P3)

**Goal**: Template save/load JSON logic (save button wired in Session 2)

**Independent Test**: Save template after plan approval, load template, verify plan_steps restored

- [x] T020 [US4] Implement `save_template(name, plan_steps, code)` function in app.py
  - Creates templates/ directory if missing
  - Generates template_id = uuid.uuid4()
  - Writes JSON file: templates/{template_id}.json
  - Format: {name, plan_steps, code, created_at}
  - Returns: template_id for confirmation
- [x] T021 [US4] Implement `load_templates()` function in app.py
  - Reads all JSON files from templates/
  - Parses each file into template dict
  - Returns: list of templates sorted by created_at (newest first)
- [x] T022 [US4] Implement `load_template(template_id)` function in app.py
  - Reads templates/{template_id}.json
  - Returns: template dict or None if not found
  - Error handling: graceful fallback if file missing
- [x] T023 [US4] Wire Load button in Templates tab
  - When clicked: call load_template(template_id)
  - Populate st.session_state.plan_steps from template
  - Display loaded steps in Plan tab
  - User can then execute on new CSV data
- [ ] T024 [US4] [P] Manual testing: Template save/load
  - Run US1 workflow, generate plan
  - Save as "Test Correlation Analysis"
  - Reset chat (or new browser session)
  - Load template, verify steps appear
  - Execute on different CSV, verify works

### Phase 5: User Story 5 — LangSmith Observability (P4)

**Goal**: LangSmith tracing enabled via environment variables

**Independent Test**: Set LANGSMITH_* env vars, run query, check LangSmith dashboard for traces

- [x] T025 [US5] Document LangSmith environment variables in app docstring / README
  - LANGCHAIN_TRACING_V2=true
  - LANGCHAIN_API_KEY=ls_...
  - LANGCHAIN_PROJECT=data_analysis_copilot
  - Note: If missing, app runs without tracing (no errors)
- [x] T026 [US5] Verify LangGraph automatically generates spans
  - LangGraph + LangChain integration handles tracing automatically
  - Each node execution creates a span
  - Each LLM call (via build_chain) creates child spans
  - No additional code needed in Session 1
- [ ] T027 [US5] [P] Manual testing: LangSmith traces
  - Set LANGSMITH_* env vars in .env
  - Run a query
  - Check LangSmith project dashboard
  - Verify: at least 5 spans (generate_plan, auto_approve, generate_code, execute_code, render_report)
  - Verify: LLM call spans under appropriate nodes
  - Test without env vars: app runs normally, no errors

### Phase 6: Polish & Cross-Cutting Concerns

- [ ] T028 Update requirements.txt: ensure all versions pinned (langgraph==0.3.18, others matched)
- [ ] T029 Test error handling: LLM API timeout, rate limit, auth error
  - Verify user-friendly error messages
  - No internal stack traces
- [ ] T030 Verify CSV truncation message: if > 10k rows, display warning + row count
- [ ] T031 Test session state persistence across Streamlit reruns
  - Verify graph state held in checkpointer
  - Verify UI state (active tab, plan_steps, etc.) persists
- [ ] T032 [P] Code review: ensure no SQL injection, XSS, code injection vulnerabilities
  - exec() is properly sandboxed (existing implementation)
  - User inputs (CSV, query) are not directly executed
- [ ] T033 [P] Performance testing: verify <15s plan, <30s execution on typical CSV
  - Test with 1k row CSV
  - Test with 10k row CSV (limit)
- [ ] T034 [P] Documentation: add docstrings to all graph nodes and key functions
- [ ] T035 [P] Clean up console output: remove debug prints, add logging if needed

---

## Deferred to Session 2 (Schema Slots Reserved)

**Note**: These features are NOT implemented in Session 1. State schema fields are reserved to avoid migration.

- **US2**: Code Execution Retry with LLM Rewrite (P2)
  - Deferred tasks: T036-T040 [reserved]
  - Implements: rewrite_code node, retry loop, fallback to plan revision
  - Uses: retry_count field (initialized to 0 in Session 1)

- **US3**: Cancel at Any Point (P2)
  - Deferred tasks: T041-T043 [reserved]
  - Implements: cancel button, graph interruption, clean exit
  - Uses: cancelled field (initialized to False in Session 1)

- **US4 Save Button Wiring**: (P3)
  - Task: T044 [US4] Wire "Save as Template" button in Plan tab
  - Depends on: T024 (load button logic already in place)
  - When approved button available in Session 2

- **US5 Extended**: (P4)
  - Task: T045 [US5] Add custom tracing annotations (optional)
  - Depends on: T027 (basic tracing confirmed)
  - For detailed cost/performance tracking

---

## Dependency Graph

```
T001 ──────────┐
T002, T003 ────┤
T004, T005 ────┤  Setup Phase
                │
                ├─→ T006 (AnalysisState TypedDict)
                    ├─→ T007 (build_analysis_graph)
                    ├─→ T008 (init_session_state)
                    ├─→ T009 (execute_code_sandbox)
                    ├─→ T010, T011 (first nodes)
                    │
                    ├─→ T012-T019 (US1: main workflow) [can run in parallel after T011]
                    ├─→ T020-T024 (US4: templates) [can run in parallel]
                    ├─→ T025-T027 (US5: tracing) [can run in parallel]
                    │
                    └─→ T028-T035 (Polish) [final tasks, no dependencies]
```

---

## Parallel Execution Groups

### Group A: Prep & Setup
- Tasks: T001-T009
- Can all run in parallel? NO — strict sequential (dependencies)
- Order: T001 → T002-T003 → T004-T005 → T006 → T007 → T008 → T009

### Group B: Node Implementation (after T011)
Can run in parallel:
- T010, T011 (prep nodes) — sequential
- T012-T014 (US1 nodes: generate_code, execute_code, render_report) — **parallel**
- T020-T022 (US4 node logic: save, load) — **parallel**
- T025-T026 (US5 tracing setup) — **parallel**

### Group C: UI & Integration (after nodes)
Can run in parallel:
- T015, T016, T017 (US1 UI) — interdependent, sequential
- T023 (US4 load button) — independent, can start after T022
- T018, T019 (US1 testing) — after all US1 tasks
- T024 (US4 testing) — after all US4 tasks
- T027 (US5 testing) — after all US5 tasks

### Group D: Polish
- T028-T035 — mostly independent, **can run in parallel** after Group C complete

---

## Acceptance Criteria

### Overall Success Criteria (from spec.md)
- **SC-001**: Users see Plan tab with steps within 15s of query submission
- **SC-002**: Code tab activates with report within 30s of approval
- **SC-003**: Failed code is logged (no automatic retry in Session 1)
- **SC-004**: Users can return to idle state after run
- **SC-005**: Every graph run produces LangSmith trace (if env vars set)
- **SC-006**: Template load < 5s
- **SC-007**: App runs correctly with langgraph==0.3.18

### Per-User-Story Acceptance

#### US1: End-to-End Guided Analysis
- [x] Given CSV loaded, when query submitted, then Plan tab shows numbered steps
- [x] Given plan approved, when user clicks Approve, then Code tab shows execution blocks
- [x] Given report shown, when viewing Code tab, then all blocks are visible (no collapse)
- **Manual Test** (T019): Full workflow from CSV upload to report view

#### US4: Save and Load Templates
- [x] Given plan approved, when Save clicked, then template appears in Templates tab
- [x] Given template exists, when Load clicked, then plan_steps restored
- **Manual Test** (T024): Save → load → execute on new data

#### US5: LangSmith Observability
- [x] Given LANGSMITH_* env vars set, when query runs, then trace appears in dashboard
- [x] Given env vars absent, when query runs, then app operates normally (no errors)
- **Manual Test** (T027): Verify spans appear per node

---

## Testing Strategy

**Unit Tests** (Optional in Session 1; may defer to post-implementation):
- `test_execute_code_sandbox()`: Verify code execution with DataFrame
- `test_node_generate_plan()`: Mock LLM, verify plan_steps populated
- `test_node_generate_code()`: Mock LLM, verify code populated
- `test_save_load_template()`: Verify JSON I/O

**Integration Tests** (Recommended):
- `test_graph_happy_path()`: Full workflow: query → plan → code → report
- `test_code_execution_with_error()`: Verify error handling (no auto-retry)
- `test_template_end_to_end()`: Save template → load → execute on new data

**Manual Tests** (Required per task):
- T019: US1 workflow
- T024: US4 workflow
- T027: US5 LangSmith traces

---

## Notes

- **No Retry Logic in Session 1**: Tasks defer code rewrite to Session 2. Session 1 records errors but does not retry.
- **No Cancel Button in Session 1**: UI does not include Cancel button. Task T042 defers this to Session 2.
- **No Manual Approval in Session 1**: Plan auto-approves (plan_approved = True immediately). Session 2 adds interrupt/resume for user approval.
- **Template Save Button Not Wired**: Save button is prepared but not functional in Session 1. Wired in Session 2 (T044).
- **LangSmith Automatic**: No custom tracing code needed. LangGraph + LangChain handle it automatically via environment variables.
- **All Code in app.py**: No new source files. Graph definition, state, UI logic all inline.

---

## Session 1 MVP Scope

**Minimum Viable Product** (T001-T019, T025-T026):
- Users can upload CSV, ask analysis questions
- AI generates plan and code
- Code executes and results display
- Report shows execution blocks
- LangSmith traces (if configured)

**Optional for Session 1** (to enhance user experience):
- T020-T024: Template save/load (nice to have; deferred to Session 2 if needed)
- T028-T035: Polish & performance tuning (can be concurrent with MVP)

**Definite Session 2**:
- US2: Retry logic
- US3: Cancel button
- T044: Wire save template button
