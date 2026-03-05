---

description: "Task list for 002-analyst-copilot-ui implementation"
---

# Tasks: AI Data Analyst Copilot — Core UI Layout

**Input**: Design documents from `/specs/002-analyst-copilot-ui/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ui-contract.md ✓, quickstart.md ✓

**Tests**: Not included (not explicitly requested in spec).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**Source file**: All implementation is in a single file — `app.py` at repository root — plus `requirements.txt`.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (no dependencies on incomplete tasks, non-conflicting file regions)
- **[Story]**: Which user story this task belongs to (US1–US4)

---

## Phase 1: Setup

**Purpose**: Update dependencies to unlock required Streamlit API

- [x] T001 Update `requirements.txt`: bump `streamlit>=1.31.0` (required for `st.container(height=...)`) and add `pydantic>=2.0.0`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core structural changes that ALL user stories depend on — layout skeleton, Pydantic model, session state, and updated LangChain chain.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T002 [P] Define `AnalysisResponse` Pydantic model in `app.py` (near top constants section): fields `chat_reply: str`, `plan_steps: list[str]`, `code: str`
- [x] T003 [P] Extend `init_session_state()` in `app.py` with four new keys: `plan_steps=[]`, `plan_code=""`, `templates=[]`, `analysis_result=None`
- [x] T004 Change `st.set_page_config` in `app.py` from `layout="centered"` to `layout="wide"` and replace the main render body with a 4-quadrant column skeleton: `top_left, top_right = st.columns([1,1])` and `bottom_left, bottom_right = st.columns([1,1])` with `pass` placeholders
- [x] T005 Update `SYSTEM_PROMPT` constant in `app.py` to the data analyst persona (instructs model to always populate `chat_reply`, `plan_steps`, `code`; leave steps/code empty for conversational queries)
- [x] T006 Rebuild `build_chain()` in `app.py`: replace `StrOutputParser()` with `llm.with_structured_output(AnalysisResponse)`; update return type annotation accordingly (depends on T002)

**Checkpoint**: Foundation ready — all 4 quadrant placeholders visible in wide layout; chain returns `AnalysisResponse` objects

---

## Phase 3: User Story 1 — Chat-Driven CSV Analysis (Priority: P1) 🎯 MVP

**Goal**: Engineer sends an analysis question, AI replies in chat, Plan tab and Code tab populate with structured output.

**Independent Test**: Open app, type "plot the relationship between A and B", verify: (1) chat panel shows AI reply, (2) Plan tab shows numbered steps, (3) Code tab shows Python code.

### Implementation for User Story 1

- [x] T007 [US1] Implement `render_chat_panel(col)` in `app.py`: render chat history inside `col.container(height=400)` using `st.chat_message` loop; render chat input below container; disable input when `is_loading=True`
- [x] T008 [US1] Update `get_ai_response(prompt)` in `app.py`: invoke the structured chain, extract `chat_reply` for messages list, write `plan_steps` and `plan_code` to `st.session_state`; return `chat_reply` string (depends on T006)
- [x] T009 [US1] Implement `handle_user_input(prompt)` in `app.py`: append user message, set `is_loading=True`, call `get_ai_response`, append assistant reply, set `is_loading=False`, call `st.rerun()`
- [x] T010 [US1] Add "New Chat" button to the chat panel header in `app.py`: clears `messages`, reinitializes `history`, resets `plan_steps`, `plan_code`, `analysis_result` to defaults; preserves `df` and `templates`
- [x] T011 [US1] Wire `render_chat_panel(top_left)` into the top-left quadrant in the main render block of `app.py` (replaces `pass` placeholder from T004)

**Checkpoint**: User Story 1 fully functional — can run `streamlit run app.py`, send an analysis query, and see structured chat + plan + code output

---

## Phase 4: User Story 2 — CSV Upload and Preview (Priority: P2)

**Goal**: Engineer uploads their own CSV; table in bottom-left updates immediately with new data.

**Independent Test**: Open app, upload a CSV file, verify the table in bottom-left shows uploaded data without sending any chat message.

### Implementation for User Story 2

- [x] T012 [US2] Implement `render_csv_panel(col)` in `app.py`: file uploader (`type=["csv"]`), on success parse with `pd.read_csv`, truncate at `MAX_CSV_ROWS`, set `st.session_state.df`; `st.dataframe` display, row/col caption, truncation warning, "Reset to default data" button
- [x] T013 [US2] Remove `render_sidebar()` function and the `render_sidebar()` call from `app.py`; remove all `with st.sidebar:` blocks (depends on T012 being ready to replace its functionality)
- [x] T014 [US2] Wire `render_csv_panel(bottom_left)` into the bottom-left quadrant in the main render block of `app.py` (replaces `pass` placeholder from T004)

**Checkpoint**: User Stories 1 AND 2 both work — chat + plan/code flow intact; CSV table appears bottom-left; upload and reset work correctly

---

## Phase 5: User Story 3 — Plan / Code / Template Tab Navigation (Priority: P2)

**Goal**: Engineer navigates Template, Plan, and Code tabs; can save and load analysis templates.

**Independent Test**: After a chat analysis query, switch to Plan tab (numbered steps visible), Code tab (syntax-highlighted code visible), Template tab (save a template, verify it appears in list, reload it).

### Implementation for User Story 3

- [x] T015 [US3] Implement `render_tabs_panel(col)` skeleton in `app.py`: create `st.tabs(["Template", "Plan", "Code"])` inside `col`
- [x] T016 [US3] Implement **Plan tab** content in `render_tabs_panel()` in `app.py`: render each step from `plan_steps` as a numbered `st.write`; show empty-state message "Send a data analysis question to generate a plan." when `plan_steps == []`; add "Execute Plan" button (wired up in US4, disabled for now)
- [x] T017 [US3] Implement **Code tab** content in `render_tabs_panel()` in `app.py`: `st.code(plan_code, language="python")`; show empty-state message "Code will appear here after your first analysis." when `plan_code == ""`
- [x] T018 [US3] Implement **Template tab** content in `render_tabs_panel()` in `app.py`: text input for template name; "Save as Template" button (disabled when `plan_steps == []`); iterate `templates` list and render each with name, timestamp, and "Load" button
- [x] T019 [US3] Implement **save template** logic in `app.py`: on "Save as Template" click, append `{"name": name, "plan_steps": plan_steps, "code": plan_code, "created_at": iso_timestamp}` to `st.session_state.templates`; clear name input
- [x] T020 [US3] Implement **load template** logic in `app.py`: on "Load" button click for a template, overwrite `st.session_state.plan_steps` and `st.session_state.plan_code` with template values; call `st.rerun()`
- [x] T021 [US3] Wire `render_tabs_panel(top_right)` into the top-right quadrant in the main render block of `app.py` (replaces `pass` placeholder from T004)

**Checkpoint**: User Stories 1, 2, and 3 all work — full top row functional; templates persist through session; Code tab shows syntax-highlighted code

---

## Phase 6: User Story 4 — Analysis Results Display (Priority: P3)

**Goal**: Clicking "Execute Plan" runs the AI-generated code and displays output (chart, table, or text) in the bottom-right panel.

**Independent Test**: After receiving an analysis response, click "Execute Plan" in the Plan tab; verify a chart or data appears in the bottom-right panel. With no analysis run, verify the placeholder message is shown.

### Implementation for User Story 4

- [x] T022 [US4] Implement `execute_code(code, df)` function in `app.py`: run `exec(code, {"df": df, "pd": pd, "plt": plt})` in an isolated namespace; detect `fig` (matplotlib Figure), `result` (DataFrame), or text output; return `{"output_type": ..., "content": ..., "error": None}`; on exception return `{"output_type": "none", "error": str(e)}`
- [x] T023 [US4] Implement `render_results_panel(col)` in `app.py`: if `analysis_result is None` show `col.info("Analysis results will appear here after you execute a plan.")`; if `output_type == "figure"` call `col.pyplot(content)`; if `output_type == "dataframe"` call `col.dataframe(content)`; if `output_type == "text"` call `col.markdown(content)`; if `output_type == "none"` call `col.error(error)`
- [x] T024 [US4] Wire "Execute Plan" button (T016) to call `execute_code(st.session_state.plan_code, st.session_state.df)`, store result in `st.session_state.analysis_result`, and call `st.rerun()` in `app.py`
- [x] T025 [US4] Wire `render_results_panel(bottom_right)` into the bottom-right quadrant in the main render block of `app.py` (replaces `pass` placeholder from T004)

**Checkpoint**: All 4 user stories functional end-to-end — complete analysis loop works: query → plan → code → execute → results

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup, title update, and validation.

- [x] T026 [P] Update `page_title` in `st.set_page_config` to `"Data Analyst Copilot"` in `app.py`
- [x] T027 [P] Remove any remaining references to `st.sidebar` or `render_sidebar` in `app.py` (cleanup from T013 if any were missed)
- [x] T028 Run end-to-end quickstart.md validation: steps 1–5 (upload, chat, execute, save template, reset) — fix any discovered issues in `app.py`
- [x] T029 Manual layout check: open app at 100% browser zoom on a 1920×1080 display (or equivalent viewport), confirm all 4 quadrants visible without outer-page scrolling

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — **BLOCKS all user stories**
- **US1 (Phase 3)**: Depends on Phase 2 — start after Foundational
- **US2 (Phase 4)**: Depends on Phase 2 — independent of US1 (different quadrant/functions)
- **US3 (Phase 5)**: Depends on Phase 2 — independent of US1/US2 (different quadrant/functions)
- **US4 (Phase 6)**: Depends on US3 (T016 "Execute Plan" button stub) — needs T016 before T024
- **Polish (Phase 7)**: Depends on all user stories complete

### User Story Dependencies

- **US1 (P1)**: Can start after Phase 2 — no dependencies on US2, US3, US4
- **US2 (P2)**: Can start after Phase 2 — no dependencies on US1, US3, US4
- **US3 (P2)**: Can start after Phase 2 — no dependencies on US1, US2; US4 depends on US3's Execute Plan stub
- **US4 (P3)**: Depends on US3 (T016 must exist) — otherwise independent

### Within Each User Story

- T002 and T003 are parallelizable (different functions in app.py)
- Within US3: T015 skeleton → T016, T017, T018 can proceed in parallel → T019, T020 → T021
- US1 and US2 can be implemented in parallel by different agents/developers after Phase 2

### Parallel Opportunities

- T002 and T003 (both Phase 2, different app.py sections)
- T007 and T012 (US1 chat render vs US2 CSV panel — different functions, independent)
- T016 and T017 and T018 (US3 — three different tab sections in same function, but sequential writes to same function body; treat as sequential)
- T026 and T027 (Phase 7 — different constants/cleanup, non-conflicting)

---

## Parallel Example: Phase 2 Foundation

```bash
# T002 and T003 can be written simultaneously (different function regions of app.py):
Task: "Define AnalysisResponse Pydantic model near top constants in app.py"
Task: "Extend init_session_state() with plan_steps, plan_code, templates, analysis_result"

# Then sequentially:
Task: "T004 - layout skeleton (depends on knowing layout is wide)"
Task: "T005 - system prompt update (independent)"
Task: "T006 - rebuild chain with structured output (depends on T002)"
```

## Parallel Example: US1 + US2 (after Phase 2)

```bash
# Agent A works on chat panel (top-left functions):
Task: "T007 - render_chat_panel()"
Task: "T008 - get_ai_response() structured output update"
Task: "T009 - handle_user_input()"
Task: "T010 - New Chat button"
Task: "T011 - wire into top_left"

# Agent B works on CSV panel (bottom-left functions) simultaneously:
Task: "T012 - render_csv_panel()"
Task: "T013 - remove render_sidebar()"
Task: "T014 - wire into bottom_left"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: Foundational (T002–T006)
3. Complete Phase 3: User Story 1 (T007–T011)
4. **STOP and VALIDATE**: `streamlit run app.py` → send "plot A vs B" → confirm Plan + Code tabs populate
5. Demo with 4-quadrant layout even if US2/US3/US4 panels are empty placeholders

### Incremental Delivery

1. T001 → T002–T006 → **Foundation checkpoint**
2. T007–T011 → **US1 checkpoint**: chat → plan → code loop works
3. T012–T014 → **US2 checkpoint**: CSV upload in bottom-left
4. T015–T021 → **US3 checkpoint**: full tab navigation + templates
5. T022–T025 → **US4 checkpoint**: Execute Plan → results panel
6. T026–T029 → **Polish & validation**

### Total Task Count: 29 tasks

| Phase | Tasks | Count |
|-------|-------|-------|
| Setup | T001 | 1 |
| Foundational | T002–T006 | 5 |
| US1 (P1) | T007–T011 | 5 |
| US2 (P2) | T012–T014 | 3 |
| US3 (P2) | T015–T021 | 7 |
| US4 (P3) | T022–T025 | 4 |
| Polish | T026–T029 | 4 |
| **Total** | | **29** |

---

## Notes

- All tasks modify `app.py` (single-file app) or `requirements.txt`
- [P] tasks = non-conflicting regions of app.py (different functions/constants)
- Each user story adds one complete quadrant of the UI
- US4 is marked P3 and explicitly a placeholder in the spec — `render_results_panel` with `st.info()` is sufficient for the bottom-right, Execute Plan integration is bonus
- Verify layout at 100% zoom before marking T029 complete
- `st.container(height=400)` requires `streamlit>=1.31.0` — T001 must complete before running app
