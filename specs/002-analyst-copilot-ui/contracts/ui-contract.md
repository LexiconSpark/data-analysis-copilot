# UI Contracts: AI Data Analyst Copilot — Core UI Layout

**Branch**: `002-analyst-copilot-ui` | **Date**: 2026-03-05

This document defines the interface contract for each of the four UI quadrants. Each contract specifies what the quadrant reads from session state, what it writes to session state, what the user can do, and what the visible output is.

---

## Quadrant 1: Chat Panel (top-left)

### Purpose
Allows the engineer to send natural language queries and view AI responses in a scrollable conversation thread.

### Reads from session state
| Key | Type | Usage |
|-----|------|-------|
| `st.session_state.messages` | `list[dict]` | Renders each message in order |
| `st.session_state.is_loading` | `bool` | Disables input and shows spinner when `True` |

### Writes to session state
| Key | Type | When |
|-----|------|------|
| `st.session_state.messages` | `list[dict]` | Appends user message on submit; appends assistant reply on AI response |
| `st.session_state.is_loading` | `bool` | Set to `True` on submit; `False` after response |
| `st.session_state.plan_steps` | `list[str]` | Updated with structured output plan |
| `st.session_state.plan_code` | `str` | Updated with structured output code |

### User interactions
| Action | Trigger | Outcome |
|--------|---------|---------|
| Submit message | Type + Enter / Send button | Message appended; AI invoked; Plan and Code tabs updated |
| New Chat | "New Chat" button (header) | Clears `messages`, `history`, `plan_steps`, `plan_code`, `analysis_result` |

### Visible output
- Scrollable conversation thread (fixed height ~400px) with user and assistant messages visually distinguished
- `st.spinner("Thinking…")` while AI is responding
- "New Chat" button in the panel header

### Constraints
- Chat input is disabled (`disabled=True`) while `is_loading=True`
- Empty or whitespace-only submissions are silently rejected
- AI errors display as `st.error()` inline without adding a message to the thread

---

## Quadrant 2: Tabs Panel (top-right)

### Purpose
Presents three tabs — Template, Plan, Code — for reviewing, saving, and loading AI-generated analysis artifacts.

### Reads from session state
| Key | Type | Usage |
|-----|------|-------|
| `st.session_state.plan_steps` | `list[str]` | Displays numbered steps in Plan tab |
| `st.session_state.plan_code` | `str` | Displays code with syntax highlighting in Code tab |
| `st.session_state.templates` | `list[dict]` | Lists saved templates in Template tab |

### Writes to session state
| Key | Type | When |
|-----|------|------|
| `st.session_state.templates` | `list[dict]` | Appended when user saves a template |
| `st.session_state.plan_steps` | `list[str]` | Overwritten when user loads a template |
| `st.session_state.plan_code` | `str` | Overwritten when user loads a template |

### User interactions
| Tab | Action | Trigger | Outcome |
|-----|--------|---------|---------|
| Template | Save template | Enter name + click "Save as Template" | New template appended to `templates` list |
| Template | Load template | Click "Load" next to a template | `plan_steps` and `plan_code` overwritten; Plan and Code tabs update |
| Plan | Execute Plan | Click "Execute Plan" button | `plan_code` executed; result stored in `analysis_result` |
| Code | (read-only) | — | No user action; displays `plan_code` |

### Visible output

**Template tab**:
- Text input for template name
- "Save as Template" button (disabled when `plan_steps` is empty)
- List of saved templates, each with name, created timestamp, and "Load" button

**Plan tab**:
- Numbered list of `plan_steps` (e.g., "1. Filter data for temperature below 60")
- "Execute Plan" button (disabled when `plan_steps` is empty)
- Empty state message: "Send a data analysis question to generate a plan." when `plan_steps == []`

**Code tab**:
- `st.code(plan_code, language="python")` with copy button
- Empty state message: "Code will appear here after your first analysis." when `plan_code == ""`

### Constraints
- "Save as Template" is disabled when `plan_steps` is empty
- "Execute Plan" is disabled when `plan_steps` is empty
- Loading a template does not submit a new chat message

---

## Quadrant 3: CSV Panel (bottom-left)

### Purpose
Displays the active dataset as a scrollable table. Allows the engineer to upload their own CSV or reset to the default dataset.

### Reads from session state
| Key | Type | Usage |
|-----|------|-------|
| `st.session_state.df` | `pd.DataFrame` | Displays as `st.dataframe()` |
| `st.session_state.csv_truncated` | `bool` | Shows truncation warning when `True` |

### Writes to session state
| Key | Type | When |
|-----|------|------|
| `st.session_state.df` | `pd.DataFrame` | Overwritten on successful CSV upload or reset |
| `st.session_state.csv_truncated` | `bool` | Set `True` if uploaded CSV > MAX_CSV_ROWS; `False` on reset |

### User interactions
| Action | Trigger | Outcome |
|--------|---------|---------|
| Upload CSV | File uploader | `df` replaced with uploaded data; truncation applied if > 10k rows |
| Reset | "Reset to default data" button | `df` restored to `DEFAULT_CSV_DATA`; `csv_truncated = False` |

### Visible output
- `st.dataframe(df, use_container_width=True)` — scrollable, full-width table
- Row and column count caption: "N rows × M columns"
- `st.warning("CSV truncated to 10,000 rows.")` when `csv_truncated=True`
- `st.error("Could not parse the uploaded file…")` on parse failure

### Constraints
- Only `.csv` files accepted by the file uploader
- Upload failure does not change `df` (previous data preserved)
- CSV panel survives "New Chat" reset (dataset is independent of conversation state)

---

## Quadrant 4: Results Panel (bottom-right)

### Purpose
Displays the output of executed analysis — charts, tables, or text summaries. Shows a placeholder when no analysis has been executed.

### Reads from session state
| Key | Type | Usage |
|-----|------|-------|
| `st.session_state.analysis_result` | `dict \| None` | Determines what to render |

### Writes to session state
| Key | Type | When |
|-----|------|------|
| `st.session_state.analysis_result` | `dict \| None` | Set to `None` on "New Chat" |

### Visible output

| Condition | Display |
|-----------|---------|
| `analysis_result is None` | `st.info("Analysis results will appear here after you execute a plan.")` |
| `output_type == "figure"` | `st.pyplot(content)` |
| `output_type == "dataframe"` | `st.dataframe(content)` |
| `output_type == "text"` | `st.markdown(content)` |
| `output_type == "none"` (error) | `st.error(error_message)` |

### Constraints
- Results panel is read-only; no user actions in this quadrant
- Results are cleared on "New Chat"
- Results are NOT cleared on CSV upload (allows comparing old result against new data)
