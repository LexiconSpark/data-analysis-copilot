# `streamlit_app.py` — Codebase Guide

This document explains the logical flow, structure, main functions, and classes in `streamlit_app.py`.

---

## 1. Logical flow (high level)

1. **Startup**
   - Load env (`.env`), set LangSmith/OpenAI config.
   - Initialize `langsmith_client`, `openai_client`, `OPENAI_API_KEY`.
   - Set default `st.session_state["openai_model"]` (e.g. `"gpt-4o"`).
   - Initialize session state: `plan`, `code`, `agent_thoughtflow`, `formatted_output`, and the auto-refresh helper.

2. **UI layout (every run)**
   - Wide layout, then a 2×2 grid:
     - **Top-left:** Chatbot (history + chat input).
     - **Top-right:** Tabs “Plan” and “Code” (plan text, “Execute Plan” button, intermediate steps, ACE editor for report code).
     - **Bottom-left:** “User Data Set” — editable dataframe (default or CSV).
     - **Bottom-right:** “AI Generated Report” — output of executing the report code.

3. **User sends a chat message**
   - Message is appended to `messages` and shown.
   - `generate_chatbot_response(openai_client, st.session_state, user_input)` is called.
   - The chatbot uses the OpenAI API with a **function/tool**: `trigger_report_generation(user_message)`.
   - **If the model calls `trigger_report_generation`:**
     - App asks the model for a **plan** (steps + suggested tools: `python_repl_tool`, `web_search`).
     - Plan is streamed and stored in `st.session_state.plan`.
     - User is told to click “Execute Plan” or ask for plan changes.
   - **If the model does not call the tool:**
     - The assistant’s text reply is streamed (e.g. explaining what the copilot can do).

4. **User clicks “Execute Plan”**
   - `execute_plan(st.session_state.plan)` runs:
     - Builds tools: DuckDuckGo search + Python REPL.
     - Creates a LangChain agent (OpenAI functions agent) and executor.
     - Invokes the agent with the plan + current dataframe (CSV).
     - Agent runs Python (and optionally web search) and produces an output.
     - `generate_code_for_display_report(agent_response)` uses OpenAI to turn the agent’s thoughtflow into **Streamlit code** (e.g. `st.write` / `st.image`).
     - That code is stored in `st.session_state.code`.
     - Intermediate steps are formatted and assigned to `session_state_auto.formatted_output` (triggers rerun).
     - Chat history is updated and `st.rerun()` is called.

5. **Report display**
   - “Code” tab shows the generated (or edited) report code in `st_ace`.
   - The bottom-right section runs that code with `exec(reporting_code)`, so the “AI Generated Report” area shows the result of that code (writes and images).

6. **Data changes**
   - The bottom-left data editor updates `st.session_state.df`.
   - `on_change=handle_table_change` can append a bot message when the table is edited.

So: **Chat → optional plan (via tool) → Execute Plan → agent runs tools → code for report generated → that code is shown and executed in the Report quadrant.**

---

## 2. Codebase diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           streamlit_app.py                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  IMPORTS & CONFIG                                                                │
│  load_dotenv, ROW_HEIGHT, TEXTBOX_HEIGHT                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  initialize_environment()                                                        │
│       → LangSmithClient, OpenAI client, OPENAI_API_KEY                           │
│  langsmith_client, openai_client, OPENAI_API_KEY = ...                           │
│  session_state["openai_model"] default                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│  HELPERS                                                                         │
│  get_stream(sentence)          → yield words + " ", sleep 0.05                   │
│  get_data(placeholder)         → st.session_state.df.to_csv()                    │
│  get_dataframe()               → sample pd.DataFrame (A, B, C columns)           │
│  handle_table_change()         → if table_changed, append bot message            │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CHATBOT CORE                                                                    │
│  generate_chatbot_response(openai_client, session_state, user_input)              │
│       │                                                                          │
│       ├─ OpenAI chat.completions.create(..., tools=[trigger_report_generation])  │
│       ├─ if tool_calls and name == "trigger_report_generation":                  │
│       │      get_data() → plan via OpenAI (stream) → session_state.plan         │
│       ├─ elif tool_calls and name == "simple_data_analysis":  (unused in API)   │
│       │      pandas agent + summarize (duplicate block)                         │
│       └─ else: stream assistant text reply                                       │
│       return response                                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│  PLAN EXECUTION                                                                  │
│  execute_plan(plan)                                                              │
│       │                                                                          │
│       ├─ tools: DuckDuckGo (web_search), PythonREPLTool (python_repl_tool)        │
│       ├─ prompt from hub (openai-functions-template) + instructions             │
│       ├─ agent = create_openai_functions_agent(ChatOpenAI, tools, prompt)         │
│       ├─ agent_executor = AgentExecutor(agent, tools, ...)                       │
│       ├─ response = agent_executor.invoke({ "input": plan + df.to_csv() })       │
│       ├─ generate_code_for_display_report(response) → OpenAI → code string      │
│       ├─ session_state.code = code, messages append, formatted_output, st.rerun │
│       └─ format_intermediate_steps(response) → markdown string                   │
│                                                                                  │
│  format_intermediate_steps(response)  → "### Intermediate Steps" + tool logs     │
│  generate_code_for_display_report(execution_agent_response)                      │
│       → session_state.agent_thoughtflow = output + step logs                     │
│       → openai_client.chat.completions.create(...) → code (st.write/st.image)   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  SESSION STATE INIT                                                              │
│  plan, code, agent_thoughtflow, formatted_output                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  SessionStateAutoClass                                                           │
│  __setattr__(name, value)  → if value changed: session_state[name]=value; rerun  │
│  __getattr__(name)         → session_state.get(name, None)                       │
│  session_state_auto = SessionStateAutoClass()                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  UI (runs every script run)                                                      │
│  set_page_config(layout="wide")                                                  │
│  ┌──────────────────────┬──────────────────────┐                                 │
│  │ col1row1             │ col2row1             │                                 │
│  │ Chatbot              │ Tabs: [Plan][Code]   │                                 │
│  │ - chat history       │ Plan: plan + button  │                                 │
│  │ - chat_input         │ Code: formatted_output│                                │
│  │ → generate_chatbot_  │   + st_ace(code)     │                                 │
│  │   response()         │   "Execute Plan"     │                                 │
│  ├──────────────────────┼──────────────────────┤                                 │
│  │ col1row2             │ col2row2             │                                 │
│  │ User Data Set        │ AI Generated Report  │                                 │
│  │ st.data_editor(df)   │ exec(reporting_code) │                                 │
│  │ on_change=handle_    │                      │                                 │
│  │   table_change       │                      │                                 │
│  └──────────────────────┴──────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Data flow (simplified):**

```
user message
    → generate_chatbot_response
        → OpenAI (tool: trigger_report_generation?)
            → YES: get plan (stream) → session_state.plan
            → NO: stream text reply
user clicks "Execute Plan"
    → execute_plan(plan)
        → Agent (REPL + web_search) with plan + df
        → generate_code_for_display_report(agent_response)
        → session_state.code, formatted_output, rerun
UI: Code tab shows st_ace(session_state.code)
UI: Report area runs exec(reporting_code)
```

---

## 3. Most important functions and parameters

| Function | Purpose | Parameters | Returns |
|----------|---------|------------|--------|
| **`initialize_environment()`** | Loads `.env`, sets LangSmith env vars, creates LangSmith and OpenAI clients. | None | `(LangSmithClient(), OpenAI(api_key=...), OPENAI_API_KEY)` |
| **`get_stream(sentence)`** | Generator that yields words from `sentence` with a short sleep between them (for streaming effect in UI). | `sentence: str` | Generator yielding `str` (words + space). |
| **`get_data(placeholder)`** | Returns the current dataframe as CSV string. `placeholder` is unused. | `placeholder` (unused) | `str` (CSV of `st.session_state.df`). |
| **`get_dataframe()`** | Builds a small sample DataFrame (columns A, B, C) used as default user data. | None | `pd.DataFrame` |
| **`handle_table_change()`** | Callback when the data editor says the table changed; appends a bot message to chat history. | None | None |
| **`generate_chatbot_response(openai_client, session_state, user_input)`** | Main chatbot logic: calls OpenAI with a “report” tool; if tool is used, generates and stores a plan and streams it; otherwise streams a normal reply. | `openai_client`: OpenAI client; `session_state`: Streamlit session state; `user_input`: str | `response`: str (content streamed to the user). |
| **`execute_plan(plan)`** | Runs the plan with a LangChain agent (Python REPL + web search), then turns the agent output into report code and updates session state and UI. | `plan`: str (the plan text from the chatbot) | None (side effects: session state, rerun). |
| **`format_intermediate_steps(response)`** | Converts the agent’s intermediate steps into a markdown string for the “Code” tab. | `response`: dict from `AgentExecutor.invoke` (has `intermediate_steps`, `output`) | `str` (markdown). |
| **`generate_code_for_display_report(execution_agent_response)`** | Stores agent output + step logs in `agent_thoughtflow`, then asks OpenAI to generate Streamlit code (e.g. `st.write`/`st.image`) to display that result. | `execution_agent_response`: dict with `output` and `intermediate_steps` | OpenAI response object; `.choices[0].message.content` is the code string. |

**Important parameters in depth**

- **`generate_chatbot_response(openai_client, session_state, user_input)`**
  - `session_state` must have: `openai_model`, `messages`, `df`, and will get `plan`, `current_user_input` when a report is requested.
  - The model sees a system message (data analysis copilot, report creation, tools) and `messages` as conversation history; it can call `trigger_report_generation(user_message)`.

- **`execute_plan(plan)`**
  - `plan` is the text plan stored in `st.session_state.plan` (with optional `[Tool: ...]` tags). The function also reads `st.session_state.df` and passes it as CSV to the agent.

- **`generate_code_for_display_report(execution_agent_response)`**
  - Expects `execution_agent_response["output"]` and `execution_agent_response["intermediate_steps"]`; uses `st.session_state.current_user_input` and writes to `st.session_state.agent_thoughtflow`.

---

## 4. Classes and dunder methods

There is one class used for state and UI refresh.

### `SessionStateAutoClass`

**Role:** Mirrors attribute access and assignment to `st.session_state` and triggers a Streamlit rerun when a value changes, so the UI can update without the user manually refreshing.

**Usage:** A single instance is created: `session_state_auto = SessionStateAutoClass()`. The only attribute used in the app is `formatted_output` (intermediate steps from the agent). Setting `session_state_auto.formatted_output = ...` both stores the value in session state and reruns the script so the “Code” tab shows the new steps.

**Dunder methods**

| Method | Purpose |
|--------|--------|
| **`__setattr__(self, name, value)`** | Intercepts attribute assignment (e.g. `session_state_auto.formatted_output = x`). If the new value is different from the current one, it writes `st.session_state[name] = value` and calls `st.rerun()` so the app reruns and the new value is visible. |
| **`__getattr__(self, name)`** | Called when an attribute is not found on the instance. Returns `st.session_state.get(name, None)`, so reading `session_state_auto.formatted_output` effectively reads from session state. |

**Why these matter**

- Without `__setattr__`, assigning to `session_state_auto.formatted_output` would only set an instance attribute and would not trigger a rerun; the “Code” tab would not show new intermediate steps until some other action caused a rerun.
- Without `__getattr__`, the UI would not see the latest `formatted_output` from session state when rendering the “Code” tab.

**Note:** The comment in the code warns that this pattern can trigger `st.rerun()` in the middle of other logic, so it should be used carefully. In this app it is used at the end of `execute_plan()` to set the formatted steps and then rerun so the new plan/code and steps are visible.

---

## 5. Summary

- **Flow:** Chat → (optional) report plan via tool → Execute Plan → agent runs Python + search → report code generated from agent output → code shown in Code tab and run in Report area.
- **Diagram:** Section 2 gives a structural and data-flow diagram of the file.
- **Important functions:** Environment init, stream/data/table helpers, `generate_chatbot_response`, `execute_plan`, `format_intermediate_steps`, `generate_code_for_display_report` — with parameters and return values as in the table and “parameters in depth” above.
- **Class:** `SessionStateAutoClass` uses `__setattr__` and `__getattr__` to bind an attribute to `st.session_state` and trigger `st.rerun()` on change so the “Code” tab updates when the agent’s intermediate steps are set.
