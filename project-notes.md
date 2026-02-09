# Test Prompts
Analyze the data from column A and column B.

Calculate the correlation between the two columns
======
Goal: get MVP by Feb 25th
*5-2-26*
continued to instl dependencies
had to install pandas separately bc of error
---
works
but to generate report, asking for duckduckgo import
tried again, no
asked for langchainhub import

*6-2-26*
current setup 2.5 agents, with 2 being langchain
next steps:
1. understand code base
2. get langsmith debugging tool to work (in code, but doesn't work)
3. get MVP: get langgraph agent to replace second agent.
---
"""
===FIXES===
1. 120: replace with simple_data_analysis
2. 184: duplicate elif block
"""
*6-2-26*
1. studied codebase
2. researched langchain vs langraph, vs langsmith
3. made a langsmith basic tracing agent

# goals
1. update langchain to 0.3
2. integrate langsmith
3. integrate langraph

# on code base
- **Flow:** Chat → (optional) report plan via tool → Execute Plan → agent runs Python + search → report code generated from agent output → code shown in Code tab and run in Report area.
- **Diagram:** Section 2 gives a structural and data-flow diagram of the file.
- **Important functions:** Environment init, stream/data/table helpers, `generate_chatbot_response`, `execute_plan`, `format_intermediate_steps`, `generate_code_for_display_report` — with parameters and return values as in the table and “parameters in depth” above.
- **Class:** `SessionStateAutoClass` uses `__setattr__` and `__getattr__` to bind an attribute to `st.session_state` and trigger `st.rerun()` on change so the “Code” tab updates when the agent’s intermediate steps are set.

Codebase diagram

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
