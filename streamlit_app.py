import os
import sys
import ast
import subprocess
import tempfile
import shutil
import streamlit as st
import pandas as pd
import time

from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langsmith.wrappers import wrap_openai
from langsmith import traceable, Client as LangSmithClient
from streamlit_ace import st_ace
from openai import OpenAI

load_dotenv()
ROW_HIGHT = 600
TEXTBOX_HIGHT = 90


def initialize_environment():
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "data_analysis_copilot"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    return (
        LangSmithClient(),
        OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        os.getenv("OPENAI_API_KEY"),
    )


langsmith_client, openai_client, OPENAI_API_KEY = initialize_environment()
openai_client = wrap_openai(openai_client)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"


# ─────────────────────────────────────────────
# Shared utility helpers
# ─────────────────────────────────────────────

def get_stream(sentence):
    for word in sentence.split():
        yield word + " "
        time.sleep(0.05)


def get_data(placeholder):
    new_data = st.session_state.df.to_csv()
    return new_data


def get_dataframe():
    df = pd.DataFrame(
        {
            "A": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "B": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
            "C": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        }
    )
    return df


def handle_table_change():
    if "table_changed" in st.session_state and st.session_state["table_changed"]:
        st.session_state["chat_history"].append(
            {"role": "bot", "content": "A change was made to the table."}
        )


# ─────────────────────────────────────────────
# LangGraph State
# ─────────────────────────────────────────────

class CodePlanState(TypedDict):
    messages: list      # conversation messages, first entry is the user task
    plan: list          # remaining plan steps (strings), consumed step by step
    code_files: dict    # accumulated per-step code files {filename: code}
    test_results: dict  # result from last subprocess run
    errors: list        # errors from last failed run
    iterations: int     # rewrite attempts for the current step
    step_count: int     # total steps successfully completed


MAX_CODE_RETRIES = 3
MAX_PLAN_RETRIES = 2


# ─────────────────────────────────────────────
# Subprocess-based code runner
# ─────────────────────────────────────────────

def run_tests(code_files: dict) -> dict:
    """Execute the latest generated code file in a temp directory."""
    if not code_files:
        return {"passed": False, "errors": ["No code files to run"]}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            st.session_state.df.to_csv(os.path.join(tmpdir, "data.csv"), index=False)

            for filename, code in code_files.items():
                full_code = (
                    "import pandas as pd\n"
                    "import matplotlib\nmatplotlib.use('Agg')\n"
                    "import matplotlib.pyplot as plt\n"
                    "import numpy as np\n"
                    "df = pd.read_csv('data.csv')\n\n"
                    + code
                )
                with open(os.path.join(tmpdir, filename), "w") as f:
                    f.write(full_code)

            latest_file = list(code_files.keys())[-1]
            result = subprocess.run(
                [sys.executable, latest_file],
                cwd=tmpdir, capture_output=True, text=True, timeout=30,
            )

            # Copy *.png to workspace if it was generated
            for fname in os.listdir(tmpdir):
                if fname.endswith(".png"):
                    shutil.copy(os.path.join(tmpdir, fname), os.path.join(os.getcwd(), fname))
                    passed = result.returncode == 0
            if not passed:
                print(f"[LG] STDERR: {result.stderr}")
                print(f"[LG] STDOUT: {result.stdout}")
            return {
                "passed": passed,
                "errors": [result.stderr] if not passed else [],
                "stdout": result.stdout,
            }
    except subprocess.TimeoutExpired:
        return {"passed": False, "errors": ["Code execution timed out"]}
    except Exception as e:
        return {"passed": False, "errors": [str(e)]}


# ─────────────────────────────────────────────
# LangGraph nodes
# ─────────────────────────────────────────────

def lg_plan_node(state: CodePlanState) -> Command[Literal["write_code"]]:
    print(f"\n[LG] Entering lg_plan_node (step_count: {state.get('step_count', 0)})")
    task = state["messages"][-1]["content"]
    data_csv = st.session_state.df.to_csv()

    prompt = (
        task
        + """ \n make a simple plan that is simple to understand without technical terms to create code in python 
            to analyze this data(do not include the code), only include the plan as list of steps in the output. 
            At the same time, you are also given a list of tools, they are python_repl_tool for writing code, and another one is called web_search for searching on the web for knowledge you do not know. 
            Please assign the right tool to do each step, knowing the tools that got activated later will know the output of the previous tools. 
            the plan can be hierarchical, meaning that when multiple related and consecutive step can be grouped in one big step and be achieve by the same tool,
            you can group under a parent step and have them as sub-steps and only mention the tool recommended for the partent step. try to limit your parent step to be less than 5 steps. 
            At the each parent step of the plan, please indicate the tool you recommend in a [] such as [Tool: web_search], and put it at the begining of that step. Do not indicate the tool recommendation for sub-steps
            In your output please only give one coherent plan with no analysis
                """
        + "\n this is the data \n"
        + data_csv
    )

    # If replanning due to repeated failures, add context
    if state.get("errors"):
        prompt += (
            f"\n\nNote: A previous plan was attempted but kept failing with:\n"
            f"{chr(10).join(state['errors'])}\n"
            "Please revise the plan to avoid these issues."
        )

    response = openai_client.chat.completions.create(
        model=st.session_state["openai_model"],
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()
    steps = [line.strip("- •").strip() for line in raw.splitlines() if line.strip()]

    return Command(update={"plan": steps, "iterations": 0}, goto="write_code")


def lg_write_code(state: CodePlanState) -> Command[Literal["check_code"]]:
    print(f"\n[LG] Entering lg_write_code (step_count: {state.get('step_count', 0)})")
    all_steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["plan"]))
    print(f"[LG] Writing code for all {len(state['plan'])} steps at once")
    data_csv = st.session_state.df.to_csv()

    response = openai_client.chat.completions.create(
        model=st.session_state["openai_model"],
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                f"Write a single complete Python script to execute ALL of these steps:\n{all_steps}\n\n"
                "Assume these are already available: `df` (pandas DataFrame), pd, plt, np.\n"
                f"Full dataset:\n{data_csv}\n\n"
                "Instructions:\n"
                "  - Write one complete Python script covering all steps\n"
                "  - Use print() to output results with descriptive labels\n"
                "  - Save plots as 'plot.png' with plt.savefig('plot.png')\n"
                "  - Do not redefine df or re-import libraries\n"
                "  - Return ONLY plain Python code without markdown or code fences"
            ),
        }],
    )

    code = response.choices[0].message.content.strip()
    if code.startswith("```"):
        code = "\n".join(code.splitlines()[1:])
    if code.endswith("```"):
        code = "\n".join(code.splitlines()[:-1])

    return Command(update={"code_files": {"analysis.py": code}}, goto="check_code")


def lg_check_code(state: CodePlanState) -> Command[Literal["rewrite_code", END]]:
    print(f"\n[LG] Entering lg_check_code (step_count: {state.get('step_count', 0)})")

    code = state["code_files"]["analysis.py"]

    # ── Static checks (free — no LLM call) ──

    # Syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        return Command(
            update={"errors": [f"SyntaxError: {e}"], "test_results": {"passed": False}},
            goto="rewrite_code",
        )

    # Forbidden library check
    forbidden = ["requests", "flask", "django", "sklearn", "tensorflow", "torch", "scipy"]
    for lib in forbidden:
        if f"import {lib}" in code or f"from {lib}" in code:
            return Command(
                update={"errors": [f"Forbidden library: {lib}"], "test_results": {"passed": False}},
                goto="rewrite_code",
            )

    # ── LLM logic review — only on first attempt (iterations == 0) ──
    if state.get("iterations", 0) == 0:
        review_response = openai_client.chat.completions.create(
            model=st.session_state["openai_model"],
            temperature=0,
            messages=[{
                "role": "user",
                "content": (
                    f"Review this Python code for correctness.\n"
                    f"It is meant to: {state['messages'][-1]['content']}\n\n"
                    f"CODE:\n{code}\n\n"
                    "If correct and complete, respond with exactly: OK\n"
                    "If there is a logical bug or missing step, respond with ONE sentence describing the issue only. Do not rewrite the code."
                ),
            }],
        )
        review = review_response.choices[0].message.content.strip()
        if review.upper() != "OK":
            return Command(
                update={"errors": [f"Logic issue: {review}"], "test_results": {"passed": False}},
                goto="rewrite_code",
            )

    # ── Subprocess execution check ──
    print(f"[LG] Running tests on analysis.py...")
    test_results = run_tests(state["code_files"])
    print(f"[LG] Test result: {'PASSED' if test_results['passed'] else 'FAILED'}")

    if test_results["passed"]:
        return Command(update={"step_count": 1}, goto=END)
    else:
        return Command(
            update={"errors": test_results["errors"], "test_results": test_results},
            goto="rewrite_code",
        )

def lg_rewrite_code(state: CodePlanState) -> Command[Literal["check_code", "update_plan"]]:
    print(f"\n[LG] Entering lg_rewrite_code (step_count: {state.get('step_count', 0)}, iterations: {state['iterations']})")

    if state["iterations"] >= MAX_CODE_RETRIES:
        print(f"[LG] Max iterations ({MAX_CODE_RETRIES}) reached, moving to update_plan")
        return Command(update={"step_count": state.get("step_count", 0)}, goto="update_plan")

    print(f"[LG] Attempt {state['iterations'] + 1} to fix code...")

    broken_code = state["code_files"]["analysis.py"]
    errors = "\n".join(state["errors"])

    response = openai_client.chat.completions.create(
        model=st.session_state["openai_model"],
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                f"Fix this Python code that failed:\n\n"
                f"Code:\n{broken_code}\n\n"
                f"Errors:\n{errors}\n\n"
                "The DataFrame `df` is already loaded. Return ONLY fixed Python code without markdown."
            ),
        }],
    )

    fixed_code = response.choices[0].message.content.strip()
    if fixed_code.startswith("```"):
        fixed_code = "\n".join(fixed_code.splitlines()[1:])
    if fixed_code.endswith("```"):
        fixed_code = "\n".join(fixed_code.splitlines()[:-1])

    updated_files = {"analysis.py": fixed_code}
    return Command(
        update={"code_files": updated_files, "iterations": state["iterations"] + 1},
        goto="check_code",
    )


def lg_update_plan(state: CodePlanState) -> Command[Literal["write_code", END]]:
    print(f"\n[LG] Entering lg_update_plan (step_count: {state.get('step_count', 0)})")

    errors = "\n".join(state["errors"])
    remaining = "\n".join(state["plan"])

    response = openai_client.chat.completions.create(
        model=st.session_state["openai_model"],
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                f"The plan keeps failing with these errors:\n{errors}\n\n"
                f"Remaining steps:\n{remaining}\n\n"
                "Revise the steps to avoid these errors. "
                "Return ONLY a Python list of strings. No explanation, no markdown."
            ),
        }],
    )

    raw = response.choices[0].message.content.strip()
    try:
        new_steps = ast.literal_eval(raw)
        if not isinstance(new_steps, list):
            new_steps = [raw]
    except Exception:
        new_steps = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]

    if not new_steps:
        return Command(goto=END)
    return Command(update={"plan": new_steps, "iterations": 0}, goto="write_code")


# ─────────────────────────────────────────────
# Build and compile the LangGraph
# ─────────────────────────────────────────────

_graph = StateGraph(CodePlanState)
_graph.add_node("planner", lg_plan_node)
_graph.add_node("write_code", lg_write_code)
_graph.add_node("check_code", lg_check_code)
_graph.add_node("rewrite_code", lg_rewrite_code)
_graph.add_node("update_plan", lg_update_plan)
_graph.set_entry_point("planner")
langgraph_app = _graph.compile()


def _print_langgraph_structure(graph):
    try:
        graph_view = graph.get_graph()
        if hasattr(graph_view, "draw_ascii"):
            print("\n" + "=" * 50)
            print("LangGraph structure:")
            print("=" * 50 + "\n" + graph_view.draw_ascii())
            print("=" * 50 + "\n")
            return
        print(f"LangGraph nodes: {list(getattr(graph_view, 'nodes', []))}")
        print(f"LangGraph edges: {list(getattr(graph_view, 'edges', []))}")
    except Exception as exc:
        print(f"LangGraph structure unavailable: {exc}")


if "langgraph_initialized" not in st.session_state:
    _print_langgraph_structure(langgraph_app)
    st.session_state.langgraph_initialized = True


# ─────────────────────────────────────────────
# generate_code_for_display_report
# ─────────────────────────────────────────────

def generate_code_for_display_report(execution_agent_response):
    st.session_state.agent_thoughtflow = (
        "Here is the final output: "
        + str(execution_agent_response["output"])
        + "\nHere is the log of the different step's output, you will be able to find the useful information within there: \n"
        + "".join(
            str(step.log) for step in execution_agent_response["intermediate_steps"]
        )
    )

    code_with_display = openai_client.chat.completions.create(
        model=st.session_state["openai_model"],
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": """You are creating a report for the user's question: """
                + st.session_state.current_user_input
                + """use st.write or st.image to display the result of the given thoughtflow of an agent that already did all the calculation needed to answer the question: \n\n\n------------------------\n"""
                + st.session_state.agent_thoughtflow
                + """
\n\n\n------------------------\nNote that all the results are already in the thoughtflow, you just need to print them out rather than trying to recalculate them.
Only use st.write(), st.image(), st.metric() and Streamlit functions to display. Do not reference df or variables directly.
Only respond with code as plain text without code block syntax around it. Again, do not write code to do any calculation. you are only here to print the results from the above thought flow""",
            }
        ],
    )

    return code_with_display


# ─────────────────────────────────────────────
# execute_plan
# ─────────────────────────────────────────────

def execute_plan(plan):
    print("[PLAN EXECUTION STARTED]")
    status_container = st.empty()

    steps = [line.strip("- •").strip() for line in plan.splitlines() if line.strip()]
    print(f"[PLAN] Total steps to execute: {len(steps)}")
    for i, step in enumerate(steps, 1):
        print(f"[PLAN] {i}. {step}")

    initial_state: CodePlanState = {
        "messages": [
            {"role": "user", "content": st.session_state.get("current_user_input", plan)}
        ],
        "plan": steps,
        "code_files": {},
        "test_results": {},
        "errors": [],
        "iterations": 0,
        "step_count": 0,
    }

    final_code_files = {}

    for step_output in langgraph_app.stream(initial_state, config={"recursion_limit": 500}):
        node_name = list(step_output.keys())[0]
        node_data = step_output[node_name] or {}


        if node_data.get("code_files"):
            final_code_files = node_data["code_files"]

    status_container.success("✅ Done!")
    formatted_output = ""

    if final_code_files:
        code_block = "\n\n".join(
            f"```python\n# --- {fname} ---\n{code}\n```"
            for fname, code in final_code_files.items()
        )
        formatted_output = f"### Final Generated Code\n\n{code_block}"

    st.session_state.formatted_output = formatted_output

    if final_code_files:
        print(f"\n[RESULTS] Generated {len(final_code_files)} code file(s)")
        all_code = "\n\n".join(
            f"# --- {fname} ---\n{code}" for fname, code in final_code_files.items()
        )

        # Collect stdout from all steps for the thoughtflow
        all_stdout = ""
        for filename, code in final_code_files.items():
            result = run_tests({filename: code})
            all_stdout += f"\n# {filename} output:\n{result.get('stdout', '')}"

        # Build response object matching generate_code_for_display_report interface
        langgraph_response = {
            "output": f"Successfully executed {len(final_code_files)} code files",
            "intermediate_steps": [
                type("Step", (), {
                    "tool": "langgraph_executor",
                    "tool_input": all_code,
                    "log": "LangGraph executed and tested the following code:\n" + all_code + "\n\nOutput:\n" + all_stdout,
                })()
            ],
        }

        code_for_displaying_report = generate_code_for_display_report(langgraph_response)
        st.session_state.code = code_for_displaying_report.choices[0].message.content

    st.session_state.messages.append(
        {"role": "assistant", "content": "Report generated! Check the Report panel below."}
    )

    print(f"\n{'=' * 60}")
    print("[PLAN EXECUTION COMPLETED]")
    print(f"{'=' * 60}\n")

    st.rerun()


# ─────────────────────────────────────────────
# Chatbot response
# ─────────────────────────────────────────────

@traceable(name="generate_chatbot_reponse")
def generate_chatbot_response(openai_client, session_state, user_input):
    stream = openai_client.chat.completions.create(
        model=session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": """You are a data analysis Copilot that is able to help user to generate report with data analysis in them. you are able to search on internet and you're able to help people to look into the table data from the user. however currently you can only do those if user is sending you a message stating clearly that they like to create a report. if they are not asking you about creating a report please try to answer their questions and explain what you can do to help, and ask them to create a report if that's their goal if you think it is needed, 
                   for example, create a report of column B and column C and caluclate the correlation between the two columns""",
            }
        ]
        + [{"role": m["role"], "content": m["content"]} for m in session_state.messages],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "trigger_report_generation",
                    "description": "Trigger this function when user asks about creating a report or any calculation to do with the existing dataset",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_message": {
                                "type": "string",
                                "description": "The user's message asking about creating a report or any calculation to do with the existing dataset",
                            }
                        },
                        "required": ["user_message"],
                    },
                },
            }
        ],
        tool_choice="auto",
    )

    response_message = stream.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls and tool_calls[0].function.name == "trigger_report_generation":
        st.write_stream(
            get_stream("Got it, here is a plan to create report for this request of yours:")
        )
        result = get_data(session_state.df)
        session_state.current_user_input = user_input

        plan = openai_client.chat.completions.create(
            model=session_state["openai_model"],
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": user_input
                    + """ \n make a simple plan that is simple to understand without technical terms to create code in python 
                        to analyze this data(do not include the code), only include the plan as list of steps in the output. 
                        At the same time, you are also given a list of tools, they are python_repl_tool for writing code, and another one is called web_search for searching on the web for knowledge you do not know. 
                        Please assign the right tool to do each step, knowing the tools that got activated later will know the output of the previous tools. 
                        the plan can be hierarchical, meaning that when multiple related and consecutive step can be grouped in one big step and be achieve by the same tool,
                        you can group under a parent step and have them as sub-steps and only mention the tool recommended for the partent step. try to limit your parent step to be less than 5 steps. 
                        At the each parent step of the plan, please indicate the tool you recommend in a [] such as [Tool: web_search], and put it at the begining of that step. Do not indicate the tool recommendation for sub-steps
                        In your output please only give one coherent plan with no analysis
                            """
                    + "\n this is the data \n"
                    + result,
                }
            ],
            stream=True,
        )
        response = st.write_stream(plan)
        st.write_stream(
            get_stream(
                "📝 If you like the plan, please click on 'Execute Plan' button on the 'Plan' tab in the top right panel. Or feel free to ask me to revise the plan in this chat"
            )
        )
        session_state.plan = response
    else:
        response = st.write_stream(get_stream(stream.choices[0].message.content))

    return response


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────

if "plan" not in st.session_state:
    st.session_state.plan = ""
if "code" not in st.session_state:
    st.session_state.code = """
st.write("There is no report created yet, please ask the chatbot to create a report if you need")
"""
if "thoughtflow" not in st.session_state:
    st.session_state.agent_thoughtflow = ""
if "current_user_input" not in st.session_state:
    st.session_state.current_user_input = ""


class SessionStateAutoClass:
    def __setattr__(self, name, value):
        if getattr(self, name, None) != value:
            st.session_state[name] = value
            st.rerun()

    def __getattr__(self, name):
        return st.session_state.get(name, None)


session_state_auto = SessionStateAutoClass()

if "formatted_output" not in st.session_state:
    st.session_state.formatted_output = ""
session_state_auto.formatted_output = st.session_state.formatted_output


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.set_page_config(layout="wide")

with st.container():
    col1row1, col2row1 = st.columns(2)

    with col1row1:
        with st.container(height=ROW_HIGHT):
            chat_history_container = st.container(height=ROW_HIGHT - TEXTBOX_HIGHT)
            with chat_history_container:
                chat_history_container.title("Chatbot")
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                for message in st.session_state.messages:
                    with chat_history_container.chat_message(message["role"]):
                        st.markdown(message["content"])

            input_textbox_container = st.container()
            with input_textbox_container:
                user_input = st.chat_input("What is up?")
                if user_input:
                    with chat_history_container.chat_message("user"):
                        st.markdown(user_input)
                    st.session_state.messages.append(
                        {"role": "user", "content": user_input}
                    )
                    with chat_history_container.chat_message("assistant"):
                        response = generate_chatbot_response(
                            openai_client, st.session_state, user_input
                        )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

    with col2row1:
        with st.container(height=ROW_HIGHT):
            col2row1_plan_tab, col2row1_code_tab = st.tabs(["Plan", "Code"])

            with col2row1_plan_tab:
                st.write(st.session_state.plan)
                if st.button("Execute Plan"):
                    execute_plan(st.session_state.plan)

            with col2row1_code_tab:
                st.write(session_state_auto.formatted_output)
                st.write("### Code For Visualizing Report")
                reporting_code = st_ace(
                    value=st.session_state.code, language="python", theme="monokai"
                )
                if reporting_code:
                    st.session_state.code = reporting_code

    col1row2, col2row2 = st.columns(2)

    with col1row2:
        with st.container(height=ROW_HIGHT):
            st.write("### User Data Set")
            if "df" not in st.session_state:
                st.session_state.df = get_dataframe()
            edited_df = st.data_editor(
                st.session_state.df,
                key="editable_table",
                num_rows="dynamic",
                on_change=handle_table_change,
            )
            st.session_state.df = edited_df

    with col2row2:
        with st.container(height=ROW_HIGHT):
            st.write("### AI Generated Report")
            exec(st.session_state.code)