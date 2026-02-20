import os
import streamlit as st
import pandas as pd
import numpy as np
import random
import time

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.tools import PythonAstREPLTool
from langchain_community.llms import OpenAI as OpenAI_2
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMMathChain
import langsmith
from langsmith.wrappers import wrap_openai
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain import hub


from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
import matplotlib.pyplot as plt
import re
from langsmith import Client as LangSmithClient
from streamlit_chat import message
from streamlit_ace import st_ace
from openai import OpenAI

from langsmith.wrappers import wrap_openai
from langsmith import traceable

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

load_dotenv()
ROW_HIGHT = 600
TEXTBOX_HIGHT = 90


def initialize_environment():
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "data_analysis_copilot"  # enable tracing
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    #openai_client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    
    return (    
        LangSmithClient(),
        wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY"))),
        os.getenv("OPENAI_API_KEY"),

)
    
# Initialize an OpenAI client, this will be used for handling individual AI tasks in the code as well as chatbot for the the top left cornor
langsmith_client, openai_client, OPENAI_API_KEY = initialize_environment()


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"


# function to convert a sentence to stream
def get_stream(sentence):
    for word in sentence.split():
        yield word + " "
        time.sleep(0.05)


# function to get the dataframe of the csv
def get_data(placeholder):
    new_data = st.session_state.df.to_csv()
    return new_data


# function to generate a random dataframe
def get_dataframe():
    df = pd.DataFrame(
        {
            "A": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "B": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
            "C": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        }
    )
    return df


# This is the function for handling changes in csv
def handle_table_change():
    if "table_changed" in st.session_state and st.session_state["table_changed"]:
        st.session_state["chat_history"].append(
            {"role": "bot", "content": "A change was made to the table."}
        )


# function to generate the reponse of the chatbot
@traceable(nme="generate_chatbot_reponse")
def generate_chatbot_response(openai_client, session_state, user_input):
    # generate the response from openai api
    stream = openai_client.chat.completions.create(
        model=session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": """You are a data analysis Copilot that is able to help user to generate report with data analysis in them. you are able to search on internet and you're able to help people to look into the table data from the user. however currently you can only do those if user is sending you a message stating clearly that they like to create a report. if they are not asking you about creating a report please try to answer their questions and explain what you can do to help, and ask them to create a report if that's their goal if you think it is needed, 
                   for example, create a report of column B and column C and caluclate the correlation between the two columns""",
            }
        ]
        + [
            {"role": m["role"], "content": m["content"]} for m in session_state.messages
        ],
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
    # get the tool calls
    tool_calls = response_message.tool_calls
    # if the tool is called to generated report
    if tool_calls and tool_calls[0].function.name == "trigger_report_generation":
        # display the message when the report is created
        st.write_stream(
            get_stream(
                "Got it, here is a plan to create report for this request of yours:"
            )
        )
        # get the dataframe of the csv
        result = get_data(session_state.df)
        # update the current user input
        session_state.current_user_input = user_input
        # generate the plan
        plan = openai_client.chat.completions.create(
            model=session_state["openai_model"],
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
        # display the plan
        response = st.write_stream(plan)
        # update the plan
        st.write_stream(
            get_stream(
                "📝 If you like the plan, please click on 'Execute Plan' button on the 'Plan' tab in the top right panel. Or feel free to ask me to revise the plan in this chat"
            )
        )
        session_state.plan = response
    # if a simple data question is asked
    elif tool_calls and tool_calls[0].function.name == "simple_data_analysis":
        # call the data agent
        data_agent = create_pandas_dataframe_agent(
            ChatOpenAI(
                temperature=0,
                api_key="sk-proj-pPMRDpoxQeXFmBk1HGmRT3BlbkFJRPax8CTo4YfwzzgmCXJD",
            ),
            st.session_state.df,
            verbose=True,
        )
        # generate response
        answer = data_agent.invoke(user_input)["output"]
        asnwer_reported = openai_client.chat.completions.create(
            model=session_state["openai_model"],
            messages=[
                {
                    "role": "user",
                    "content": "Based on the following answer, "
                    + answer
                    + " answer this question with a simple sentence"
                    + user_input,
                }
            ],
            stream=True,
        )
        response = st.write_stream(asnwer_reported)
    elif tool_calls and tool_calls[0].function.name == "simple_data_analysis":
        # call the data agent
        data_agent = create_pandas_dataframe_agent(
            ChatOpenAI(
                temperature=0,
                api_key="sk-proj-pPMRDpoxQeXFmBk1HGmRT3BlbkFJRPax8CTo4YfwzzgmCXJD",
            ),
            st.session_state.df,
            verbose=True,
        )
        # generate response
        answer = data_agent.invoke(user_input)["output"]
        asnwer_reported = openai_client.chat.completions.create(
            model=session_state["openai_model"],
            messages=[
                {
                    "role": "user",
                    "content": "Based on the following answer, "
                    + answer
                    + " answer this question with a simple sentence"
                    + user_input,
                }
            ],
            stream=True,
        )
        response = st.write_stream(asnwer_reported)
    else:
        response = st.write_stream(get_stream(stream.choices[0].message.content))

    return response


# Functions to execute the plan generated
def execute_plan(plan):

    # initialize the tools
    ddg_search = DuckDuckGoSearchResults()

    def _ddg_wrapper(tool_input):
        if isinstance(tool_input, dict):
            query = tool_input.get("query") or tool_input.get("input") or tool_input.get("text") or ""
        else:
            query = str(tool_input)
        return ddg_search.run({"query": query})

    websearch_tool = Tool(
        name="web_search",
        func=_ddg_wrapper,
        description="Useful for when you need to answer questions about current events or the current state of the world, by searching on internet",
    )

    # Initialize the Python REPL tool
    python_repl_tool = PythonREPLTool(
        name="python_repl_tool",
        description="""
You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question.
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
if you are ask to plot data, save the plot as "plot.png", and log the name of the image.
Note that you only have python libaraies including Pandas, NumPy, Matplotlib, Math. Do not use any other libaraies
You can use st.write("your answer") to display the answer if you are asked to calculate something.
Anything of the part of the code that is todo with searching on the internet please do not write code for it. just skip it and write a comment as a placeholder.
""",
    )

    tools = [python_repl_tool, websearch_tool]

    # This defines the langchain agent's general behavior
    instructions = """
    You will receive an instruction with different sub steps under the parent steps, each parent step will have recommneded tools for you to use.
    You will trigger the corresponding recommneded tool to execute different parent steps along with its substep in one big step.
    and clearly noting down the results from the each steps.
    """
    # Pull the base prompt (using LangSmith; hub is deprecated)
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)

    # Create the agent
    agent = create_openai_functions_agent(
        ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), tools, prompt
    )

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
    )

    # building the input data
    input_data = (
        "execute this plan "
        + plan
        + " with this dataseset "
        + st.session_state.df.to_csv()
    )
    '''
    result = langgraph_app.invoke({"messages": [HumanMessage(content=input_data)]})
    langraph_repsonse = result["messages"][-1].content
    '''
    # Execute the agent
    response = agent_executor.invoke({"input": input_data})

    # generate the code for displaying the report
    code_for_displaying_report = generate_code_for_display_report(response)

    # update the code
    st.session_state.code = code_for_displaying_report.choices[0].message.content

    # update the chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response["output"]}
    )

    # format the agent intermediate steps, having this step at the botton of the function because it uses the special session_state_auto variable, as it could trigger st.rerun() within it and sometimes interrupt other steps.
    session_state_auto.formatted_output = format_intermediate_steps(response)

    st.rerun()


# Function to format the intermediate steps from the agent for display
def format_intermediate_steps(response):
    formatted_output = "### Intermediate Steps\n"
    steps = intermediate_steps = response.get("intermediate_steps", [])
    for step in steps:
        tool = step[0].tool
        tool_input = step[0].tool_input
        log = step[0].log.strip()
        formatted_output += (
            f"Invoked `{tool}` with: \n```python\n\n{tool_input}\n```\n\n"
        )
    formatted_output += f"Final Output: \n `{response['output']}`"
    return formatted_output


# Function to generate code that is used to display the information within the report
def generate_code_for_display_report(execution_agent_response):

    # update the agent thoughtflow(all of steps agent took)
    st.session_state.agent_thoughtflow = (
        "Here is the final output: "
        + str(execution_agent_response["output"])
        + """
Here is the log of the different step's output, you will be able to find the useful information within there: \n"""
        + "".join(
            str(step[0].log) for step in execution_agent_response["intermediate_steps"]
        )
    )

    # generate the code for displaying the report
    code_with_display = openai_client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "user",
                "content": """You are creating a report for the user's question: """
                + st.session_state.current_user_input
                + """use st.write or st.image to display the result of the given thoughtflow of an agent that already did all the calculation needed to answer the question: \n\n\n------------------------\n"""
                + st.session_state.agent_thoughtflow
                + """
\n\n\n------------------------\nNote that all the results are already in the thoughtflow, you just need to print them out rather than trying to recalculate them.
Only respond with code as plain text without code block syntax around it. Again, do not write code to do any calculation. you are only here to print the results from the above thought flow""",
            }
        ],
    )

    return code_with_display


# Initialize the state variables
if "plan" not in st.session_state:
    st.session_state.plan = ""
if "code" not in st.session_state:
    st.session_state.code = """    
st.write("There is no report created yet, please ask the chatbot to create a report if you need")
"""
if "thoughtflow" not in st.session_state:
    st.session_state.agent_thoughtflow = ""
if "kb_chunks" not in st.session_state:
    st.session_state.kb_chunks = []   # list of {"text":..., "meta":...}
if "kb_embs" not in st.session_state:
    st.session_state.kb_embs = None   # np.ndarray shape (n, d), L2-normalized

# Below is a method for creating a state variable that auto refreshes on the frontend as the value changes, without the need to manually do st.rerun()
# Be careful with using it since it uses the special session_state_auto variable, as it could trigger st.rerun() within it and sometimes interrupt other steps.
# Find the tutorial of implimentation here: https://discuss.streamlit.io/t/i-created-way-simpler-cleaner-session-state-code-with-auto-refresh/36150
class SessionStateAutoClass:
    def __setattr__(self, name, value):
        if getattr(self, name, None) != value:
            st.session_state[name] = value
            st.rerun()

    def __getattr__(self, name):
        return st.session_state.get(name, None)


session_state_auto = SessionStateAutoClass()

# Initialize session_state_auto.formatted_output as a state variable that auto refreshes on the frontend
if "formatted_output" not in st.session_state:
    st.session_state.formatted_output = ""
session_state_auto.formatted_output = st.session_state.formatted_output


################################################
##### Below are all the code to do with UI #####
################################################

# set the page config
st.set_page_config(layout="wide")

# Create the main container of the UI
with st.container():
    # Create two columns for the top row
    col1row1, col2row1 = st.columns(2)

    # Add chatbot to the first quadrant (top-left)
    with col1row1:
        with st.container(height=ROW_HIGHT):
            chat_history_container = st.container(height=ROW_HIGHT - TEXTBOX_HIGHT)
            with chat_history_container:
                chat_history_container.title("Chatbot")
                # Initialize chat history if it doesn't exist
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                # Display chat messages from history on app rerun
                for message in st.session_state.messages:
                    with chat_history_container.chat_message(message["role"]):
                        st.markdown(message["content"])

            input_textbox_container = st.container()
            with input_textbox_container:
                # Collect user input via chat
                user_input = st.chat_input("What is up?")
                if user_input:
                    # Display user message in chat message container
                    with chat_history_container.chat_message("user"):
                        st.markdown(user_input)
                    # Add user message to chat history
                    st.session_state.messages.append(
                        {"role": "user", "content": user_input}
                    )

                    # Display assistant response in chat message container
                    with chat_history_container.chat_message("assistant"):
                        response = generate_chatbot_response(
                            openai_client, st.session_state, user_input
                        )
                    # Append the assistant's response to the chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                    # Orchestrator integration: let user route this input to specialized agents
                    if st.button("Route with Orchestrator", key="route_orchestrator"):
                        from langgraph_orchestrator import initialize_orchestrator

                        # Data agent wrapper
                        class DataAgentWrapper:
                            def invoke(self, payload):
                                if isinstance(payload, dict):
                                    task = payload.get("input") or payload.get("task") or ""
                                else:
                                    task = str(payload)
                                try:
                                    data_agent = create_pandas_dataframe_agent(
                                        ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY),
                                        st.session_state.df,
                                        verbose=True,
                                    )
                                    # langchain pandas agent returns dict or text depending on version
                                    return data_agent.invoke(task)
                                except Exception as e:
                                    return {"error": str(e)}

                        # Execution agent wrapper (reuses execute_plan logic without UI side-effects)
                        class ExecAgentWrapper:
                            def invoke(self, payload):
                                if isinstance(payload, dict):
                                    task = payload.get("input") or payload.get("task") or ""
                                else:
                                    task = str(payload)

                                # initialize the tools
                                ddg_search = DuckDuckGoSearchResults()

                                def _ddg_wrapper_local(tool_input):
                                    if isinstance(tool_input, dict):
                                        query = tool_input.get("query") or tool_input.get("input") or tool_input.get("text") or ""
                                    else:
                                        query = str(tool_input)
                                    return ddg_search.run({"query": query})

                                websearch_tool = Tool(
                                    name="web_search",
                                    func=_ddg_wrapper_local,
                                    description="Search the web",
                                )

                                python_repl_tool = PythonREPLTool(
                                    name="python_repl_tool",
                                    description="Python REPL for running code",
                                )

                                tools = [python_repl_tool, websearch_tool]

                                instructions_local = "Execute the provided plan using the recommended tools."
                                base_prompt_local = hub.pull("langchain-ai/openai-functions-template")
                                prompt_local = base_prompt_local.partial(instructions=instructions_local)

                                agent_local = create_openai_functions_agent(
                                    ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
                                    tools,
                                    prompt_local,
                                )

                                agent_executor_local = AgentExecutor(
                                    agent=agent_local, tools=tools, verbose=True, return_intermediate_steps=True
                                )

                                input_data = "execute this plan " + task + " with this dataset " + st.session_state.df.to_csv()
                                try:
                                    return agent_executor_local.invoke({"input": input_data})
                                except Exception as e:
                                    return {"error": str(e)}

                        # Try to include the code agent if available
                        registry = {"data": DataAgentWrapper(), "executor": ExecAgentWrapper()}
                        try:
                            from langgraph_code_agent import app as code_agent_app

                            registry["code"] = code_agent_app
                        except Exception:
                            pass

                        orchestrator = initialize_orchestrator(registry)
                        out = orchestrator.invoke({"task": user_input})
                        st.write("Orchestrator chose:", out.get("agent"))
                        st.write("Result:", out.get("result"))

    # create the second column in first row
    with col2row1:
        with st.container(height=ROW_HIGHT):
            # create the tabs
            col2row1_plan_tab, col2row1_code_tab = st.tabs(["Plan", "Code"])

            # display the plan
            with col2row1_plan_tab:
                st.write(st.session_state.plan)

                # if the execute button is pressed
                if st.button("Execute Plan"):
                    execute_plan(st.session_state.plan)

            # display the code for formatting the report
            with col2row1_code_tab:
                # print out the intermediate steps
                st.write(session_state_auto.formatted_output)

                # Display code for visualizating report
                st.write("### Code For Visualizing Report")
                reporting_code = st_ace(
                    value=st.session_state.code, language="python", theme="monokai"
                )

                # if the update button is pressed
                if reporting_code:
                    st.session_state.code = reporting_code

    # create second row
    col1row2, col2row2 = st.columns(2)

    # Add editable table to the third quadrant (bottom-left)
    with col1row2:
        with st.container(height=ROW_HIGHT):
            st.write("### User Data Set")

            # if the dataframe is not in the session state
            if "df" not in st.session_state:
                st.session_state.df = get_dataframe()

            # display the editable table
            edited_df = st.data_editor(
                st.session_state.df,
                key="editable_table",
                num_rows="dynamic",
                on_change=handle_table_change,
            )

            # update the dataframe
            st.session_state.df = edited_df

    # create the fourth column in second row
    with col2row2:
        with st.container(height=ROW_HIGHT):
            st.write("### AI Generated Report")

            # execute the code
            exec(reporting_code)
            
            exec(reporting_code)

class OrchestratorState(TypedDict):
    task: str
    agent: str
    result: str
    messages: Annotated[list, "foobar"]


# shared model used for routing/classification
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)


def initialize_orchestrator(agent_registry: dict):
    """Create and return a compiled StateGraph orchestrator.

    agent_registry: mapping of agent_key -> compiled_langgraph_app
    The orchestrator will classify incoming `task` strings to choose an
    agent key, dispatch the task to that agent, and collect the result.
    """

    def receive(state: OrchestratorState):
        return state

    def classify(state: OrchestratorState):
        prompt = (
            "You are an orchestrator router.\n"
            f"Available agents: {', '.join(agent_registry.keys())}.\n"
            "Given the following task, reply with exactly the agent key that should handle it.\n"
            f"Task: {state.get('task', '')}"
        )
        resp = model.invoke([HumanMessage(content=prompt)])
        chosen = getattr(resp, "content", str(resp)).strip()
        if chosen not in agent_registry:
            chosen = next(iter(agent_registry)) if agent_registry else ""
        return {"agent": chosen}

    def dispatch(state: OrchestratorState):
        agent_key = state.get("agent")
        agent_app = agent_registry.get(agent_key)
        if agent_app is None:
            return {"result": {"error": f"unknown agent '{agent_key}'"}}

        # Attempt to call the agent with a messages payload first
        try:
            result = agent_app.invoke({"messages": [HumanMessage(content=state.get("task", ""))]})
        except Exception:
            try:
                result = agent_app.invoke({"task": state.get("task", "")})
            except Exception as e:
                result = {"error": str(e)}

        return {"result": result}

    def collect(state: OrchestratorState):
        return state

    graph = StateGraph(OrchestratorState)
    graph.add_node("receive", receive)
    graph.add_node("classify", classify)
    graph.add_node("dispatch", dispatch)
    graph.add_node("collect", collect)

    graph.add_edge("__start__", "receive")
    graph.add_edge("receive", "classify")
    graph.add_edge("classify", "dispatch")
    graph.add_edge("dispatch", "collect")
    graph.add_edge("collect", "__end__")

    return graph.compile()


