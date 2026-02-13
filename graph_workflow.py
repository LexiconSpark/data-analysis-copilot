"""
LangGraph workflow for data analysis copilot
"""
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langsmith import traceable
import pandas as pd
import os


# Define the state for our graph
class DataAnalysisState(TypedDict):
    """State for the data analysis copilot workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    dataframe_csv: str
    user_intent: Literal["report_generation", "simple_qa", "general_chat"]
    plan: str
    execution_output: str
    intermediate_steps: list
    report_code: str
    error: str | None


@traceable(name="classify_intent", tags=["intent_classification"])
def classify_intent_node(state: DataAnalysisState) -> dict:
    """Classify user intent from their message"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Get last user message
    last_message = state["messages"][-1].content

    # Classify intent
    system_prompt = """Analyze the user's message and respond with ONLY ONE of these exact words:
    - "report_generation" if they want to create a report, do calculations, or analyze data
    - "simple_qa" if they're asking a simple question about the data
    - "general_chat" if they're just chatting or asking what you can do

    Only respond with one of those three exact phrases, nothing else."""

    classification = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": last_message}
    ])

    intent = classification.content.strip().lower()

    # Default to general_chat if unclear
    if "report" in intent:
        intent = "report_generation"
    elif "simple" in intent or "qa" in intent:
        intent = "simple_qa"
    else:
        intent = "general_chat"

    return {"user_intent": intent}


@traceable(name="generate_plan", tags=["planning", "report_generation"])
def generate_plan_node(state: DataAnalysisState) -> dict:
    """Generate execution plan for report creation"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    user_request = state["messages"][-1].content
    dataframe = state["dataframe_csv"]

    plan_prompt = f"""Create a plan to analyze this data and answer: {user_request}

Data preview:
{dataframe[:500]}

Available tools:
- python_repl_tool: For writing Python code to analyze data
- web_search: For searching the internet

Output a clear step-by-step plan with [Tool: tool_name] for each parent step.
Try to limit to less than 5 parent steps.
Format each step clearly with the recommended tool.
"""

    plan = llm.invoke([{"role": "user", "content": plan_prompt}])

    return {"plan": plan.content}


@traceable(name="execute_plan", tags=["execution", "tools"])
def execute_plan_node(state: DataAnalysisState) -> dict:
    """Execute the plan using tools"""

    # Initialize tools
    python_repl_tool = PythonREPLTool(
        name="python_repl_tool",
        description="""Execute python code to analyze data.
If you need to plot data, save the plot as "plot.png".
Available libraries: Pandas, NumPy, Matplotlib.
Use st.write() to display results."""
    )

    websearch_tool = DuckDuckGoSearchResults(name="web_search")

    tools = [python_repl_tool, websearch_tool]

    # Create agent
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    instructions = """You will receive a plan with different steps and recommended tools.
Execute each step using the recommended tool.
Clearly note down the results from each step.
"""

    base_prompt = hub.pull("langchain-ai/openai-tools-template")
    prompt = base_prompt.partial(instructions=instructions)

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )

    # Build input
    input_data = f"""Execute this plan:
{state['plan']}

With this dataset:
{state['dataframe_csv']}
"""

    # Execute
    result = agent_executor.invoke({"input": input_data})

    return {
        "execution_output": result["output"],
        "intermediate_steps": result.get("intermediate_steps", [])
    }


@traceable(name="generate_report_code", tags=["code_generation"])
def generate_report_code_node(state: DataAnalysisState) -> dict:
    """Generate Streamlit code to display the report"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Build thoughtflow from intermediate steps
    thoughtflow = "Final output: " + str(state['execution_output']) + "\n\nIntermediate steps:\n"
    thoughtflow += "".join(str(step[0].log) for step in state.get('intermediate_steps', []))

    code_prompt = f"""Generate Streamlit code to display these results:

{thoughtflow}

Requirements:
- Use st.write() or st.image() to display results
- If there's a plot.png mentioned, use st.image("plot.png")
- Only return plain Python code, no markdown code blocks
- Do NOT recalculate anything, just display the results
"""

    code = llm.invoke([{"role": "user", "content": code_prompt}])

    return {"report_code": code.content}


@traceable(name="simple_qa", tags=["qa", "pandas"])
def simple_qa_node(state: DataAnalysisState) -> dict:
    """Handle simple Q&A about the dataframe"""

    # Convert CSV back to dataframe
    df = pd.read_csv(pd.StringIO(state["dataframe_csv"]))

    # Create pandas agent
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_type="openai-tools"
    )

    user_question = state["messages"][-1].content
    answer = agent.invoke(user_question)

    return {"execution_output": answer["output"]}


@traceable(name="general_chat", tags=["chat"])
def general_chat_node(state: DataAnalysisState) -> dict:
    """Handle general conversation"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    system_message = """You are a data analysis copilot. You can help users:

1. Create reports by analyzing their data (just ask me to "create a report about X")
2. Answer simple questions about their data
3. Explain what you can do

Currently, the user has a dataset loaded. If they want to analyze it, ask them to request a report."""

    response = llm.invoke([
        {"role": "system", "content": system_message},
        {"role": "user", "content": state["messages"][-1].content}
    ])

    return {"execution_output": response.content}


def route_by_intent(state: DataAnalysisState) -> str:
    """Conditional edge: route based on user intent"""
    intent = state.get("user_intent", "general_chat")

    if intent == "report_generation":
        return "generate_plan"
    elif intent == "simple_qa":
        return "simple_qa"
    else:
        return "general_chat"


def create_data_analysis_graph():
    """Create and compile the LangGraph workflow"""

    # Initialize graph
    workflow = StateGraph(DataAnalysisState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("generate_plan", generate_plan_node)
    workflow.add_node("execute_plan", execute_plan_node)
    workflow.add_node("generate_report_code", generate_report_code_node)
    workflow.add_node("simple_qa", simple_qa_node)
    workflow.add_node("general_chat", general_chat_node)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Add conditional edges from classify_intent
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "generate_plan": "generate_plan",
            "simple_qa": "simple_qa",
            "general_chat": "general_chat"
        }
    )

    # Add sequential edges for report generation flow
    workflow.add_edge("generate_plan", "execute_plan")
    workflow.add_edge("execute_plan", "generate_report_code")
    workflow.add_edge("generate_report_code", "__end__")

    # Add edges to END for other paths
    workflow.add_edge("simple_qa", "__end__")
    workflow.add_edge("general_chat", "__end__")

    # Compile the graph
    app = workflow.compile()

    return app
