import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# 1. Define state (what flows through the graph)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Model and nodes
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


def call_model(state: AgentState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def run_code(state: AgentState):
    return state  # placeholder


# 3. Build and compile the graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("code_executor", run_code)
graph.add_edge("__start__", "agent")
graph.add_edge("agent", "code_executor")
graph.add_edge("code_executor", "__end__")
app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"messages": [HumanMessage(content="what is the square root of -1?")]})
    print("Agent reply:", result["messages"][-1].content)
