import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Any
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


class OrchestratorState(TypedDict):
    task: str
    agent: str
    result: Any
    normalized_result: dict  # canonical output: {"text": "...", "meta": {...}}
    messages: Annotated[list, add_messages]


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

        task_text = state.get("task", "")

        def try_invoke(obj, payload):
            try:
                return obj.invoke(payload)
            except Exception:
                raise

        # Try a few common invocation patterns for LangGraph / LangChain agents
        result = None
        errors = []
        # 1) LangGraph compiled apps expect dict state, often with `messages`
        try:
            result = agent_app.invoke({"messages": [HumanMessage(content=task_text)]})
        except Exception as e:
            errors.append(str(e))

        # 2) LangChain AgentExecutor / tools-style: invoke with {"input": ...}
        if result is None:
            try:
                result = agent_app.invoke({"input": task_text})
            except Exception as e:
                errors.append(str(e))

        # 3) Some `invoke` implementations accept the raw task string
        if result is None:
            try:
                result = agent_app.invoke(task_text)
            except Exception as e:
                errors.append(str(e))

        # 4) Fallbacks: `run` method or callable
        if result is None:
            try:
                if hasattr(agent_app, "run"):
                    result = agent_app.run(task_text)
                elif callable(agent_app):
                    result = agent_app(task_text)
                else:
                    result = {"error": "un-callable agent"}
            except Exception as e:
                errors.append(str(e))

        if result is None:
            result = {"error": "all invocation attempts failed", "errors": errors}

        return {"result": result}

    def normalize(state: OrchestratorState):
        """Convert agent output into a canonical {"text": "...", "meta": {...}} format."""
        result = state.get("result", {})
        normalized = {"text": "", "meta": {"agent": state.get("agent"), "task": state.get("task")}}

        if isinstance(result, dict):
            # LangChain AgentExecutor: {"output": "..."}
            if "output" in result:
                normalized["text"] = str(result["output"])
                normalized["meta"]["intermediate_steps"] = result.get("intermediate_steps", [])
            # LangGraph: {"messages": [...]}
            elif "messages" in result:
                messages_list = result.get("messages", [])
                if messages_list:
                    last_msg = messages_list[-1]
                    normalized["text"] = getattr(last_msg, "content", str(last_msg))
                normalized["meta"]["messages_count"] = len(messages_list)
            # Pandas agent or other: {"text": "..."}
            elif "text" in result:
                normalized["text"] = str(result["text"])
            # Error dict
            elif "error" in result:
                normalized["text"] = f"Error: {result['error']}"
                normalized["meta"]["error"] = True
            else:
                # fallback: convert dict to string
                normalized["text"] = str(result)
        elif isinstance(result, str):
            normalized["text"] = result
        else:
            # Raw object
            normalized["text"] = str(result)

        return {"normalized_result": normalized}

    def collect(state: OrchestratorState):
        return state

    graph = StateGraph(OrchestratorState)
    graph.add_node("receive", receive)
    graph.add_node("classify", classify)
    graph.add_node("dispatch", dispatch)
    graph.add_node("normalize", normalize)
    graph.add_node("collect", collect)

    graph.add_edge("__start__", "receive")
    graph.add_edge("receive", "classify")
    graph.add_edge("classify", "dispatch")
    graph.add_edge("dispatch", "normalize")
    graph.add_edge("normalize", "collect")
    graph.add_edge("collect", "__end__")

    return graph.compile()


if __name__ == "__main__":
    # simple example using the provided code agent if available
    try:
        from langgraph_code_agent import app as code_agent_app

        registry = {"code_agent": code_agent_app}
        orchestrator = initialize_orchestrator(registry)
        resp = orchestrator.invoke({"task": "Write a short Python function that computes Fibonacci numbers."})
        print("Orchestrator result (raw):", resp.get("result"))
        print("Orchestrator result (normalized):", resp.get("normalized_result"))
    except Exception as e:
        print("Example run failed:", e)
