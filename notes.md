# one-off error

After inputting: "give the mean of column A"
-> asked if I wanted to generate report
-> generate report
2026-02-17 09:09:25.482 Uncaught app exception
Traceback (most recent call last):
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 589, in _run_script
    exec(code, module.__dict__)
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/streamlit_app.py", line 470, in <module>
    execute_plan(st.session_state.plan)
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/streamlit_app.py", line 306, in execute_plan
    response = agent_executor.invoke({"input": input_data})
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain/chains/base.py", line 166, in invoke
    raise e
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain/chains/base.py", line 156, in invoke
    self._call(inputs, run_manager=run_manager)
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain/agents/agent.py", line 1433, in _call
    next_step_output = self._take_next_step(
                       ^^^^^^^^^^^^^^^^^^^^^
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain/agents/agent.py", line 1139, in _take_next_step
    [
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain/agents/agent.py", line 1224, in _iter_next_step
    yield self._perform_agent_action(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain/agents/agent.py", line 1246, in _perform_agent_action
    observation = tool.run(
                  ^^^^^^^^^
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain_core/tools/base.py", line 586, in run
    raise error_to_raise
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain_core/tools/base.py", line 555, in run
    response = context.run(self._run, *tool_args, **tool_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain_core/tools/simple.py", line 84, in _run
    return self.func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain_core/tools/base.py", line 586, in run
    raise error_to_raise
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain_core/tools/base.py", line 549, in run
    tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain_core/tools/base.py", line 472, in _to_args_and_kwargs
    tool_input = self._parse_input(tool_input)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/langchain_core/tools/base.py", line 424, in _parse_input
    result = input_args.parse_obj(tool_input)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/pydantic/v1/main.py", line 526, in parse_obj
    return cls(**obj)
           ^^^^^^^^^^
  File "/home/emeka/data-analysis-recloned/data-analysis-copilot/venv/lib/python3.12/site-packages/pydantic/v1/main.py", line 341, in __init__
    raise validation_error
pydantic.v1.error_wrappers.ValidationError: 1 validation error for DDGInput

# LangGraph Structure
have the plan. Write code and then replan workflow, and then also have the write code check error and rewrite code loop, these two loops. And then so they added together.

# Orchestrator example
# create agents exactly like in your app
data_agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY),
    st.session_state.df,
    verbose=True,
)

# reuse your execution agent creation (AgentExecutor) if needed
agent = create_openai_functions_agent(
    ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
    tools, prompt
)
execution_agent = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

# optional compiled LangGraph code agent
from langgraph_code_agent import app as code_agent_app

registry = {
    "data": data_agent,
    "executor": execution_agent,
    "code": code_agent_app,
}

orchestrator = initialize_orchestrator(registry)

# invoke orchestrator with task
resp = orchestrator.invoke({"task": "Compute the mean and stddev of column A and return a short summary."})
print("chosen agent:", resp.get("agent"))
print("raw result:", resp.get("result"))

# orchestrator agent added with integrated workflow:

#### **1. Real Multi-Agent Architecture**
```
Before: START → mock_llm → END

After:  START → Orchestrator → Worker → Evaluator → END
```

#### **2. Intelligent Task Classification**
- **Orchestrator node** uses LLM to classify user requests
- Routes to appropriate specialized worker (planning vs code)
- No hardcoded logic - flexible to handle various request types

#### **3. Specialized Workers**
- **Planning Worker**: Creates analysis plans
- **Code Worker**: Writes and executes Python code
- Each worker focused on specific domain → better quality outputs

#### **4. Quality Control Layer**
- **Evaluator node** checks all worker outputs
- Catches errors and hallucinations before reaching user
- Provides "APPROVED" or "REJECTED" with reasons
- Significantly reduces bad outputs

#### **5. Linear Flow (No Recursion)**
- Workers → Evaluator → END (no loops back)
- Eliminates GraphRecursionError
- Predictable execution path
- Clearer debugging

#### **6. Enhanced State Management**

Tracks complete workflow from classification → execution → evaluation

#### **7. Conditional Routing**
- `route_to_worker()` function dynamically routes based on orchestrator's classification
- Extensible - easy to add more workers
- No hardcoded paths

### **Key Architectural Pattern Implemented:**

**Orchestrator-Worker-Evaluator Pattern** (recommended by LangGraph docs for production systems)

```
┌─────────────┐
│ Orchestrator│ ← Classifies task
└──────┬──────┘
       ↓
   [Routes to]
       ↓
┌─────────────┐
│   Workers   │ ← Specialized execution
└──────┬──────┘
       ↓
┌─────────────┐
│  Evaluator  │ ← Quality control
└──────┬──────┘
       ↓
    Output
```

### **Practical Benefits:**
- ✅ Production-ready structure
- ✅ Handles multiple task types
- ✅ Self-correcting (evaluator catches errors)
- ✅ Scalable (add workers easily)
- ✅ No infinite loops
- ✅ Clear execution path for debugging
- ✅ Follows LangGraph best practices

### NOTE: LangGraph is a Directed Graph

START → Orchestrator → [Planning Worker OR Code Worker] → Evaluator → END

This is a directed graph where:
- Nodes = functions (orchestrator, workers, evaluator)
- Edges = transitions between nodes
- Directed = flow goes one way (can't go backwards)

#### graph traversal

#### state space searching
class OrchestratorState(TypedDict):
    # This is like a "state" in state space search
    user_request: str       # Initial state
    assigned_worker: str    # Intermediate state
    worker_output: str      # Intermediate state
    final_output: str       # Goal state

#### decision trees
def route_to_worker(state):
    if "code" in worker:
        return "code_worker"      # Branch A
    else:
        return "planning_worker"   # Branch B

This is like:
- If-then rules in expert systems
- Decision nodes in decision trees
- Conditional branches in search algorithms

#### Google's Pregel
# Traditional graph algorithms:
def bfs(graph, start):
    queue = [start]
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            queue.append(neighbor)  # Pass "message" to neighbor

# LangGraph does similar:
def execute_graph(state):
    current_node = START
    while current_node != END:
        state = current_node.execute(state)  # Pass state (message)
        current_node = get_next_node(state)   # Based on edges
```

## **Your Graph Visualized as Search Tree**
```
                    START
                      |
                [Orchestrator]  ← Root node
                   /      \
                  /        \
        [Planning Worker] [Code Worker]  ← Search branches
                  \        /
                   \      /
                 [Evaluator]  ← Converging paths
                      |
                    END  ← Goal state

### Next Optimization Steps

#### Implement an A* Search
# In more complex graphs, you could add "costs" to edges:

def route_with_cost(state):
    if complexity_score < 5:
        return "simple_worker"  # Lower cost path
    else:
        return "complex_worker" # Higher cost path

LangGraph could choose paths based on:
- Token cost
- Execution time
- Quality metrics

