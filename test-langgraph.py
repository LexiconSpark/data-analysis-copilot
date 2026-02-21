from langchain.chat_models import init_chat_model
from langgraph.types import Command
from langgraph.graph import StateGraph, END, MessagesState
from typing_extensions import TypedDict, Literal


# State that tracks both planning and code execution
class CodePlanState(TypedDict):
    messages: list
    plan: list[str]  # Current plan steps
    code_files: dict  # Generated code by file
    test_results: dict  # Test outcomes
    errors: list  # Current errors
    iterations: int

# Node 1: Planning (uses Deep Agents planning tool)
def plan_node(state: CodePlanState) -> Command[Literal["write_code"]]:
    """Create initial plan or replan based on errors"""
    # Deep agent generates plan using write_todos tool
    plan = agent.invoke({"task": state["messages"][-1]["content"]})
    
    return Command(
        update={"plan": plan.steps, "iterations": 0},
        goto="write_code"
    )

# Node 2: Write Code (iterates through plan steps)
def write_code(state: CodePlanState) -> Command[Literal["check_code"]]:
    """Generate code for current plan step"""
    current_step = state["plan"][0]
    
    # Use CodeAct for code generation
    code_result = code_agent.invoke({
        "messages": [{"role": "user", "content": f"Implement: {current_step}"}],
        "context": state["code_files"]
    })
    
    return Command(
        update={"code_files": code_result.files},
        goto="check_code"
    )

# Node 3: Check Code (execute and test)
def check_code(state: CodePlanState) -> Command[Literal["rewrite_code", "update_plan", END]]:
    """Execute code and check for errors"""
    test_results = run_tests(state["code_files"])
    
    if test_results["passed"]:
        # Check if more steps in plan
        if len(state["plan"]) > 1:
            return Command(
                update={"plan": state["plan"][1:]},  # Remove completed step
                goto="write_code"
            )
        return Command(goto=END)  # All steps complete
    else:
        return Command(
            update={"errors": test_results["errors"]},
            goto="rewrite_code"
        )

# Node 4: Rewrite Code (fix errors)
def rewrite_code(state: CodePlanState) -> Command[Literal["check_code", "update_plan"]]:
    """Fix errors in current code"""
    if state["iterations"] >= 3:
        return Command(goto="update_plan")  # Too many retries, replan
    
    # Use reflection to fix errors
    fixed_code = code_agent.invoke({
        "messages": [{"role": "user", "content": "Fix these errors"}],
        "errors": state["errors"],
        "code": state["code_files"]
    })
    
    return Command(
        update={"code_files": fixed_code.files, "iterations": state["iterations"] + 1},
        goto="check_code"
    )

# Node 5: Update Plan (adaptive replanning)
def update_plan(state: CodePlanState) -> Command[Literal["write_code", END]]:
    """Revise plan based on errors encountered"""
    new_plan = agent.invoke({
        "task": "Revise plan to address errors",
        "errors": state["errors"],
        "current_plan": state["plan"]
    })
    
    return Command(
        update={"plan": new_plan.steps, "iterations": 0},
        goto="write_code"
    )

# Build the graph
graph = StateGraph(CodePlanState)
graph.add_node("planner", plan_node)
graph.add_node("write_code", write_code)
graph.add_node("check_code", check_code)
graph.add_node("rewrite_code", rewrite_code)
graph.add_node("update_plan", update_plan)

# Command pattern handles all routing automatically
graph.set_entry_point("planner")


app = graph.compile()

if __name__ == "__main__":
    from unittest.mock import MagicMock
    
    agent = MagicMock()
    agent.return_value.steps = ["Step 1: Create main.py", "Step 2: Add tests"]

    code_agent = MagicMock()
    code_agent.return_value.files = {"main.py": "print('hello')"}
    
    run_tests_results = iter([
        {"passed": False, "errors": ["TypeError on line 5"]},  # first check_code
        {"passed": True, "errors": []},                         # after rewrite
        {"passed": True, "errors": []},                         # step 2
        {"passed": True, "errors": []},                         # add this
    ])
    run_tests = lambda files: next(run_tests_results)
    
    for step in app.stream({
        "messages": [{"role": "user", "content": "Build a data analysis script"}],
        "plan": [],
        "code_files": {},
        "test_results": {},
        "errors": [],
        "iterations": 0
    }):
        print(step)
        print("---")