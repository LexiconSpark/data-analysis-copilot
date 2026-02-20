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