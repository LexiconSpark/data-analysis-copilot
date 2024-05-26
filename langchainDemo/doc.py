import os
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools

os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0.6)

tools = load_tools(["wikipedia", "llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("When was Queen Elizabeth II born? What's her age right now? Square it and show the result")