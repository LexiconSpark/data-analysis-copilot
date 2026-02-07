"""LangChain test agent using the same API as streamlit_app (create_openai_functions_agent + AgentExecutor)."""
import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool

load_dotenv()

'''  res_dict = client.pull_repo(owner_repo_commit)'''
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# Wrap the function as a LangChain Tool (required by the 0.2.x agent API)
weather_tool = Tool(
    name="get_weather",
    description="Get weather for a given city. Input should be a city name.",
    func=lambda x: get_weather(x.strip()),
)

# Build agent with the same pattern as streamlit_app.py
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions="You are a helpful assistant.")
agent = create_openai_functions_agent(llm, [weather_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[weather_tool], verbose=True)

# Run the agent
result = agent_executor.invoke(
    {"input": "What is the weather in San Francisco?"}
)
print(result["output"])
