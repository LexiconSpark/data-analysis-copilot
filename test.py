from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI



import pandas as pd
from langchain_openai import OpenAI

df = pd.DataFrame(
        {
            "A": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "B": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
            "C": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        }
    )


agent = create_pandas_dataframe_agent(OpenAI(temperature=0,api_key= "sk-proj-pPMRDpoxQeXFmBk1HGmRT3BlbkFJRPax8CTo4YfwzzgmCXJD"), df, verbose=True)


print(agent.invoke("Calculate the covariance between B and C")["output"])