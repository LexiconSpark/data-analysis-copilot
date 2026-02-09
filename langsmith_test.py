"""LangChain test agent using the same API as streamlit_app (create_openai_functions_agent + AgentExecutor)."""
import os
from dotenv import load_dotenv
#from langchain import hub
from langchain_openai import ChatOpenAI
# used to trace openai calls
from langsmith.wrappers import wrap_openai 
from langsmith import traceable
from openai import OpenAI

load_dotenv()

def retriever(query: str):
    return["test query"]

client = wrap_openai(OpenAI())

@traceable
def rag(question: str) -> str:
    docs = retriever(question)
    system_message = (
        "Answer the user's question using only the provided information below:\n"
        + "\n".join(docs)
    )
    
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    print(rag("Where did how did query go?"))
