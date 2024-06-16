import streamlit as st
import pandas as pd
import random
import time
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.tools import PythonAstREPLTool
from langchain import OpenAI as OpenAI_2
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMMathChain

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
import matplotlib.pyplot as plt
import re


from streamlit_chat import message
from streamlit_ace import st_ace
from openai import OpenAI

load_dotenv()

## using langsmith 
# video for how langsmith is used in this demo code: https://share.descript.com/view/k4b3fyvaESB
# To learn about this: https://youtu.be/tFXm5ijih98
# To check the running result of langsmith, please go to: https://smith.langchain.com/
from langsmith import Client
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "data_analysis_copilot"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
#Initialize LangSmith client
client_langsmith = Client()

client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

#list the functions want open ai to use
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_data",
            "description": "get the dataframe as csv formate",
            "parameters": {
                "type": "object",
                "properties": {
                    "placeholder": {
                        "type": "string",
                        "description": "the placeholder of the method",
                    }
                },
                "required": ["message"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
]

def get_stream(sentence):
        
        for word in sentence.split():
            yield word + " "
            time.sleep(0.05)


def get_data(placeholder):

    new_data = st.session_state.df.to_csv()


    return new_data
# Function to generate a random dataframe


def generate_random_dataframe():
    df = pd.DataFrame(
        {
            "A": [random.randint(0, 100) for _ in range(10)],
            "B": [random.randint(0, 100) for _ in range(10)],
            "C": [random.randint(0, 100) for _ in range(10)],
        }
    )
    return df


# Function to handle changes in the editable table
def handle_table_change():
    st.session_state['table_changed'] = True

st.set_page_config(layout='wide')

if "plan" not in st.session_state:
    st.session_state.plan = ""
if "code" not in st.session_state:
    st.session_state.code = """st.header("Streamlit Sandbox")
st.write("Play with Streamlit live in the browser!")

table_data = {'Column 1': [1, 2], 'Column 2': [3, 4]}
st.write(pd.DataFrame(data=table_data))"""
# Create the main container
with st.container():
    # Create two columns for the top row
    col1row1, col2row1 = st.columns(2)
    
    # Add chatbot to the first quadrant (top-left)
    with col1row1:
                
                container1 = st.container(height = 700)
                with container1:
                    container1.title("Chatbot")
                    # Initialize chat history
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    # Display chat messages from history on app rerun
                    for message in st.session_state.messages:
                        with container1.chat_message(message["role"]):
                            st.markdown(message["content"])
                            
                with st.container():

                    user_input1 = st.chat_input("What is up?")
                    if user_input1:
                # Display user message in chat message container
                        with container1.chat_message("user"):
                            st.markdown(user_input1)
                # Add user message to chat history
                            st.session_state.messages.append({"role": "user", "content": user_input1})

# Display assistant response in chat message container
                        message = {"role": "system", "content":"You are a data analysis expert if you are asked anything about the data, make a plan for implementing code to analyze"}
                        with container1.chat_message("assistant"):
                            stream = client.chat.completions.create(
                            model=st.session_state["openai_model"],
                            messages=[{"role": "system", "content":"You are a data analysis expert if you are asked to analyze data, get the data make a plan for implementing code to analyze"}]+
                                [{"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.messages
                                    ],
                                
                                tools = tools,
                                tool_choice = "auto"
                            )
                            # Append the message to messages list
                            response_message = stream.choices[0].message 

                            #response = st.write_stream(stream)
                            tool_calls = response_message.tool_calls
                            if tool_calls:
                                tool_call_id = tool_calls[0].id
                                tool_function_name = tool_calls[0].function.name

                                result = get_data(st.session_state.df)
                            
                                plan = client.chat.completions.create(
                                    model = st.session_state["openai_model"],
                                    messages = [{"role": "user", "content": user_input1 +" \n make a plan that is simple to understand without technical terms to create code in python to analyze this data(do not include the code), only include the plan as list of steps in the output"+ "\n this is the data \n" + result}],
                                    stream = True
                                )   
                                
 
                            
                                response = st.write_stream(plan)
                                st.session_state.plan = response
                            else:
                                response = st.write_stream(get_stream(stream.choices[0].message.content))
                    
                        st.session_state.messages.append({"role": "assistant", "content": response})
    with col2row1:
        
        tab1, tab2, tab3 = st.tabs(["Template", "Plan", "Code"])
        with st.container():
            with tab2:
                st.write(st.session_state.plan)
                
                if st.button("Execute Plan"):
#                   load_dotenv()
                    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
                    # Initialize the language model
                    # llm = OpenAI_2(temperature=0)# Define the search tool

                    # llm_math = LLMMathChain(llm=llm)
                    # math_tool = Tool(
                    # name='Calculator',
                    # func=llm_math.run,
                    # description='Useful for when you need to answer questions about math.'
                    # )

                    ddg_search = DuckDuckGoSearchResults()
                    websearch_tool = Tool(
                        name="Current_Search",
                        func=ddg_search.run,
                         description="Useful for when you need to answer questions about current events or the current state of the world, by searching on internet"
                 )

                    # python_repl = PythonREPLTool()
                    # python_tool = Tool(
                    #     name = "Python Executer",
                    #     func = python_repl.run,
                    #     description = "Useful for when you want to test the code you wrote to debug"
                    # )
                    # # Define the tools the agent will use
                    # tools = [math_tool, websearch_tool,python_tool]

                    # # Initialize the agent with memory
                    # memory = ConversationBufferMemory(memory_key="chat_history")
                    # agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory = memory)

                    # # Define the input query
                    # input_query = "write a code to execute this plan " + st.session_state.plan + " with this dataseset "+ st.session_state.df.to_csv()+" and display the result using st.write or st.image" + "only use the code you write to answer the question"

                    # # Run the agent
                    # response = agent.run(input=input_query, handle_parsing_errors=True)
                    # print(response)
                    tools = [PythonREPLTool(), websearch_tool]
# Define the instructions for the agent
                    instructions = """
You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question.
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
if you are ask to plot data, save the plot as "plot.png", then always use st.image("plot.png") to display the plot.
You can use st.write("your answer") to display the answer if you are asked to calculate something.
Anything of the part of the code that is todo with searching on the internet please do not write code for it. just skip it and write a comment as a placeholder
"""

# Pull the base prompt
                    base_prompt = hub.pull("langchain-ai/openai-functions-template")
                    prompt = base_prompt.partial(instructions=instructions)

# Create the agent
                    openai_api_key_1 = os.getenv("OPENAI_API_KEY")
                    agent = create_openai_functions_agent(ChatOpenAI(temperature=0, openai_api_key=openai_api_key_1), tools, prompt)

# Create the agent executor
                    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose= False, return_intermediate_steps = True)

# Example input
                    input_data = "execute this plan "+ st.session_state.plan + " with this dataseset "+ st.session_state.df.to_csv()

# Execute the agent
                    response = agent_executor.invoke({"input": input_data})
                    generated_code = ""
                    #import pdb; pdb.set_trace()
                    
                    for i in range(0,len(response["intermediate_steps"])):
                         generated_code += response["intermediate_steps"][i][0].tool_input
                    
                    #st.session_state.code = generated_code
                    code_with_display = client.chat.completions.create(
                            model=st.session_state["openai_model"],
                            messages = [{"role": "user", "content": "use st.write or st.image to display the result of the given code"+ generated_code
                                         +" Only respond with code as plain text without code block syntax around it."}],
                            
                    )
                    #print(generated_code)

                    
                    st.session_state.code = code_with_display.choices[0].message.content
                    #print("new"+new_code)
                    st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                    st.experimental_rerun()
                    # instructions_display = "display the result of the given code using st.write or st.image"
                    # agent_display = create_openai_functions_agent(ChatOpenAI(temperature=0, openai_api_key=openai_api_key_1), tools, prompt)


            with tab3:
                st.write('### Code editor')
                THEMES = [
                "ambiance", "chaos", "chrome", "clouds", "clouds_midnight" "cobalt", "crimson_editor", "dawn",
                "dracula", "dreamweaver", "eclipse", "github", "gob", "gruvbox", "idle_fingers", "iplastic",
                "katzenmilch", "kr_theme", "kuroir", "merbivore", "merbivore_soft", "mono_industrial", "monokai",
                "nord_dark", "pastel_on_dark", "solarized_dark", "solarized_light", "sqlserver", "terminal",
                "textmate", "tomorrow", "tomorrow_night", "tomorrow_night_blue", "tomorrow_night_bright",
                "tomorrow_night_eighties", "twilight", "vibrant_ink", "xcode"
            ]
                KEYBINDINGS = ["emacs", "sublime", "vim", "vscode"]

                code = st_ace(value=st.session_state.code, language='python', theme='monokai')
            #     code = st_ace(
            #     value=st.session_state.code,
            #     language="python",
            #     placeholder="st.header('Hello world!')",
            #     theme=st.selectbox("Theme", options=THEMES, index=26, key="theme"),
            #     keybinding=st.selectbox("Keybinding mode", options=KEYBINDINGS, index=3, key="keybinding"),
            #     font_size=st.slider("Font size", 5, 24, 14, key="font_size"),
            #     tab_size=st.slider("Tab size", 1, 8, 4, key="tab_size"),
            #     wrap=st.checkbox("Wrap lines", value=False, key="wrap"),
            #     show_gutter=True,
            #     show_print_margin=True,
            #     auto_update=True,
            #     readonly=False,
            #     key="ace-editor"
            # )
            # Question: how to edit code in here
                #if the update button is pressed
                if code:
                    st.session_state.code  = code
                    
            
                #st.write('Hit `CTRL+ENTER` to refresh')
                #st.write('*Remember to save your code separately!*')

    # Create two columns for the bottom row
    col1row2, col2row2 = st.columns(2)
    
    # Add editable table to the third quadrant (bottom-left)
    with col1row2:
        with st.container():
            st.write("Editable Table with Random Values")
            if 'df' not in st.session_state:
                st.session_state.df = generate_random_dataframe()
            edited_df = st.data_editor(
                st.session_state.df,
                key="editable_table",
                num_rows="dynamic",
                on_change=handle_table_change
            )
            st.session_state.df = edited_df
    
    # Add content to the fourth quadrant (bottom-right)
    with col2row2:
        with st.container():

            st.write("### Code Output")
            exec(code)

# Check if the table was changed and send a message in the chatbot
if 'table_changed' in st.session_state and st.session_state['table_changed']:
    st.session_state['chat_history'].append({"role": "bot", "content": "A change was made to the table."})
    st.session_state['table_changed'] = False