import streamlit as st
import pandas as pd
import random
from streamlit_chat import message
from streamlit_ace import st_ace

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

# Function to simulate chatbot response
def chatbot_response(user_input):
    responses = [
        "Hello! How can I help you today?",
        "I'm here to assist you with any questions.",
        "What would you like to know?",
        "Feel free to ask me anything.",
    ]
    return random.choice(responses)

# Function to handle changes in the editable table
def handle_table_change():
    st.session_state['table_changed'] = True

# Create the main container
with st.container():
    # Create two columns for the top row
    col1row1, col2row1 = st.columns(2)
    
    # Add chatbot to the first quadrant (top-left)
    with col1row1:
        with st.container():
            st.write("Chatbot 1")
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []
            user_input1 = st.text_input("You: ", key="input1")
            if user_input1:
                st.session_state['chat_history'].append({"role": "user", "content": user_input1})
                response1 = chatbot_response(user_input1)
                st.session_state['chat_history'].append({"role": "bot", "content": response1})
            for i, msg in enumerate(st.session_state['chat_history']):
                message(msg["content"], is_user=msg["role"] == "user", key=f"msg1_{i}")
    
    # Add code editor to the second quadrant (top-right)
    with col2row1:
        with st.container():
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
            st.session_state.code = """st.header("Streamlit Sandbox")
st.write("Play with Streamlit live in the browser!")

table_data = {'Column 1': [1, 2], 'Column 2': [3, 4]}
st.write(pd.DataFrame(data=table_data))
"""
            code = st_ace(
                value=st.session_state.code,
                language="python",
                placeholder="st.header('Hello world!')",
                theme=st.selectbox("Theme", options=THEMES, index=26, key="theme"),
                keybinding=st.selectbox("Keybinding mode", options=KEYBINDINGS, index=3, key="keybinding"),
                font_size=st.slider("Font size", 5, 24, 14, key="font_size"),
                tab_size=st.slider("Tab size", 1, 8, 4, key="tab_size"),
                wrap=st.checkbox("Wrap lines", value=False, key="wrap"),
                show_gutter=True,
                show_print_margin=True,
                auto_update=False,
                readonly=False,
                key="ace-editor"
            )
            # Question: how to edit code in here

            st.session_state.code += "#new commment"
            
            
            st.write('Hit `CTRL+ENTER` to refresh')
            st.write('*Remember to save your code separately!*')
    
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