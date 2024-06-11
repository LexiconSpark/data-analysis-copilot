import streamlit as st
from streamlit_ace import st_ace

# Initialize session state for the code value if not already set
if 'code' not in st.session_state:
    st.session_state.code = "print('Hello, World!')"

# Function to update the code
def update_code():
    st.session_state.code = "print('Updated Code!')"

# Display the Ace editor with the current code value
content = st_ace(value=st.session_state.code, language='python', theme='monokai')

# Button to update the code
if st.button('Update Code'):
    update_code()

print(type(content))

# Display the current content of the editor
st.write("Editor content:")
st.write(content)