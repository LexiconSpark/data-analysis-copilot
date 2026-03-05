# Run these in terminal 

## Setup virtual environment for python
```
python3.12 -m venv .venv
```

```
source .venv/bin/activate
```

## Install spec-kit once and use everywhere:


```bash

uv tool install specify-cli --from git+https://github.com/github/spec-kit.git

```

## Use speckit to start the project:

 
```bash
# Create new project
specify init --here
```
Note: if your terminal does not recognize the "specify" command at this step, you might need to add the specify command into the system environment variable. 
Afterwards select claude code, then select sh (if you are on macOS/Linux)

# Run in claude code
## Implement the initial project
Launch claude code in the project directory. The `/speckit.*` commands are available in the assistant.

Note: Further tutorial to followthrough for using speckit: https://github.com/github/spec-kit
 
Use the **`/speckit.constitution`** command to create your project's governing principles and development guidelines that will guide all subsequent development.


```bash

/speckit.constitution Create principles focused on code quality, testing standards, user experience consistency, and performance requirements. Project is in Python.
```

Here's the full conversation history for this session:

```
/speckit.specify Build a conversational chatbot using OpenAI GPT-4o with conversation memory and it should have a simple interface
```
```
/speckit.plan This chatbot should run in streamlit and use langchain architecture for its implementation. Model and temperature can be specified in the file.
```
```
/speckit.clarify why is there config.toml file
```
```
/speckit.tasks break down into tasks
```
```
/speckit.implement
```

After implementing this if there are any errors, please use claude code to fix the bugs, you don't need to use speckit


## Feature improvements

### Add langsmith
```
/speckit.plan I want to add langsmith tracing to this.
```

### Add csv
```
/speckit.plan I want to extend this app by adding a csv to the context - either user can upload a csv or use the default csv data
            "A": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "B": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
            "C": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
and it is passed as context for the chatbot. The csv data should also be displayed on the streamlit UI. The user should be able to interact with the chatbot based on the csv. Keep the changes in this git branch only.
```
