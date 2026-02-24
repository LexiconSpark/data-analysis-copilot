# Data Analysis Copilot

An AI-powered data analysis assistant built with Streamlit and LangGraph. Upload CSV data, ask questions, and generate visual reports through natural conversation.

Video Walk through of the (earlier version) codebase: https://share.descript.com/view/m0Hx8rH6NC0

## Features

- **Interactive Data Editor** - Load and edit CSV data directly in the browser
- **AI Chat Interface** - Ask questions about your data in plain English
- **Automated Report Generation** - Request analysis reports and the AI creates a step-by-step plan, generates Python code, validates it, and displays the results
- **Smart Intent Classification** - Automatically routes between report generation, simple Q&A, and general chat
- **Self-Correcting Code Execution** - LangGraph workflow with up to 3 retry attempts and adaptive replanning on failure
- **LangSmith Tracing** - Optional observability for debugging LLM calls and agent workflows

## Architecture

The app uses a LangGraph state machine to orchestrate the report generation pipeline:

```
User Request
    |
[Classify Intent] ──> Simple Q&A ──> Pandas Agent ──> Display Answer
    |                  General Chat ──> LLM Response ──> Display
    v
[Generate Plan] ──> User reviews plan
    |
[Write Code] ──> Generate Python script
    |
[Check Code] ──> Syntax + Security + Logic + Runtime validation
    |
  Success? ─── Yes ──> [Generate Display Code] ──> Render Report
    |
    No (retries < 3) ──> [Rewrite Code] ──> Check Code
    |
    No (retries >= 3) ──> [Update Plan] ──> Write Code
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit 1.36 |
| LLM Orchestration | LangGraph 0.3 |
| LLM Provider | OpenAI GPT-4o |
| LLM Framework | LangChain 0.3 |
| Observability | LangSmith (optional) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Altair |
| Web Search | DuckDuckGo Search |

## Setup

### Prerequisites

- Python 3.12
- An OpenAI API key

### Installation

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY="your-key-from-https://platform.openai.com/api-keys"
   ```

4. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

   The app will open at `http://localhost:8501/`.

### Optional: LangSmith Tracing

To enable detailed tracing of LLM calls and agent workflows:

1. Get an API key from [smith.langchain.com](https://smith.langchain.com/) > Settings > API Keys

2. Add to your `.env` file:
   ```
   LANGSMITH_API_KEY="your-langsmith-api-key"
   ```

3. Add this to `streamlit_app.py`:
   ```python
   os.environ["LANGCHAIN_TRACING_V2"] = "true"
   os.environ["LANGCHAIN_PROJECT"] = "data_analysis_copilot"
   os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
   ```

4. View traces at [smith.langchain.com](https://smith.langchain.com/)

## Project Structure

```
├── streamlit_app.py               # Main application
├── requirements.txt               # Python dependencies
├── readme.md                      # This file
├── .env                           # API keys (not committed)
├── .python-version                # Python 3.12
└── code-for-learning/             # Educational reference code
    ├── graph_workflow.py           # Standalone LangGraph workflow example
    └── streamlit_app_langchain.py  # Alternative LangChain agent approach
```

## Usage

1. **Load data** - Edit or paste CSV data in the bottom-left data editor
2. **Ask a question** - Type in the chat box (e.g., "Create a report analyzing sales trends")
3. **Review the plan** - The AI generates a step-by-step analysis plan in the Plan tab
4. **Execute** - Click "Execute Plan" to run the analysis
5. **View results** - Charts and summaries appear in the AI Generated Report panel

## Branch Navigation

- Use `main` and `dev` branches to view different stages of the project
- Branches `V0.2_Data_Analysis_Copilot` and `V0.2_Data_Analysis_Copilot_dev` contain earlier versions

