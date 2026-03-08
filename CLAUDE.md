# data-analysis-copilot Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-03

## Active Technologies
- Python 3.11+ + streamlit>=1.31.0, langchain>=0.3.0, langchain-openai>=0.1.0, langchain-community>=0.3.0, **langgraph==0.3.18 (to add)**, python-dotenv>=1.0.0, pandas>=2.0.0, matplotlib>=3.7.0, pydantic>=2.0.0, langsmith>=0.1.0 (004-langgraph-langsmith)
- Session state only (Streamlit); templates as JSON files in local `templates/` folder; artifacts (charts) in `artifacts/` folder (cleared per run) (004-langgraph-langsmith)

- Python 3.11+ + streamlit>=1.28.0, langchain>=0.3.0, langchain-openai>=0.1.0, langchain-community>=0.3.0, openai>=1.0.0, python-dotenv>=1.0.0, pandas>=2.0.0 (001-gpt4o-chatbot)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.11+: Follow standard conventions

## Recent Changes
- 004-langgraph-langsmith: Added Python 3.11+ + streamlit>=1.31.0, langchain>=0.3.0, langchain-openai>=0.1.0, langchain-community>=0.3.0, **langgraph==0.3.18 (to add)**, python-dotenv>=1.0.0, pandas>=2.0.0, matplotlib>=3.7.0, pydantic>=2.0.0, langsmith>=0.1.0
- 001-gpt4o-chatbot: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]
- 001-gpt4o-chatbot: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
