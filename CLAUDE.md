# data-analysis-copilot Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-03

## Active Technologies
- Python 3.11+ + streamlit==1.49.1, langchain==1.2.9, langchain-openai==1.1.9, langchain-community==0.4.1, langgraph==1.0.10, python-dotenv==1.0.0, pandas==2.2.3, matplotlib==3.9.2, pydantic==2.8.2, langsmith==0.7.14 (004-langgraph-langsmith)
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
- 004-langgraph-langsmith: Upgraded to latest stable versions (langchain 1.2.9, langgraph 1.0.10, streamlit 1.49.1)
- 004-langgraph-langsmith: Resolved all dependency conflicts (numpy 2.1.0, all packages have pre-built wheels)
- 004-langgraph-langsmith: Completed Session 2 implementation (plan approval, code retry, plan revision, cancellation)
- 001-gpt4o-chatbot: Original chatbot implementation with structured outputs


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
