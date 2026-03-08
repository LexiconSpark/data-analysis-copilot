# Quickstart: AI Data Analyst Copilot — Core UI Layout

**Branch**: `002-analyst-copilot-ui` | **Date**: 2026-03-05

---

## Prerequisites

- Python 3.11+
- `OPENAI_API_KEY` set in `.env`

---

## Setup

```bash
# 1. Install updated dependencies (streamlit bump to >=1.31.0)
pip install -r requirements.txt

# 2. Verify API key is configured
cat .env  # Should contain: OPENAI_API_KEY=sk-...

# 3. Run the app
streamlit run app.py
```

---

## What You'll See

The app opens in your browser with four quadrants:

```
┌────────────────────┬────────────────────┐
│  Chat (top-left)   │  Tabs (top-right)  │
│                    │  Template|Plan|Code │
│  [Conversation]    │  [AI reasoning]    │
│  [Chat input]      │                    │
├────────────────────┼────────────────────┤
│  CSV (bot-left)    │  Results (bot-rgt) │
│  [Data table]      │  [Analysis output] │
│  [Upload / Reset]  │                    │
└────────────────────┴────────────────────┘
```

---

## Basic Workflow

### 1. Chat with the default dataset

Type a question in the chat input:

```
Plot the relationship between columns A and B
```

- The AI responds in the **Chat** panel with a summary
- The **Plan** tab shows numbered analysis steps
- The **Code** tab shows executable Python code

### 2. Execute the plan

Click **"Execute Plan"** in the Plan tab to run the generated code.

- A chart or data table appears in the **Results** panel (bottom-right)

### 3. Save a template

In the **Template** tab:
1. Enter a name (e.g., "Column A vs B scatter")
2. Click **"Save as Template"**

The template is saved for the rest of your session and can be reloaded from the Template tab.

### 4. Upload your own CSV

In the **CSV** panel (bottom-left):
1. Click **"Browse files"**
2. Select any `.csv` file
3. The table updates immediately

You can now ask questions about your own data.

### 5. Reset

- **"Reset to default data"** — restores the original sample dataset (chat and templates are preserved)
- **"New Chat"** — clears the conversation history (CSV and templates are preserved)

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `LANGSMITH_TRACING` | No | Set to `true` to enable LangSmith tracing |
| `LANGSMITH_API_KEY` | No | LangSmith API key (required if tracing enabled) |
| `LANGSMITH_PROJECT` | No | LangSmith project name (default: `data-analysis-copilot`) |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "API key is invalid" | Check `.env` contains a valid `OPENAI_API_KEY` |
| Bottom rows not visible | Ensure browser zoom is ≤100%; app targets 1920×1080 |
| Code execution error | The AI occasionally generates code with variable name mismatches — retry with a rephrased question |
| Upload fails | Confirm the file is a valid UTF-8 CSV; files >10,000 rows are truncated automatically |
