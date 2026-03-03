# Quickstart: GPT-4o Conversational Chatbot

**Branch**: `001-gpt4o-chatbot` | **Date**: 2026-03-03

---

## Prerequisites

- Python 3.11 or later
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

## Setup

### 1. Clone the repository and switch to the feature branch

```bash
git clone <repository-url>
cd data-analysis-copilot
git checkout 001-gpt4o-chatbot
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root (this file is git-ignored and must never be committed):

```bash
# .env
OPENAI_API_KEY=sk-your-api-key-here
```

### 5. Adjust model settings (optional)

Open `app.py` and edit the constants block near the top of the file:

```python
MODEL = "gpt-4o"       # OpenAI model name
TEMPERATURE = 0.7      # 0.0 (deterministic) to 2.0 (creative)
SYSTEM_PROMPT = "You are a helpful assistant."
```

Default settings work out of the box. Supported models include `"gpt-4o"` and `"gpt-4o-mini"`.

---

## Run the App

```bash
streamlit run app.py
```

Streamlit will open the chatbot in your default browser at `http://localhost:8501`.

---

## Usage

| Action | How |
|--------|-----|
| Send a message | Type in the input box at the bottom and press **Enter** or click the send arrow |
| View conversation history | Scroll up in the conversation thread |
| Start a fresh conversation | Click the **New Chat** button at the top |
| Change model or temperature | Edit the `MODEL` / `TEMPERATURE` constants at the top of `app.py` and restart the app |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `AuthenticationError` | Missing or invalid API key | Check `.env` contains a valid `OPENAI_API_KEY` |
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` with venv active |
| Rate limit error | OpenAI quota exceeded | Wait and retry; check your OpenAI usage dashboard |
| App opens but no response | Wrong model name | Check the `MODEL` constant at the top of `app.py` is a valid OpenAI model name |

---

## Deploying to Streamlit Community Cloud

1. Push your branch to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository
3. Set the main file to `app.py`
4. Add `OPENAI_API_KEY` under **Settings → Secrets** (do not commit `.env` to the repository)
5. Deploy
