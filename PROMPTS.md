# AI Data Analyst Co-Pilot — Setup & Prompt Guide

Complete setup guide and all prompts for building the AI Data Analyst Co-Pilot using OpenSpec, Figma MCP, Next.js, LangGraph.js, and Vercel.

---

## Stack

| Layer | Tool |
|---|---|
| Frontend | Next.js + TypeScript + Tailwind CSS |
| UI Streaming | Vercel AI SDK (`useChat`, streaming) |
| Agentic Workflow | LangGraph.js |
| LLM | OpenAI GPT-4o |
| CSV Parsing | Papa Parse |
| Charts | Recharts |
| Tracing | Vercel AI SDK telemetry |
| Deployment | Vercel |
| UI Design | Figma Console MCP |
| Specs | OpenSpec |

---

## 1. Terminal Setup

```bash
# Navigate into your existing empty project folder
cd your-folder-name

# Scaffold Next.js into current folder
npx create-next-app@latest . --typescript --tailwind --app

# Install AI + graph dependencies
npm install ai @ai-sdk/openai @langchain/langgraph @langchain/core zod papaparse recharts
npm install -D @types/papaparse

# Init OpenSpec — select Claude Code when prompted
npx @fission-ai/openspec@latest init

> **Note:** When `create-next-app` asks "Would you like to use React Compiler?" — select **No**.

---

## 2. Figma Console MCP Setup

### Get your Figma Personal Access Token
1. Go to figma.com → click your profile → **Settings**
2. Scroll to **Personal access tokens** → Generate new token
3. Copy it (starts with `figd_`) — paste it into the command below


# Run in terminal - Add Figma Console MCP (works with free Figma accounts)
claude mcp add figma-console -s user \
  -e FIGMA_ACCESS_TOKEN=figd_YOUR_TOKEN_HERE \
  -e ENABLE_MCP_APPS=true \
  -- npx -y figma-console-mcp@latest
```


### Install the Desktop Bridge Plugin
```bash
# Clone the repo to get the bridge plugin
git clone https://github.com/southleft/figma-console-mcp.git ~/figma-console-mcp
```

Then in Figma Desktop:
1. **Main menu** → **Plugins** → **Development** → **Import plugin from manifest...**
2. Select `~/figma-console-mcp/figma-desktop-bridge/manifest.json`

### Run the Bridge Plugin
1. Open your Figma design file
2. **Main menu(left panel)** → **Plugins** → **Development** → **Figma Desktop Bridge**
3. Click to run — plugin panel appears confirming connection

### Add your design sketchs to claude 
1. **Main menu** → **New design file**
2. Drag and drop your design images onto the canvas
   OR **Main menu** → **File** → **Place image...**

---

## 3. Inside Claude Code — Full Prompt Sequence

### Step 1 — Verify Figma MCP connection
```
Check Figma status or /mcp 
```

### Step 2 — Create the design you want in Figma
```
(I did this in claude cli)
1. Add your design screenshots to claude (copy-pasting images or drag and drop)
2. Give this prompt - Please scaffold this design sketch in a new board in figma
3. Check for what's generated in Figma. You can continue prompting till you get the desired design
```

### Step 3 — Run the OpenSpec proposal
```
/opsx:propose "Build an AI Data Analyst Co-Pilot using Next.js App Router, TypeScript, Tailwind CSS, deployed on Vercel.

UI Layout — based on Figma designs, 2x2 grid, 4 quadrants, fixed height, visible borders:
- Top-left: chat interface with streaming (Vercel AI SDK useChat hook)
- Top-right: tabs — Templates (default), Plan, Code
- Bottom-left: editable CSV table with default dataset and upload option
- Bottom-right: analysis results and charts (Recharts)

AI Architecture:
- Vercel AI SDK handles streaming between UI and Next.js API routes
- LangGraph.js handles the agentic backend workflow with nodes:
  1. Plan node — generates step-by-step analysis plan
  2. Interrupt — pauses for human approval
  3. Execute node — generates and runs code per step
  4. Retry node — on failure LLM rewrites code up to 3 times, then revises plan
  5. Report node — renders results and charts
  6. Cancellation check at every node
- State schema includes retry counters and cancellation flag from the start
- LangGraph checkpointer per session for interrupt/resume

Features:
- Save and load reusable plan templates (localStorage)
- Vercel AI SDK telemetry for tracing
- CSV data passed as context to all AI nodes
- Code tab shows execution blocks in order, syntax highlighted, always visible
- Plan tab auto-activates when waiting for approval
- Code tab auto-activates after run completes
- Approve Plan button on Plan tab
- Save as Template button on Plan tab

Tech: Next.js, TypeScript, Tailwind, Vercel AI SDK, OpenAI GPT-4o, LangGraph.js, Papa Parse, Recharts, Vercel deployment."
```

### Step 4 — Review generated spec files
OpenSpec will create:
- `openspec/changes/your-change/proposal.md` — why and what
- `openspec/changes/your-change/specs/` — requirements
- `openspec/changes/your-change/design.md` — technical approach
- `openspec/changes/your-change/tasks.md` — implementation checklist

Review these before proceeding.

### Step 5 — Implement
```
/opsx:apply
```

### Step 6 — Archive after implementation
```
/opsx:archive
```

---

## 4. Deploy to Vercel

```In terminal
npx vercel
```
First it would ask for authentication. So create an account in Vercel.com beforehand.

Add the env variables (OpenAI api key, in our case) in vercel, select project -> Left panel -> Settings -> Env variables, then redeploy

---

## 5. Making Changes After Implementation

For any future change, use the same OpenSpec flow:

```
/opsx:propose "describe your change here"
```

Then:
```
/opsx:apply    ← implements the change
/opsx:archive  ← consolidates and archives it
```

---

## 6. OpenSpec Command Reference

| Command | When to use |
|---|---|
| `/opsx:propose "your idea"` | Always start here — generates spec before any code |
| `/opsx:apply` | After reviewing proposal — implements the tasks |
| `/opsx:continue` | If implementation was interrupted — picks up where it left off |
| `/opsx:archive` | After implementation — consolidates specs and archives change |

---

## 7. How the Stack Fits Together

```
User (Next.js UI)
    ↕ Vercel AI SDK (streaming, useChat)
Next.js API Route
    ↕ calls
LangGraph.js (backend workflow)
    → Plan node
    → Interrupt (human approval)
    → Execute node
    → Retry node (up to 3x, then revise plan)
    → Report node
```

- **Vercel AI SDK** = the bridge between your UI and AI (streaming, tool calls)
- **LangGraph.js** = the brain that orchestrates the multi-step agentic workflow
- **OpenAI GPT-4o** = the LLM powering all nodes
- **Figma Console MCP** = pulls your design into Claude Code as context
- **OpenSpec** = keeps specs and intent documented alongside your code