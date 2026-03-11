# AI Data Analyst Co-Pilot

## What
An intelligent data analysis tool that lets users analyze datasets through a conversational interface. Users describe their analysis needs in natural language, the system generates a step-by-step plan, executes it interactively with human approval checkpoints, and returns visualized results with insights.

## Why
Data analysis today requires switching between tools (spreadsheets, Python, charting libraries) and writing code. This copilot brings analysis into a unified, conversational interface where non-technical users can ask questions about their data and get instant answers with full visibility into the analysis process.

## Key Capabilities
- **Conversational Interface**: Chat-based input for natural language analysis requests
- **Plan Generation**: AI generates numbered step-by-step analysis plans with human review/edit before execution
- **Interactive Execution**: Steps run with intermediate output visible, LLM can retry failures up to 3x before revising plan
- **Visualizations**: Automatic chart generation (scatter plots, line charts, etc.) using Recharts
- **Data Management**: CSV upload, preview, and editing in an interactive table
- **Template System**: Save favorite analysis plans for reuse
- **Interruptible Workflow**: Pause for human approval at plan stage, resume execution after edits
- **Session Persistence**: Save and resume work via LangGraph checkpointer

## User Experience
1. User uploads a CSV dataset → see preview in editable table
2. Describe analysis goal in chat (e.g., "Show vibration vs temperature for temps below 60")
3. System generates a step-by-step plan, auto-activates Plan tab
4. Review plan, optionally edit individual steps
5. Click "Approve Plan" → system executes each step, showing code and results
6. View final visualizations and conclusions in Results tab
7. Save plan as reusable template for similar analyses

## Technology
- **Frontend**: Next.js 16 App Router, React 19, TypeScript, Tailwind CSS 4
- **UI Layout**: 2x2 grid with chat, tabs, data table, and results pane
- **Streaming**: Vercel AI SDK for real-time updates
- **AI Backend**: OpenAI GPT-4o with LangGraph.js agentic workflow
- **Workflow Nodes**: Plan → Interrupt → Execute → Retry → Report → (Cancellation checks)
- **Data**: Papa Parse for CSV handling, Recharts for visualizations
- **Deployment**: Vercel with telemetry

## Goals
- ✅ Make data analysis accessible to non-technical users
- ✅ Provide full visibility into analysis steps (code, intermediate results)
- ✅ Enable human control with edit-before-execute workflow
- ✅ Build reusable analysis templates
- ✅ Deliver fast, interactive experience with streaming

## Non-Goals
- Complex statistical modeling or advanced ML
- Real-time data streaming or large dataset optimization (focus on small-to-medium files)
- Comparison with other BI tools or deployment on user infrastructure
