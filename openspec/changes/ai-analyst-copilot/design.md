# Design: AI Data Analyst Co-Pilot

## Architecture Overview

### Frontend Architecture
- **Layout**: 2x2 fixed-height grid with visible borders
  - Top-left: Chat interface (streaming via useChat)
  - Top-right: Tabs container (Templates/Plan/Code)
  - Bottom-left: CSV data table (editable, with upload)
  - Bottom-right: Analysis results (charts via Recharts)

### Backend Architecture
- **API Routes** (Next.js App Router)
  - `POST /api/chat` — receives user message + CSV context, streams chat response
  - `POST /api/plan` — receives analysis request, returns step-by-step plan
  - `POST /api/execute` — receives step, executes code, returns result
  - All endpoints integrate with LangGraph workflow

- **Streaming**: Vercel AI SDK `useChat` hook on frontend, `StreamingTextResponse` on backend
- **State Management**: Combination of React state (UI state) + LangGraph checkpointer (workflow state)

### Agentic Workflow (LangGraph.js)
State schema includes:
```typescript
{
  csvData: string,           // CSV context passed to every node
  userMessage: string,       // Original user request
  plan: Step[],              // Generated analysis plan
  currentStepIndex: number,  // Which step we're on
  executedSteps: StepResult[],
  retryCount: number,        // Per-step retry counter (0-3)
  cancelled: boolean,        // Cancellation flag, checked at every node
  finalResults: any,         // Visualizations and conclusions
}
```

**Nodes**:
1. **Plan Node** — Takes user message + CSV, uses LLM to generate step-by-step plan
   - Output: `plan` (array of {description, pythonCode} objects)
   - Next: Interrupt (pause for approval)

2. **Interrupt Node** — Pauses execution, waits for user approval
   - User can edit individual steps in the Plan tab
   - Frontend sends "Approve Plan" → resume
   - Front-end activates Plan tab automatically when waiting for approval

3. **Execute Node** — For each step, generate/refine Python code and run it
   - LLM has CSV data in context
   - Executes code in isolated environment
   - Returns output (data/chart JSON)
   - Code tab shows execution blocks in order, always visible
   - Next: Check result → Success → Next step, or Failure → Retry

4. **Retry Node** — On failure, LLM rewrites code (up to 3 retries)
   - Each retry updates retryCount
   - After 3 failures: Go to Plan Revision node
   - Cancellation check here

5. **Plan Revision Node** — If all retries exhausted, revise earlier steps
   - LLM receives error context, may suggest different approach
   - Updates plan, retryCount = 0, goto Execute

6. **Report Node** — Compile results into visualizations + conclusions
   - Renders charts using Recharts
   - Generates narrative bullet-point conclusions
   - Stores in finalResults

7. **Cancellation Check** — At every node, if `cancelled=true`, halt and return partial results

### Client-Server Flow
1. User uploads CSV → stored in React state, sent to all API calls as context
2. User types analysis request → POST /api/chat with CSV + message
3. LangGraph workflow executes, streaming results via Vercel AI SDK
4. Frontend receives streamed events: "plan_generated", "step_executing", "step_complete", "results_ready"
5. UI state updates: Plan tab activates on pause, Code tab activates when running, Results tab shows final output

## UI Components

### Layout Grid (Fixed Height, Visible Borders)
```
┌─────────────────────┬──────────────────────────┐
│                     │    Templates | Plan|Code │
│  Chat Interface     │  Plan/Code/Results Tabs  │
│                     │                          │
├─────────────────────┼──────────────────────────┤
│ CSV Data Table      │ Analysis Results         │
│ (Editable, Upload)  │ (Charts, Conclusions)   │
└─────────────────────┴──────────────────────────┘
```

### Top-Left: Chat Interface
- Message input area
- Scrollable message history
- Show file upload button and drag-drop zone
- Display CSV preview after upload

### Top-Right: Tabs Panel
- **Templates Tab** (default on load)
  - List of saved plan templates
  - New analysis button to switch to Plan tab

- **Plan Tab** (auto-activates when workflow pauses)
  - Numbered step list, editable text fields
  - "Approve Plan" button (green) at bottom

- **Code Tab** (auto-activates after execution starts)
  - Blocks of executed code with syntax highlighting
  - Below each block: result output or error message
  - Always visible during and after execution

### Bottom-Left: CSV Data Table
- Columns based on CSV header
- Editable cells
- Scrollable, fixed height
- Upload button: "Drag and drop file or click to browse"
- After upload, show file name and option to re-upload

### Bottom-Right: Analysis Results
- Chart container (Recharts)
- Chart title derived from user request
- Download button (SVG/PNG export)
- Conclusions section: bullet points with insights

## Data Flow

### CSV Handling
- Papa Parse reads CSV on upload
- Stored as string in state: `{headers: [...], rows: [...]}`
- Passed to every LLM call as context window
- Editable in the data table component

### Plan Generation
- User request + CSV context → LLM → structured plan output
- LLM returns: `[{step: 1, description: "...", pythonCode: "..."}, ...]`

### Code Execution
- Each step's pythonCode executed in isolated Python environment
- Environment has CSV data available as pandas DataFrame
- Returns: `{stepIndex, output, error?, chart?}`

### Visualization
- Chart JSON from execute node → Recharts component
- Supports: LineChart, ScatterChart, BarChart, ComposedChart
- Auto-scales based on data
- Title from user query or LLM-generated title

### Templates (localStorage)
- Save current plan with name: `localStorage.setItem('templates', JSON.stringify([...]))`
- Load template: reconstruct plan in state, user can edit and re-execute
- "Save as Template" button on Plan tab

## Key Technical Decisions

1. **Streaming over Polling**: Vercel AI SDK for real-time updates without chat-like UI
2. **LangGraph Checkpointer**: Per-session persistence enables interrupt/resume
3. **State in Root**: CSV and workflow state in React root, passed down as context
4. **Nodes with Cancellation**: Every node checks `cancelled` flag before proceeding
5. **Code Execution Isolation**: Python subprocess per step, access to CSV only
6. **Tab Auto-Activation**: UI mode switching based on workflow state:
   - Plan tab on Interrupt node
   - Code tab on Execute node
   - Results panel after Report node

## Telemetry & Observability
- Vercel AI SDK telemetry logs: model calls, latency, token usage
- LangGraph node execution times logged
- User interactions tracked: chat sends, plan approvals, edits
