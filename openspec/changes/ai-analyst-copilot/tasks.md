# Implementation Tasks: AI Data Analyst Co-Pilot

## Setup & Infrastructure
- [x] **Create project structure**
  - Scaffolding: folders for components, hooks, utils, types
  - Create src/app directory (Next.js App Router structure)
  - Add environment variables file: `.env.local` with OPENAI_API_KEY

- [x] **Install & configure dependencies**
  - Verify Tailwind CSS 4 setup in next.config.ts
  - Verify TypeScript strict mode in tsconfig.json
  - Test build: `npm run build` passes with no errors

## Phase 1: Core UI Layout & Components

### 1.1 Layout Grid Component
- [x] **Create GridLayout component** (`src/components/GridLayout.tsx`)
  - 2x2 grid with 4 fixed quadrants
  - CSS Grid with borders, responsive sizing
  - Props: children (4 quadrants)
  - Default heights: top=60%, bottom=40%; left=40%, right=60%

### 1.2 Top-Left: Chat Interface
- [x] **Create ChatPanel component** (`src/components/ChatPanel.tsx`)
  - Message input field (textarea) at bottom
  - Message history area (scrollable)
  - File upload button + drag-drop zone
  - Show CSV preview after upload (first 5 rows)
  - Integrate Vercel AI SDK `useChat` hook
  - Send message with CSV context to `/api/chat`

- [x] **Create DataUploadArea component** (`src/components/DataUploadArea.tsx`)
  - Drag-drop zone with dashed border
  - "Drag and drop file or click to browse" text
  - File input (accept .csv)
  - Parse CSV using Papa Parse
  - Store in parent state: `{headers: [...], rows: [...]}`
  - Show file name after upload

### 1.3 Top-Right: Tabs Container
- [x] **Create TabsContainer component** (`src/components/TabsContainer.tsx`)
  - Tabs: Templates | Plan | Code
  - Active tab highlighted
  - Content area below tabs
  - Props: activeTab, onTabChange, content for each tab

- [x] **Create TemplatesTab component** (`src/components/TemplatesTab.tsx`)
  - Load templates from localStorage (JSON: `{name, plan}`)
  - List templates with load button
  - "New Analysis" button → switch to Plan tab
  - Load template → populate Plan tab with steps

- [x] **Create PlanTab component** (`src/components/PlanTab.tsx`)
  - Display numbered steps (editable input fields)
  - Each step: `{step #, description text, python code block}`
  - Edit description and code in-place
  - "Approve Plan" button (bottom)
  - "Save as Template" button (bottom)
  - Send approval to backend, trigger execution

- [x] **Create CodeTab component** (`src/components/CodeTab.tsx`)
  - List of executed code blocks in order
  - Each block: `{step #, code, result/error output}`
  - Syntax highlighting (use highlight.js or Prism)
  - Auto-scroll to latest executed block during execution

### 1.4 Bottom-Left: CSV Data Table
- [x] **Create DataTableComponent** (`src/components/DataTable.tsx`)
  - Render table from CSV: headers as columns, rows as table rows
  - Cells are editable (click to edit, blur to save)
  - Fixed height, scrollable
  - Show row count: "X rows × Y columns"
  - Styled with Tailwind: borders, hover states

### 1.5 Bottom-Right: Analysis Results
- [x] **Create ResultsPanel component** (`src/components/ResultsPanel.tsx`)
  - Chart container (Recharts)
  - Chart title (from user query or LLM)
  - Download button (export chart as SVG/PNG)
  - Conclusions section: bullet points with insights
  - Loading state: "Generating analysis..."
  - Error state: show error message

- [x] **Create ChartRenderer component** (`src/components/ChartRenderer.tsx`)
  - Accept chart spec: `{type: 'scatter'|'line'|'bar', data, title, xAxis, yAxis}`
  - Render using Recharts (ResponsiveContainer)
  - Handle multiple data series
  - Fallback for unsupported chart types

## Phase 2: State Management & Hooks

### 2.1 App-Level State
- [x] **Create AppContext** (`src/context/AppContext.tsx`)
  - State: csvData, workflowState, currentTab, chatMessages, currentPlan, results
  - Actions: setCsvData, setWorkflowState, setActiveTab, etc.
  - Provider wraps entire app

- [x] **Create useAppContext hook** (exported from AppContext.tsx)
  - Access AppContext values
  - Type-safe context hook

### 2.2 Chat & Streaming
- [x] **Chat streaming implemented in page.tsx**
  - Uses Vercel AI SDK generateText for server-side calls
  - Sends messages to `/api/chat` with CSV context
  - Handles responses and integrates with workflow
  - No separate hook needed - chat is integrated into component

## Phase 3: Backend API Routes

### 3.1 Chat API Route
- [x] **Create `/api/chat` route** (`src/app/api/chat/route.ts`)
  - Accept: `{message, csvData}`
  - Call LLM to generate initial response (brief guidance)
  - If user says "analyze", call `/api/plan` internally
  - Return: StreamingTextResponse with streamed text

### 3.2 Plan API Route
- [x] **Create `/api/plan` route** (`src/app/api/plan/route.ts`)
  - Accept: `{userRequest, csvData}`
  - Call LLM with prompt: "Generate a numbered step-by-step plan for: [userRequest]. Available data: [csvData]"
  - LLM returns structured plan: `[{step, description, pythonCode}]`
  - Parse response into Plan array
  - Return: JSON `{plan, planText}`
  - Set workflow state: `{state: 'waiting_for_approval', plan}`

### 3.3 Execute API Route
- [x] **Create `/api/execute` route** (`src/app/api/execute/route.ts`)
  - Accept: `{plan, stepIndex, csvData}`
  - Execute `plan[stepIndex].pythonCode` in isolated Python environment
  - Make CSV available to Python: `import pandas as pd; df = pd.read_csv(...)`
  - Capture output and any charts (JSON spec)
  - Return: JSON `{stepIndex, output, error?, chartSpec?}`
  - On error: increment retryCount, prepare for Retry node

### 3.4 Workflow Orchestration Route (Optional)
- [ ] **Create `/api/workflow` route** (`src/app/api/workflow/route.ts`)
  - Accepts workflow state machine events: "plan", "approve", "execute_step", "save_template", "cancel"
  - Manages LangGraph workflow execution
  - Streams workflow events back to client
  - Stores session state (checkpointer)

## Phase 4: LangGraph Workflow Setup

### 4.1 State Schema
- [x] **Define workflow state** (`src/server/workflowTypes.ts`)
  - TypeScript interface for LangGraph state
  - Properties: csvData, userMessage, plan, currentStepIndex, executedSteps, retryCount, cancelled, finalResults

### 4.2 LangGraph Nodes
- [x] **Plan Node** (`src/server/nodes/planNode.ts`)
  - Input: userMessage, csvData
  - LLM prompt: Generate step-by-step plan
  - Output: plan array

- [ ] **Interrupt Node** (`src/server/nodes/interruptNode.ts`)
  - Signal to frontend: "waiting for approval"
  - Frontend sends approval → resume
  - Allow step editing in Plan tab

- [x] **Execute Node** (`src/server/nodes/executeNode.ts`)
  - Loop through plan steps
  - For each step: call LLM to generate/refine Python code
  - Execute code, capture output
  - Update executedSteps, increment stepIndex

- [ ] **Retry Node** (`src/server/nodes/retryNode.ts`)
  - If execute fails: increment retryCount
  - If retryCount < 3: revise code, retry execute
  - If retryCount >= 3: go to Plan Revision

- [ ] **Plan Revision Node** (`src/server/nodes/planRevisionNode.ts`)
  - LLM receives error context and failed step
  - Revise earlier steps or entire plan
  - Reset retryCount, return to Execute

- [x] **Report Node** (`src/server/nodes/reportNode.ts`)
  - Compile results into final visualizations
  - Generate Recharts chart spec from data
  - Generate narrative conclusions (bullet points)
  - Store in finalResults

### 4.3 Workflow Graph
- [x] **Assemble workflow graph** (`src/server/workflow.ts`)
  - Define edges: Plan → Interrupt, Approve → Execute, Execute → {Success→next step, Failure→Retry}
  - Retry → {Success→next step, Exhausted→PlanRevision}
  - PlanRevision → Execute
  - All nodes check cancelled flag
  - Final edge: last step → Report
  - Implemented as WorkflowOrchestrator class with state machine

### 4.4 Checkpointer & Session Management
- [x] **Setup session management** (`src/app/api/workflow/route.ts`)
  - In-memory session store with Map
  - Session ID generation and retrieval
  - Support for start, approve, update_plan, cancel, status actions

## Phase 5: Frontend Integration & Hooks

### 5.1 Workflow Integration Hook
- [x] **Create useWorkflow hook** (`src/hooks/useWorkflow.ts`)
  - Calls `/api/workflow` with start, approve, update_plan, cancel actions
  - Manages workflow state: sessionId, currentStep, plan, executedSteps, finalResults
  - Handles loading states and error messages
  - Integrates with AbortController for cancellation

### 5.2 Plan Approval & Edit
- [x] **Workflow hook includes plan approval**
  - `approvePlan()` sends approval to `/api/workflow`
  - `updatePlan()` handles edited steps from PlanTab
  - Trigger execution resume after plan update

### 5.3 Template Management
- [x] **AppContext already handles templates**
  - Save/load templates from localStorage
  - All operations via useAppContext hook
  - No separate hook needed

## Phase 6: Page & Layout Assembly

### 6.1 Root Layout
- [x] **Create app layout** (`src/app/layout.tsx`)
  - AppContext provider
  - Root styling (dark/light mode toggle if desired)
  - Metadata: title, description

### 6.2 Home Page
- [x] **Create home page** (`src/app/page.tsx`)
  - GridLayout component with 4 quadrants
  - Pass components: ChatPanel, TabsContainer, DataTable, ResultsPanel
  - Wire up state context

### 6.3 Styling & Polish
- [ ] **Tailwind configuration**
  - Theme colors, spacing, typography
  - Create utility classes for grid gaps, borders
  - Responsive tweaks (if any)

## Phase 7: Testing & Deployment

### 7.1 Manual Testing
- [ ] **Test CSV upload**
  - Upload sample CSV
  - Verify parse and preview

- [ ] **Test chat and plan generation**
  - Send analysis request
  - Verify plan generated and displayed

- [ ] **Test plan approval and execution**
  - Approve plan
  - Verify steps execute and results display

- [ ] **Test tab switching**
  - Verify Plan tab auto-activates on approval wait
  - Verify Code tab auto-activates on execution
  - Verify Results tab shows final output

- [ ] **Test error handling and retries**
  - Trigger code error
  - Verify retry logic and plan revision

- [ ] **Test templates**
  - Save plan as template
  - Load and re-execute template

### 7.2 Build & Deployment
- [ ] **Build verification**
  - `npm run build` passes with no errors
  - `npm run lint` passes

- [ ] **Deploy to Vercel**
  - Connect repo to Vercel
  - Set environment variables (OPENAI_API_KEY)
  - Deploy main branch

## Task Ordering & Dependencies

**Phase 1** (UI Layout) → **Phase 2** (State) → **Phase 3** (API) → **Phase 4** (Workflow) → **Phase 5** (Integration) → **Phase 6** (Assembly) → **Phase 7** (Testing)

Within each phase, tasks can be done in sequence as listed, with dependencies marked where needed (e.g., PlanTab depends on TabsContainer).

## Estimates
- Phase 1: ~4-6 hours
- Phase 2: ~1-2 hours
- Phase 3: ~2-3 hours
- Phase 4: ~4-5 hours
- Phase 5: ~2-3 hours
- Phase 6: ~1 hour
- Phase 7: ~2-3 hours

**Total**: ~16-23 hours of development time
