# Design: Fix UI Layout & CSV Context

## Architecture Changes

### 1. CSV Context Management
**Before**: Default CSV shown in table, user uploads to replace
**After**: Default CSV loaded silently as context, upload option available in chat

#### Data Flow
- App loads → `useEffect` loads DEFAULT_CSV_DATA into context (invisible)
- ChatPanel shows "CSV loaded" status
- User can click "Upload CSV" to override with custom data
- Selected CSV is used for all LLM calls (context)

### 2. Tab Structure
**Remove**: Results tab from tabs (move to quadrant 4)
**Keep**: Templates, Plan, Code tabs only

#### Tab Behavior
- **Templates**: List saved analysis plans
- **Plan**: Generated steps (auto-activates after request)
  - Shows numbered steps with editable descriptions
  - "Approve Plan" button at bottom
- **Code**: Generated code (auto-activates after approval)
  - Shows Python code for each step
  - Shows execution output/errors

### 3. Layout Grid Changes
```
Top-left (Chat):
  - Message history
  - User input field
  - "Upload CSV" button
  - Status: "CSV loaded (10 rows × 3 cols)" or "Custom CSV: filename.csv"

Top-right (Tabs):
  - Tab navigation: Templates | Plan | Code (3 tabs only)
  - Tab content area
  - Action buttons (Approve Plan, Save as Template)

Bottom-left (CSV Preview):
  - Collapsible/hidden by default
  - Show only if user explicitly wants to see data
  - Can be toggled from chat area

Bottom-right (Results):
  - Full quadrant dedicated to analysis results
  - Chart display (Recharts)
  - Conclusions (bullet points)
  - Download button
  - Auto-populated after execution
```

## Component Changes

### ChatPanel Component
**Add**:
- CSV status display ("Loaded: default" or "Loaded: custom_file.csv")
- "Upload CSV" button → triggers file input
- File input (hidden, accept .csv)
- onCsvUpload callback to parent

**Remove**:
- DataUploadArea integration (move elsewhere or make optional)

### TabsContainer Component
**Change**:
- Only render 3 tabs: Templates, Plan, Code
- Remove Results tab

### New CSVStatusPanel Component
- Shows current CSV info (name, row count, columns)
- "Change CSV" button
- Collapsible/expandable

### Updated Layout in page.tsx
```
<GridLayout
  topLeft={<ChatPanel csvData={csvData} onUploadCSV={setCsvData} />}
  topRight={<TabsContainer ... />}  // No Results tab
  bottomLeft={<CSVStatusPanel csvData={csvData} />}  // New component
  bottomRight={<ResultsPanel ... />}  // Now full quadrant
/>
```

## State Management
- AppContext: `csvData` tracks current CSV (default or uploaded)
- `setCsvData()` updates context with new CSV
- No UI clutter — CSV is silent context unless viewed

## No Breaking Changes
- All existing workflows preserved
- Default data loads automatically
- Upload still works
- Results still display (just different location)
