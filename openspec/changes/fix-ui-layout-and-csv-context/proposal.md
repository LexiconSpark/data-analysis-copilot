# Fix UI Layout & CSV Context Management

## Issues to Fix

### 1. CSV Context & Upload
**Current Issue**: Default CSV is shown in bottom-left corner, but should be used as invisible context
**Solution**:
- Load default CSV as context (already done ✓)
- Hide the data table by default
- Add "Upload CSV" button in chat interface for users who want to provide their own data
- When CSV is uploaded, use it as context instead of default
- Show brief preview of loaded CSV (filename + row count)

### 2. Tab Layout in Top-Right
**Current Issue**: 4 tabs (Templates, Plan, Code, Results) taking up space
**Solution**:
- Keep only 3 tabs: **Templates | Plan | Code**
- Move **Results** to bottom-right corner (becomes the main focus after execution)
- Plan tab shows generated analysis steps
- Code tab shows execution code
- Results panel becomes full bottom-right quadrant

## Updated Layout
```
┌─────────────────────┬──────────────────────────┐
│                     │   Templates|Plan |Code   │
│  Chat + Upload      │   Plan/Code content      │
│                     │   (tabs switch content)  │
├─────────────────────┼──────────────────────────┤
│  CSV Preview        │  Analysis Results        │
│  (collapsible)      │  (chart + conclusions)   │
└─────────────────────┴──────────────────────────┘
```

## User Flow
1. App loads → default CSV loaded as invisible context
2. User types analysis request
3. Plan tab auto-activates, shows steps
4. User approves plan
5. Code tab shows execution code
6. Results appear in bottom-right (charts + conclusions)
7. Optional: User can upload custom CSV to override default context

## Implementation Focus
- Update TabsContainer to show only 3 tabs
- Move Results display to bottom-right quadrant
- Add CSV upload button to ChatPanel
- Show CSV preview/status in ChatPanel
- Keep default CSV as context (no UI display unless explicitly opened)
