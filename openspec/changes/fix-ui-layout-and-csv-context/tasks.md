# Implementation Tasks: Fix UI Layout & CSV Context

## Phase 1: Remove Results Tab from Tabs

- [x] **Update TabsContainer component** (`src/components/TabsContainer.tsx`)
  - Remove "Results" from tabs array
  - Keep only: Templates, Plan, Code
  - Update tab navigation to show 3 tabs

- [x] **Update page.tsx GridLayout**
  - Remove `results` prop from TabsContainer
  - Move ResultsPanel to bottomRight quadrant

## Phase 2: Create CSV Status Component

- [x] **Create CSVStatusPanel component** (`src/components/CSVStatusPanel.tsx`)
  - Display current CSV info (name/status, rows, columns)
  - Show "Loaded: default" or "Custom: filename.csv"
  - Optional "Change CSV" button
  - Simple collapsible/expandable design

- [x] **Add to GridLayout**
  - Place CSVStatusPanel in bottomLeft quadrant
  - Update page.tsx to pass csvData

## Phase 3: Enhance ChatPanel with CSV Upload

- [x] **Update ChatPanel component** (`src/components/ChatPanel.tsx`)
  - Add CSV upload section at top
  - Show CSV status ("CSV loaded: 10 rows × 3 cols")
  - "Upload CSV" button
  - Hidden file input
  - onUploadCSV callback when file selected
  - Remove DataUploadArea integration (keep only upload button)

## Phase 4: Update Page Layout

- [x] **Refactor page.tsx** (`src/app/page.tsx`)
  - Update GridLayout props:
    - topRight: TabsContainer with 3 tabs (no results)
    - bottomLeft: CSVStatusPanel (was DataTable)
    - bottomRight: ResultsPanel (was empty/secondary)
  - Verify flow: chat → plan → code → results

## Phase 5: Testing & Polish

- [x] **Test tab switching**
  - Plan tab shows generated steps
  - Code tab shows execution code
  - No Results tab in navigation

- [x] **Test CSV handling**
  - Default CSV loads invisibly on mount
  - Upload button in ChatPanel works
  - Custom CSV replaces default
  - Status reflects current CSV in both ChatPanel and CSVStatusPanel

- [x] **Test results display**
  - Charts appear in bottom-right ResultsPanel
  - Conclusions display correctly
  - Full quadrant utilization

- [x] **Verify responsive layout**
  - All 4 quadrants visible
  - Proper sizing and proportions
  - No overflow or hidden content

## Phase 6: Fix Chart Generation from Matplotlib Code

- [x] **Enhanced executeNode to parse matplotlib code**
  - Detects matplotlib patterns (plt., ax., .plot() calls)
  - Extracts chart title from ax.set_title() or plt.title()
  - Parses axis labels from set_xlabel/ylabel() calls
  - Extracts column names from data['X'].plot() patterns
  - Infers chart type (scatter, line, bar) from code
  - Generates proper ChartSpec using actual CSV data
  - Applies to any step with matplotlib code, not just step 1

## Task Ordering
Phases must be sequential: 1 → 2 → 3 → 4 → 5

## Estimates
- Phase 1: ~10 minutes (remove tab)
- Phase 2: ~15 minutes (new component)
- Phase 3: ~15 minutes (enhance chat panel)
- Phase 4: ~10 minutes (refactor layout)
- Phase 5: ~20 minutes (testing)
- **Total**: ~70 minutes
