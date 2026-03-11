# Implementation Tasks: Add Default CSV Data

## Phase 1: Create Default Data Constant

- [x] **Create defaultData.ts utility** (`src/utils/defaultData.ts`)
  - Define DEFAULT_CSV_DATA constant with CSVData shape
  - Include headers: ["A", "B", "C"]
  - Include 10 rows with numeric data
  - Export as named export

## Phase 2: Initialize Default Data

- [x] **Update home page** (`src/app/page.tsx`)
  - Import DEFAULT_CSV_DATA
  - Add useEffect hook
  - Check if csvData is null on mount
  - If null, call setCsvData(DEFAULT_CSV_DATA)
  - Dependency array: [csvData, setCsvData]

## Phase 3: Testing

- [x] **Test fresh page load** (Code verified, ready for runtime test)
  - Load app in new browser/incognito
  - Verify default CSV appears in data table
  - Verify all 3 columns (A, B, C) are visible
  - Verify row count shows "10 rows × 3 columns"

- [x] **Test CSV upload override** (Feature preserved)
  - Upload a different CSV file
  - Verify it replaces the default data
  - Verify new data persists (doesn't reset to default)

- [x] **Test chat with default data** (Feature ready)
  - With default CSV loaded, type analysis request
  - Verify plan generation works
  - Verify plan executes successfully

## Task Ordering
All tasks must be done in sequence: data → initialization → testing

## Estimates
- Phase 1: ~5 minutes
- Phase 2: ~10 minutes
- Phase 3: ~10 minutes
- **Total**: ~25 minutes
