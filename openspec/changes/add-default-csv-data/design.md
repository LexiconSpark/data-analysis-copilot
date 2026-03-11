# Design: Add Default CSV Data

## Architecture

### Data Definition
Store default CSV as a constant in a utility file:
```typescript
// src/utils/defaultData.ts
export const DEFAULT_CSV_DATA = {
  headers: ["A", "B", "C"],
  rows: [
    { A: 10, B: 15, C: 5 },
    { A: 20, B: 25, C: 15 },
    { A: 30, B: 35, C: 25 },
    // ... 10 rows total
  ],
};
```

### App Initialization
On component mount in the Home page:
1. Check if CSV data is already loaded (from previous session)
2. If not, load default data into AppContext via setCsvData()
3. User can override by uploading a new file

### State Management
- AppContext already has `csvData` and `setCsvData()`
- No new state needed
- Integration is a simple 1-2 line change in page.tsx

### User Flow
1. User loads app
2. Default CSV appears in data table immediately
3. Chat interface becomes active
4. User can type an analysis request right away
5. Or: User uploads a different CSV to override default

## Implementation Details

### Files to Modify
- Create: `src/utils/defaultData.ts` — Define default CSV constant
- Modify: `src/app/page.tsx` — Load default data on mount

### Change in page.tsx
```typescript
import { DEFAULT_CSV_DATA } from '@/utils/defaultData';

export default function Home() {
  const { csvData, setCsvData } = useAppContext();

  // Load default data on first render
  useEffect(() => {
    if (!csvData) {
      setCsvData(DEFAULT_CSV_DATA);
    }
  }, [csvData, setCsvData]);

  // ... rest of component
}
```

### No Breaking Changes
- Existing CSV upload still works (overrides default)
- All workflows unchanged
- Pure addition, no modifications to core logic

## Testing
- Verify default data loads on fresh page load
- Verify upload still works (overrides default)
- Verify localStorage persists user-uploaded CSV (if implemented)
