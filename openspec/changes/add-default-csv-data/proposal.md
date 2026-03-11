# Add Default CSV Data

## What
Add built-in default CSV data that loads automatically when the app initializes, allowing users to immediately test analysis without uploading a file.

## Why
- **Faster onboarding**: Users can start experimenting immediately without needing to prepare a CSV
- **Demo-ready**: Perfect for showcasing the app's capabilities
- **Testing convenience**: Built-in test data for development and QA
- **Better UX**: Users see a preview table immediately, reducing friction

## Default Data
Three columns (A, B, C) with 10 rows each:
```
A: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
B: [15, 25, 35, 45, 55, 65, 75, 85, 95, 105]
C: [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
```

This data:
- Shows clear linear relationships (A increasing by 10, B by 10, C by 10)
- Is small enough to run quickly
- Is realistic for demo analysis workflows
- Demonstrates chart generation well

## Implementation
1. Define default data as a constant
2. Load into AppContext state on app initialization
3. Pre-fill CSV table on page load
4. User can still upload a different file to override

## Goals
- ✅ Reduce time to first analysis
- ✅ Enable immediate chart/visualization demo
- ✅ Improve first-time user experience
- ✅ Keep as optional (user can still upload custom CSV)

## Non-Goals
- Complex multi-sheet CSV support
- Data validation or cleaning
- CSV file generation/export
