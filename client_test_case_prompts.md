# Client Test Case: 3-Step Signal Diagnosis

## Background

A client building an automotive control board is seeing intermittent field failures during operation. The observed issues include unstable actuator behavior, occasional weak output, and inconsistent sensor readings that do not appear in every run.

This test case is designed to support an AI-assisted triage workflow that reads three synchronized CSV signal captures from the same machine event and classifies the issue into one of three common fault categories:

1. **Tracking failure**
2. **Power-related failure**
3. **Mode-specific sensor mismatch**

The goal is to reduce debugging time and help engineers decide whether the next step should focus on control tuning, power integrity, or sensor / logic investigation.

---

## Reference CSV Files

Use these three CSV files as the synchronized inputs for the test case:

- `chart1_tracking_command_response.csv`
- `chart2_power_supply_output.csv`
- `chart3_mode_sensor_mismatch.csv`

These files represent the same machine over the same event window.

---

## Overall Prompt

```text
Diagnose one machine event using three synchronized CSVs from the same time window. In CSV1, compare Command vs Response: if Response deviates >±5% during transitions, flag tracking failure. If tracking fails, check CSV2: compare Supply Voltage vs Output; if Output weakens, drops, or gets noisy when Voltage dips, classify power-related failure. If not, check CSV3: compare Sensor1 vs Sensor2; if they diverge mainly in one mode/event window, classify mode-specific sensor mismatch. Output format: Step 1 finding; Step 2 finding; Step 3 finding; Final fault type; Root-cause hypothesis; Recommended next check.
```

---

## Prompt 1 — CSV1

```text
Analyze CSV1 only. Compare Command and Response over time. Check whether Response stays within ±5% of Command, especially during command transitions. If yes, classify normal tracking. If Response overshoots, undershoots, oscillates, or repeatedly exceeds ±5%, classify tracking failure. Output format: CSV used; Columns used; Expected relationship; Observed relationship; Pass/Fail; Key failure timestamps; Short conclusion.
```

---

## Prompt 2 — CSV2

```text
Analyze CSV2 only. Compare Supply Voltage and Output over time. Check whether Output drops, weakens, or becomes noisy when Supply Voltage dips. If degradation clearly aligns with voltage drop, classify power-related failure. If Output remains stable despite normal voltage variation, classify no strong evidence of power failure. Output format: CSV used; Columns used; Expected relationship; Observed relationship; Pass/Fail; Key correlated timestamps; Short conclusion.
```

---

## Prompt 3 — CSV3

```text
Analyze CSV3 only. Compare Sensor1 and Sensor2 over time, plus Mode if available. Check whether the sensors agree during normal operation and diverge mainly during one mode or event window. If divergence is mode-specific, classify mode-specific sensor mismatch. If disagreement exists across the full capture, classify persistent sensor disagreement. Output format: CSV used; Columns used; Expected relationship; Observed relationship; Pass/Fail; Key mismatch timestamps; Short conclusion.
```

---

## Expected Report Format

The AI agent should return a structured engineering report using this format:

1. **Incident Summary**
2. **Step 1 — Tracking Analysis**
3. **Step 2 — Power Analysis**
4. **Step 3 — Sensor / Mode Analysis**
5. **Final Diagnosis**
6. **Root Cause Hypothesis**
7. **Recommended Next Check**

---

## Notes for Test Use

- All three CSVs are synchronized views of the same event.
- The agent should treat them as one connected diagnostic workflow, not three unrelated files.
- The agent should provide a conclusion after each step before moving to the next step.
- If the data is mixed or unclear, the agent should explicitly state that the result is inconclusive.
