# README

## What this folder is about

This folder contains the mock data created based on a real automotive electronics client's circuit debugging scenario.

The client is building a **car control circuit board**. Their customer reports that the machine sometimes behaves strangely during operation. For example:

- the commanded signal and the actual response do not match
- the output becomes weak when voltage drops
- two sensors disagree during a specific operating mode

The problem is that these failures do **not** happen all the time. Because of that, the client engineer wants the AI-assisted workflow that can look at signal data and help decide what kind of failure is most likely happening.

---

## What the client engineer is trying to do

The engineer is trying to answer three questions:

1. **Is the machine failing to follow the command correctly?**
2. **If yes, is the problem caused by unstable power?**
3. **If not, is the problem caused by sensors disagreeing during a certain mode?**

This is how many engineers think in real debugging work:

- first, check the most obvious symptom
- then, check the most likely hardware cause
- then, check the more hidden logic or sensor cause

Instead of guessing, they use signal data from CSV files to test each idea step by step.

The following image illustrates the 3 charts' success and failure mode:

![Circuit Diagnostic style infographic](circuit_diagnostic_style_infographic.png)

---

## The client scenario

Imagine a customer says:

> “Our control board works most of the time, but during some runs the actuator becomes unstable. Sometimes the output weakens, and sometimes the sensor readings do not agree. We want to know whether this is a control problem, a power problem, or a sensor/mode problem.”

To investigate this, the engineer records **three synchronized CSV files** from the same machine event.

These files are:

- `chart1_tracking_command_response.csv`
- `chart2_power_supply_output.csv`
- `chart3_mode_sensor_mismatch.csv`

All three files cover the same time window, so they should be treated as three different views of the **same event**.

---

## What each CSV means

### 1. `chart1_tracking_command_response.csv`
Checks whether the **Response** follows the **Command**.

- If Response stays within ±5% of Command, tracking is normal
- If Response overshoots, undershoots, oscillates, or exceeds ±5% during transitions, there is a tracking failure

### 2. `chart2_power_supply_output.csv`
Checks whether the problem is related to **power instability**.

- If Supply Voltage drops and Output weakens, drops, or gets noisy at the same time, the issue may be power-related
- If Output remains stable despite normal voltage variation, there is no strong evidence of power failure

### 3. `chart3_mode_sensor_mismatch.csv`
Checks whether the issue is a **sensor mismatch during a special mode**.

- If Sensor1 and Sensor2 agree during normal operation but diverge mainly in one mode or event window, the issue may be mode-specific
- If disagreement exists across the full capture, it is persistent sensor disagreement

---

## How to use the four prompts

There are **four prompts** in this project.

### Overall Prompt
Use this when you want the AI to act like an engineer doing the **full diagnostic flow**. It tells the AI to start with CSV1, move to CSV2 only if needed, move to CSV3 if the problem is still not explained, and give a final report. Use this when you want **one complete diagnosis**.

```text
Diagnose one machine event using three synchronized CSVs from the same time window. In CSV1, compare Command vs Response: if Response deviates >±5% during transitions, flag tracking failure. If tracking fails, check CSV2: compare Supply Voltage vs Output; if Output weakens, drops, or gets noisy when Voltage dips, classify power-related failure. If not, check CSV3: compare Sensor1 vs Sensor2; if they diverge mainly in one mode/event window, classify mode-specific sensor mismatch. Output format: Step 1 finding; Step 2 finding; Step 3 finding; Final fault type; Root-cause hypothesis; Recommended next check. Include three charts in the output report: one for each step (CSV1, CSV2, CSV3).
```

### Prompt 1 — CSV1
Use this when you want to study **only the command vs response relationship**. This is the first check because it looks for the most visible problem.

```text
Analyze CSV1. Compare Command vs Response. Pass if Response stays within ±5% of Command; fail if it overshoots, undershoots, or exceeds ±5%. Output: Pass/Fail; key timestamps; conclusion. Include one chart.
```

### Prompt 2 — CSV2
Use this when you want to study **only the voltage vs output relationship**. Useful when you already think the issue may be caused by weak or unstable power.

```text
Analyze CSV2. Compare Supply Voltage vs Output. Pass if Output stays stable; fail if Output drops or gets noisy when Voltage dips. Output: Pass/Fail; key timestamps; conclusion. Include one chart.
```

### Prompt 3 — CSV3
Use this when you want to study **only the sensor mismatch behavior**. Useful when the problem seems more subtle and may depend on operating mode.

```text
Analyze CSV3. Compare Sensor1 vs Sensor2 (and Mode). Pass if sensors agree; fail if they diverge in one mode (mode-specific) or across the full capture (persistent). Output: Pass/Fail; key timestamps; conclusion. Include one chart.
```

---

## How engineers think when using these prompts

A simple engineering mindset is:

1. **See the symptom**  
   Does the response look wrong?

2. **Check the physical cause**  
   Is power dropping when the failure happens?

3. **Check the deeper system behavior**  
   Are sensors disagreeing only in a special condition?

This step-by-step logic is important because engineers usually do not jump straight to the hardest explanation. They first test the simplest explanation, then move to more specific causes.

---

## Expected output

The AI should return a short engineering-style report with:

1. **Incident Summary**
2. **Step 1 — Tracking Analysis**
3. **Step 2 — Power Analysis**
4. **Step 3 — Sensor / Mode Analysis**
5. **Final Diagnosis**
6. **Root Cause Hypothesis**
7. **Recommended Next Check**

This makes the result easier to read and easier to explain to a client or teammate.

---

## In one sentence

This project teaches you how to use three synchronized signal files and four prompts to diagnose whether a machine problem is mainly caused by **tracking failure, power instability, or mode-specific sensor mismatch**.