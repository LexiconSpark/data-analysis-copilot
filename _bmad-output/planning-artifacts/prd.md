---
stepsCompleted: ['step-01-init', 'step-02-discovery', 'step-03-success', 'step-04-journeys', 'step-05-domain', 'step-06-innovation', 'step-07-project-type', 'step-08-scoping', 'step-09-functional', 'step-10-nonfunctional', 'step-11-polish']
inputDocuments:
  - _bmad-output/planning-artifacts/product-brief-data-analysis-copilot-2026-02-07.md
  - _bmad-output/brainstorming/brainstorming-session-2026-02-07.md
workflowType: 'prd'
briefCount: 1
researchCount: 0
brainstormingCount: 1
projectDocsCount: 0
classification:
  projectType: web_app
  domain: general
  complexity: low
  projectContext: greenfield
---

# Product Requirements Document - data-analysis-copilot

**Author:** Ti  
**Date:** 2026-02-08

## Executive Summary

Data-analysis-copilot is an internal, locally hosted web app that lets electrical engineers create circuit-analysis reports in ~15 minutes instead of hours. Users upload CSVs, describe the report in natural language, and get an AI-generated work plan (editable) and then a report with charts—without building Excel charts or writing code. Fallbacks: edit the plan or use a text prompt to debug/redo. Target users: 1–2 electrical engineers doing circuit debugging; MVP is a four-quadrant SPA (Chrome, real-time progress, one report template).

## Success Criteria

### User Success

- **Time-to-report:** Engineers achieve report creation in ~15 minutes instead of ~2 hours for a typical batch.
- **Charts without manual grind:** They can get to "the right chart" quickly; the system handles the mechanics (data → graph) while they stay in control of what's being analyzed and why.
- **Aha moment:** Using natural language to describe what they want (e.g. "create a report with charts for X vs Y") and getting a concrete plan and then a report with charts—without writing code or building Excel charts by hand.

### Business Success

- **Internal validation:** The product can generate a simple report with internal testers, running **reliably locally**.
- **Proof point:** "This is working" = internal testers successfully produce at least one end-to-end report (upload → natural language → report with charts) on their own machines.

### Technical Success

- **Reliability:** Runs reliably when hosted locally (no flaky runs for the core flow).
- **Execution model:** User triggers execution (e.g. click); agents run, produce the report, and the result is shown in the frontend.

### Measurable Outcomes

- Report creation time: target **≤ 15 minutes** for a typical batch (vs ~2 hours today).
- Capability: **Natural language → plan → report with charts** works for internal testers.
- Environment: **Local** (locally hosted web app or desktop); stable, repeatable runs.

## Product Scope

### MVP - Minimum Viable Product

- **CSV upload:** Users can upload one or more CSVs.
- **Natural language commands:** Users type what they want (e.g. "create a report with charts for X and Y"); the system interprets the intent.
- **Single report template:** One report template that defines structure and placeholders (e.g. title, sections, chart slots).
- **Execute and show result:** On user action (e.g. "Run" / "Generate" click), agents execute the plan, generate the report, and the **report is displayed in the frontend** (viewable in the same app).

**Outcome:** Internal testers complete one session: upload data → describe report in natural language → see generated report with charts in the UI.

### Growth Features (Post-MVP)

- Multiple report templates; reuse across batches.
- Richer natural language (e.g. "same as last time but for this CSV").
- Saving and reusing workflows/templates for repeated analysis.

### Vision (Future)

- Full multi-agent debugging flow (describe problem + debug approach → system suggests/executes analysis and report).
- AI-generated Python for custom analysis, integrated into the report pipeline.
- Deeper "understanding" of circuit/debugging context to reduce manual specification.

## User Journeys

### Journey 1: Primary User – Success Path (Sam)

**Opening scene:** Sam has just finished a circuit test run and has three CSVs (e.g. voltage, current, time). She used to spend hours in Excel building charts and writing up trends. She wants a report in minutes, not hours.

**Rising action:** She opens the locally hosted app, uploads the three CSVs, then types in natural language: *"Report with voltage vs time and current vs time, as well as analyzing the trends."* She reviews the request (and any plan/summary the system shows, if we add that). She hits **Run**.

**Climax:** The agents run, generate the analysis and charts, and assemble the report. The UI shows the report with two charts (voltage vs time, current vs time) and trend analysis. She can read and use it immediately.

**Resolution:** She gets a shareable or exportable report in ~15 minutes instead of ~2 hours. She feels the product "gets" what she asked for and delivers without manual chart-building.

### Journey 2: Primary User – Edge Case (Large Data)

**Opening scene:** Sam has a test run that produced very large CSVs (e.g. high sample rate, long capture). She uploads them and asks for the same kind of report (voltage vs time, current vs time, trend analysis).

**Rising action:** She uploads the files and hits Run. The system runs but struggles: graphs are slow, unreadable (e.g. too many points), or the run times out. The product doesn't handle the scale well.

**Climax:** The system surfaces the problem instead of failing silently: e.g. a message that data is large, that graphing may be slow or degraded, or a suggestion to downsample/sample/summarize. Sam sees what went wrong and what her options are.

**Resolution:** Sam can recover in one or more of these ways (to be decided in design): (a) the system automatically downsamples or summarizes for visualization and still produces a report; (b) the system suggests "use a subset" or "reduce rows" and she filters/subsets and retries; (c) she gets a clear warning up front so she can split the data or reduce it before running. Her expectation: the product doesn't leave her stuck when "the data is really big."

### Journey Requirements Summary

| Journey | Capabilities revealed |
|--------|------------------------|
| **1 – Success path** | CSV upload (multi-file); natural-language input; Run/Execute trigger; agent execution; report generation with specified charts and trend analysis; report display in frontend; optional plan/summary before Run. |
| **2 – Large data** | Handling large datasets (performance, timeouts); clear feedback when graphing fails or degrades; recovery path: automatic downsampling/summarization and/or user guidance (subset, filter, retry); possible upfront warning for very large uploads. |

## Innovation & Novel Patterns

### Detected Innovation Areas

- **AI-generated work plan:** The system produces the report work plan (what to analyze, which charts, in what order). That's the main novel piece—the AI drives the plan, not the user building it step by step.
- **User-editable plan when AI fails:** If the generated plan is wrong or incomplete, the user can **edit the plan themselves** (change steps, add/remove charts or analyses) instead of being stuck.
- **AI-generated code for plots and analysis:** The system **writes code** (e.g. Python) to create plots and run data analysis, so engineers don't write or maintain that code.
- **Iterative repair via natural language:** When generated code or results fail, the user can **use a text prompt** to debug or ask the AI to redo (e.g. "fix the time axis," "redo this chart with a moving average"). No need to edit code by hand—prompt-driven repair.

### Market Context & Competitive Landscape

- Internal tool for electrical engineers; no direct "market" yet. Differentiation vs Excel: AI-driven plan + code generation + NL repair, instead of manual charting and scripting.
- Similar ideas exist in generic "AI for data" tools, but the combination (work plan generation + code-for-plots/analysis + NL fallback for this workflow) is the specific innovation.

### Validation Approach

- Internal testers run real circuit-analysis batches; measure **time to report** (target ~15 min vs ~2 hours) and **success rate** (report with correct charts and trend analysis).
- Validate work-plan quality: do generated plans usually match what an engineer would do? Validate repair: when something fails, can users get to a good result via prompt (debug/redo) or by editing the plan?

### Risk Mitigation

- **Plan wrong or incomplete:** User can edit the plan; no hard dependency on perfect first-shot planning.
- **Generated code fails or is wrong:** User can prompt to debug or ask AI to redo; no requirement to write or fix code themselves.
- **AI reliability:** Fallbacks are explicit: edit plan, prompt to redo. Avoid "AI or nothing" for core flow.

## Web App Specific Requirements

### Project-Type Overview

- **Single-page application (SPA)** with a **four-quadrant layout** on one screen (e.g. upload area, natural-language input, plan/work view, report/output).
- **Internal tool**; no public deployment or SEO requirements.
- **Real-time UX:** Show **live progress** while the report is being generated (e.g. steps, progress bar, or streaming status).
- **Primary browser:** **Chrome**; no requirement to support other browsers for MVP.
- **Accessibility:** **Basic usability** for internal engineers (readable, operable, no formal WCAG target).

### Technical Architecture Considerations

- **SPA:** One shell; state and navigation handled in-app (no full page reloads for upload → plan → run → report).
- **Four-quadrant layout:** UI and state design must keep the four areas clear and responsive to content (e.g. plan updates, report loading).
- **Real-time progress:** Backend or agent layer exposes progress (e.g. via polling or server-sent events); frontend subscribes and updates the UI so the user sees live status.
- **Local hosting:** Runs on internal network or localhost; no CDN or public-edge requirements.

### Browser Matrix

| Browser | Support level | Notes |
|--------|----------------|--------|
| Chrome (evergreen) | Primary / supported | MVP target; internal use. |
| Other browsers | Not required for MVP | May be added later if needed. |

### Responsive Design

- **Desktop-first:** Optimized for typical engineer desktops/laptops.
- **Four quadrants** must remain usable at common resolutions (e.g. 1280×720 and up); reflow or stacking for smaller screens can be deferred unless needed for MVP.

### Performance Targets

- **Report generation:** Target end-to-end time consistent with success criteria (~15 min for a typical batch); progress feedback so the user knows the run is active.
- **Large data:** As per user journeys, handle large CSVs with clear feedback and recovery (e.g. progress, timeouts, downsampling or guidance).
- **UI responsiveness:** Progress and state updates feel real-time (no long frozen UI).

### SEO Strategy

- **Not applicable.** Internal tool; no public indexing or SEO requirements.

### Accessibility Level

- **Basic usability** for internal engineers: readable text, usable controls, keyboard navigation where it doesn't block the main flow. No formal WCAG level required for MVP.

### Implementation Considerations

- Implement and test the **four-quadrant layout** early (structure, resizing, overflow).
- Implement a **progress/status channel** (e.g. API + polling or SSE) and wire it to the UI so "Run" shows live progress.
- Chrome-only is acceptable for MVP; document any Chrome-specific assumptions if relevant (e.g. storage, workers).

## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**MVP Approach:** Problem-solving MVP—smallest set that delivers "I got a report in ~15 minutes without Excel." Validation = internal testers running real batches and succeeding end-to-end.

**Resource Context:** Small team (1–2 electrical engineers as primary users); MVP should be buildable and maintainable with limited capacity. No separate admin/support user types for MVP.

### MVP Feature Set (Phase 1)

**Core User Journeys Supported:**
- Journey 1 (success path): Upload CSVs → enter NL request (e.g. voltage/current vs time + trends) → Run → see report with charts in the UI.
- Journey 2 (large data): Same flow with large CSVs; user gets clear feedback and a recovery path (e.g. downsampling, guidance, or retry).

**Must-Have Capabilities:**
- Four-quadrant SPA (e.g. upload, NL input, plan/work area, report view).
- CSV upload (one or more files).
- Natural-language input and AI-generated work plan (user can edit plan).
- Single report template; Run/Execute → agents produce report and display in frontend.
- Live progress during report generation.
- Fallbacks: edit plan if wrong; use text prompt to debug/redo when code or output fails.
- Chrome support; local hosting; basic usability (no formal a11y target for MVP).

**Out of Scope for MVP:** Multiple templates, workflow save/reuse, full multi-agent debugging flow, AI-generated Python in pipeline, other browsers.

### Post-MVP Features

**Phase 2 (Growth):**
- Multiple report templates; reuse across batches.
- Richer natural language (e.g. "same as last time but for this CSV").
- Save and reuse workflows/templates for repeated analysis.
- Optional: better large-data handling (e.g. automatic downsampling, limits).

**Phase 3 (Expansion):**
- Full multi-agent debugging flow (describe problem + debug approach → system suggests/executes analysis and report).
- AI-generated Python for custom analysis, integrated into report pipeline.
- Deeper circuit/debugging context to reduce manual specification.

### Risk Mitigation Strategy

**Technical:** Mitigate agent and code-gen reliability with editable plans and prompt-to-debug/redo; avoid "AI or nothing." Large data: progress feedback, timeouts, and at least one recovery path (e.g. downsampling or user guidance) in MVP.

**Market/Validation:** Internal testers and time-to-report (~15 min) plus success rate prove value; no external market risk for MVP.

**Resource:** MVP scoped to one primary user type and one template; phased roadmap so growth/vision can be deferred if capacity is tight.

## Functional Requirements

### Data Input & Upload

- **FR1:** User can upload one or more CSV files to the application.
- **FR2:** User can remove or replace uploaded files before running report generation.
- **FR3:** System can accept and use multiple CSVs as input for a single report run.
- **FR4:** User can see which files are currently loaded and associated with the session.

### Natural Language & Work Plan

- **FR5:** User can describe the desired report in natural language (e.g. which charts, what analyses, trends).
- **FR6:** System can generate a work plan (steps, charts, analyses) from the user's natural-language request.
- **FR7:** User can view the generated work plan before execution.
- **FR8:** User can edit the generated work plan (add, remove, or change steps or chart/analysis specs) before running.
- **FR9:** System can interpret natural-language requests in the context of the uploaded data (e.g. column names, structure).
- **FR10:** User can re-enter or change the natural-language request and get a new or updated work plan.

### Execution & Progress

- **FR11:** User can trigger report generation (e.g. Run/Execute) using the current plan and uploaded data.
- **FR12:** System can execute the work plan (perform analysis, generate charts, assemble report).
- **FR13:** User can see live progress during report generation (e.g. current step or phase, status).
- **FR14:** System can provide progress or status updates during execution so the user knows the run is active and where it is.

### Report Output & Display

- **FR15:** User can view the generated report within the application (e.g. in a dedicated area or quadrant).
- **FR16:** Report can include charts/visualizations as specified in the work plan.
- **FR17:** Report can include trend analysis or text summaries as specified in the plan.
- **FR18:** System can render the report (charts and text) in the UI so the user can read and use it without leaving the app.
- **FR19:** User can see the report in the same session as the upload, plan, and run (no separate export step required for basic use).

### Recovery & Iteration

- **FR20:** User can correct or refine the work plan when the generated plan is wrong or incomplete.
- **FR21:** When generated code or output fails or is wrong, user can describe the issue or desired fix in natural language (e.g. prompt to debug or redo).
- **FR22:** System can re-run or adjust execution based on user feedback (e.g. redo with modified plan or with a natural-language correction).
- **FR23:** User can re-run report generation after editing the plan or after providing feedback, without re-uploading data if it is still valid.

### Large Data Handling

- **FR24:** When data size causes slow execution, failed graphing, or degraded results, system can inform the user (e.g. message, warning, or status).
- **FR25:** User can recover from large-data issues (e.g. via guidance to subset, retry, or system-provided downsampling/summarization) so a report can still be produced or attempted.
- **FR26:** System can offer at least one recovery path when large data prevents successful report generation (e.g. automatic downsampling, suggestion to reduce data, or clear error with next steps).

### Application Structure & Layout

- **FR27:** User can access core flows—upload, natural-language input, work plan, and report view—from a single application view (e.g. four-quadrant layout).
- **FR28:** User can see the relationship between uploaded data, current work plan, and generated report in the same session without switching applications.
- **FR29:** User can interact with the application in a single-page flow (no full page reload required for upload → plan → run → report).

## Non-Functional Requirements

### Performance

- **NFR-P1:** End-to-end report generation for a typical batch (upload → plan → run → report displayed) completes within the target window (e.g. ~15 minutes) under normal data sizes so the success criterion is achievable.
- **NFR-P2:** Progress or status updates are shown during report generation so the user sees the run is active; the UI does not appear frozen for the whole run.
- **NFR-P3:** The UI remains responsive during normal use (e.g. upload, editing plan, viewing report); user actions receive feedback within a few seconds where applicable.

### Reliability

- **NFR-R1:** The application runs reliably when hosted locally (no persistent flakiness for the core flow: upload → NL → plan → run → report).
- **NFR-R2:** When execution fails (e.g. agent or code error), the system surfaces a clear indication of failure so the user can use fallbacks (edit plan, prompt to redo).

### Security

- **NFR-S1:** For MVP, the product is intended for local or internal-network use; no requirement to transmit user data to external services. Any future cloud or external processing would introduce additional security requirements.

### Accessibility

- **NFR-A1:** The application meets basic usability for internal engineers (readable text, operable controls, sufficient contrast and layout). No formal WCAG level or compliance target for MVP.
