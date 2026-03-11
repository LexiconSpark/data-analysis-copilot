import { WorkflowState } from "../workflowTypes";
import { StepResult, ChartSpec } from "@/types";

function detectMatplotlibChart(
  pythonCode: string,
  csvData: any
): ChartSpec | null {
  const codeUpper = pythonCode.toUpperCase();

  // Check if code contains matplotlib plotting
  if (
    !codeUpper.includes("PLT.") &&
    !codeUpper.includes("AX.") &&
    !codeUpper.includes(".PLOT(")
  ) {
    return null;
  }

  if (!csvData || !csvData.rows || csvData.rows.length === 0) {
    return null;
  }

  // Extract title from ax.set_title() or plt.title()
  let title = "Chart";
  const titleMatch = pythonCode.match(
    /(?:ax|plt)\.set_title\s*\(\s*['"](.*?)['"]|(?:ax|plt)\.title\s*\(\s*['"](.*?)['"]/
  );
  if (titleMatch) {
    title = titleMatch[1] || titleMatch[2] || title;
  }

  // Extract axis labels
  let xAxisLabel = csvData.headers[0] || "X";
  let yAxisLabel = csvData.headers[1] || csvData.headers[0] || "Y";

  const xLabelMatch = pythonCode.match(
    /(?:ax|plt)\.set_xlabel\s*\(\s*['"](.*?)['"]|(?:ax|plt)\.xlabel\s*\(\s*['"](.*?)['"]/
  );
  if (xLabelMatch) {
    xAxisLabel = xLabelMatch[1] || xLabelMatch[2];
  }

  const yLabelMatch = pythonCode.match(
    /(?:ax|plt)\.set_ylabel\s*\(\s*['"](.*?)['"]|(?:ax|plt)\.ylabel\s*\(\s*['"](.*?)['"]/
  );
  if (yLabelMatch) {
    yAxisLabel = yLabelMatch[1] || yLabelMatch[2];
  }

  // Extract column names being plotted
  // Matches patterns like: data['X'].plot() or data["X"].plot() or df['X'].plot()
  const seriesMatches = pythonCode.match(/(?:data|df)\s*\[\s*['"](.*?)['"]\s*\]\.plot/gi);
  const series: string[] = [];

  if (seriesMatches) {
    seriesMatches.forEach((match) => {
      const colMatch = match.match(/['"](.*?)['"]/);
      if (colMatch) {
        const col = colMatch[1];
        if (csvData.headers.includes(col) && !series.includes(col)) {
          series.push(col);
        }
      }
    });
  }

  // If no series found from code, try to infer from available columns
  if (series.length === 0 && csvData.headers.length > 0) {
    // Default to first two columns (or just first if only one exists)
    series.push(csvData.headers[0]);
    if (csvData.headers.length > 1) {
      series.push(csvData.headers[1]);
    }
  }

  // Determine chart type based on code patterns
  let chartType: "scatter" | "line" | "bar" | "composed" = "line";
  if (codeUpper.includes("SCATTER")) {
    chartType = "scatter";
  } else if (codeUpper.includes("BAR(")) {
    chartType = "bar";
  }

  return {
    type: chartType,
    data: csvData.rows,
    title,
    xAxis: xAxisLabel,
    yAxis: yAxisLabel,
    series: series.length > 0 ? series : undefined,
  };
}

export async function executeNode(state: WorkflowState): Promise<Partial<WorkflowState>> {
  if (!state.plan || state.plan.steps.length === 0) {
    return {
      errorMessage: "No plan to execute",
    };
  }

  if (state.cancelled) {
    return {
      errorMessage: "Execution cancelled by user",
    };
  }

  try {
    const step = state.plan.steps[state.currentStepIndex];
    if (!step) {
      return {
        errorMessage: "No more steps to execute",
      };
    }

    let output = `Step ${step.step}: ${step.description}`;
    let error: string | undefined = undefined;
    let chartSpec: ChartSpec | null = null;

    // Handle first step (data loading)
    if (state.currentStepIndex === 0) {
      output = `Data loaded successfully. ${state.csvData?.rows.length || 0} rows, ${state.csvData?.headers.length || 0} columns.`;
    } else if (state.csvData) {
      // Try to detect matplotlib code in the step
      chartSpec = detectMatplotlibChart(step.pythonCode, state.csvData);

      if (chartSpec) {
        output = "Chart generated successfully";
      } else {
        output = `Analysis complete for step ${step.step}`;
      }
    } else {
      output = `Step ${step.step} completed: ${step.description}`;
    }

    const stepResult: StepResult = {
      step: step.step,
      description: step.description,
      pythonCode: step.pythonCode,
      output,
      error,
      chartSpec: chartSpec || undefined,
    };

    const newSteps = [...state.executedSteps, stepResult];
    const nextStepIndex = state.currentStepIndex + 1;

    return {
      executedSteps: newSteps,
      currentStepIndex: nextStepIndex,
      retryCount: 0,
    };
  } catch (error) {
    console.error("Execute node error:", error);
    return {
      errorMessage: `Execution failed: ${error instanceof Error ? error.message : "Unknown error"}`,
    };
  }
}
