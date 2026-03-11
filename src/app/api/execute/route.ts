import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const { plan, stepIndex, csvData } = await request.json();

    if (!plan || stepIndex === undefined || !csvData) {
      return NextResponse.json(
        { error: "Plan, stepIndex, and CSV data are required" },
        { status: 400 }
      );
    }

    const step = plan.steps[stepIndex];
    if (!step) {
      return NextResponse.json(
        { error: "Invalid step index" },
        { status: 400 }
      );
    }

    // For now, return a mock execution result
    // In production, this would execute Python code in a sandbox
    let output = `Executed step ${step.step}: ${step.description}\n\nCode:\n${step.pythonCode}\n\nOutput: Success`;
    let error = null;
    let chartSpec: any = null;

    // Simulate different results for demonstration
    if (stepIndex === 0) {
      output = `Data loaded successfully. ${csvData.rows.length} rows, ${csvData.headers.length} columns.`;
    } else if (stepIndex === 1) {
      chartSpec = {
        type: "scatter",
        data: csvData.rows.slice(0, 10),
        title: "Data Visualization",
        xAxis: csvData.headers[0],
        yAxis: csvData.headers[1] || csvData.headers[0],
      };
      output = "Chart generated successfully";
    }

    const mockOutput = {
      stepIndex,
      output,
      error,
      chartSpec,
    };

    return NextResponse.json(mockOutput);
  } catch (error) {
    console.error("Execute API error:", error);
    return NextResponse.json(
      { error: "Failed to execute step" },
      { status: 500 }
    );
  }
}
