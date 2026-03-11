import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";
import { WorkflowState } from "../workflowTypes";

export async function reportNode(state: WorkflowState): Promise<Partial<WorkflowState>> {
  if (!state.executedSteps || state.executedSteps.length === 0) {
    return {
      errorMessage: "No steps executed to generate report",
    };
  }

  try {
    // Find the last chart spec if any
    let chartSpec = null;
    for (let i = state.executedSteps.length - 1; i >= 0; i--) {
      if (state.executedSteps[i].chartSpec) {
        chartSpec = state.executedSteps[i].chartSpec;
        break;
      }
    }

    // Generate conclusions using LLM
    const executedStepsText = state.executedSteps
      .map((step) => `Step ${step.step}: ${step.description}\nOutput: ${step.output}`)
      .join("\n\n");

    // Include CSV context in the prompt
    const csvContext = state.csvData
      ? `\nData Context: ${state.csvData.rows.length} rows × ${state.csvData.headers.length} columns (${state.csvData.headers.join(", ")})`
      : "";

    const systemPrompt = `You are a data analyst expert at summarizing analysis results.
Your task is to synthesize execution results into clear, actionable insights.
Generate 3-5 bullet-point conclusions that:
- Directly address the original analysis question
- Highlight key findings and patterns
- Are concise, specific, and insightful
- Avoid technical jargon where possible`;

    const { text } = await generateText({
      model: openai("gpt-4o"),
      system: systemPrompt,
      prompt: `Summarize the analysis results in 3-5 bullet points:

User Question: "${state.userMessage}"${csvContext}

Analysis Steps and Results:
${executedStepsText}

Focus on the key insights and findings. Format as bullet points (- or •).`,
    });

    // Parse conclusions (simple split by bullet points)
    const conclusions = text
      .split(/[-•*]\s+/)
      .filter((line) => line.trim().length > 0)
      .map((line) => line.trim());

    return {
      finalResults: {
        chartSpec,
        conclusions,
      },
    };
  } catch (error) {
    console.error("Report node error:", error);
    // Return what we have even if LLM fails
    return {
      finalResults: {
        chartSpec: state.executedSteps.find((s) => s.chartSpec)?.chartSpec,
        conclusions: ["Analysis completed. See code tab for details."],
      },
    };
  }
}
