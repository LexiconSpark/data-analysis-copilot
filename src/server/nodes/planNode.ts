import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";
import { WorkflowState } from "../workflowTypes";
import { Plan } from "@/types";

export async function planNode(state: WorkflowState): Promise<Partial<WorkflowState>> {
  if (!state.csvData || !state.userMessage) {
    return {
      errorMessage: "Missing CSV data or user message",
    };
  }

  try {
    // Prepare CSV context
    const csvSummary = `
Data Summary:
- Columns: ${state.csvData.headers.join(", ")}
- Row count: ${state.csvData.rows.length}
- Sample rows: ${JSON.stringify(state.csvData.rows.slice(0, 2))}`;

    const systemPrompt = `You are an expert data analyst. Generate a detailed step-by-step analysis plan.
For each step, provide:
1. A clear, actionable description of what to do
2. Complete Python code to execute

The code will be executed in an environment with these libraries pre-imported:
- pandas (pd) - for data manipulation
- numpy (np) - for numerical operations
- matplotlib.pyplot (plt) - for plotting
- scipy - for scientific computing

In the code:
- Store any visualization in a variable named 'fig'
- Store results in a variable named 'result' (DataFrame or text)
- Code should be complete and executable

Return your response as JSON in this format:
{
  "steps": [
    {
      "step": 1,
      "description": "Clear, specific description",
      "pythonCode": "Complete Python code"
    }
  ]
}`;

    const { text } = await generateText({
      model: openai("gpt-4o"),
      system: systemPrompt,
      prompt: `Generate a step-by-step analysis plan for this user request:

"${state.userMessage}"

${csvSummary}

Provide 3-5 specific, concrete analysis steps. Each step should be:
- Actionable and specific
- Focused on data exploration, filtering, visualization, or statistical analysis
- Implementable with pandas, numpy, matplotlib, and scipy
- Clear about what the output should be

Return only the JSON response with the steps array.`,
    });

    // Parse the response
    let plan: Plan | null = null;
    try {
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        plan = {
          steps: parsed.steps || [],
          userRequest: state.userMessage,
        };
      }
    } catch (e) {
      // Fallback to a basic plan
      plan = {
        steps: [
          {
            step: 1,
            description: "Analyze the data",
            pythonCode: "import pandas as pd\n# Data analysis",
          },
        ],
        userRequest: state.userMessage,
      };
    }

    return {
      plan,
      currentStepIndex: 0,
    };
  } catch (error) {
    console.error("Plan node error:", error);
    return {
      errorMessage: `Failed to generate plan: ${error instanceof Error ? error.message : "Unknown error"}`,
    };
  }
}
