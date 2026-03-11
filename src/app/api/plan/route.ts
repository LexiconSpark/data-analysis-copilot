import { NextRequest, NextResponse } from "next/server";
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";

export async function POST(request: NextRequest) {
  try {
    const { userRequest, csvData } = await request.json();

    if (!userRequest || !csvData) {
      return NextResponse.json(
        { error: "User request and CSV data are required" },
        { status: 400 }
      );
    }

    // Prepare CSV context for the LLM
    const csvSummary = `
Data Summary:
- Columns: ${csvData.headers.join(", ")}
- Row count: ${csvData.rows.length}
- Sample rows: ${JSON.stringify(csvData.rows.slice(0, 2))}`;

    const systemPrompt = `You are an expert data analyst. Generate a detailed step-by-step analysis plan.
For each step, provide:
1. A clear description of what to do
2. Python code to execute

Return your response as JSON in this format:
{
  "steps": [
    {
      "step": 1,
      "description": "...",
      "pythonCode": "..."
    }
  ]
}`;

    const { text } = await generateText({
      model: openai("gpt-4o"),
      system: systemPrompt,
      prompt: `Create an analysis plan for: ${userRequest}\n\n${csvSummary}`,
    });

    // Parse the response
    let plan = null;
    try {
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        plan = JSON.parse(jsonMatch[0]);
      }
    } catch (e) {
      // Fallback to basic plan structure
      plan = {
        steps: [
          {
            step: 1,
            description: "Analyze the data",
            pythonCode: "import pandas as pd\n# Load and analyze data",
          },
        ],
      };
    }

    return NextResponse.json({
      plan,
      planText: text,
    });
  } catch (error) {
    console.error("Plan API error:", error);
    return NextResponse.json(
      { error: "Failed to generate plan" },
      { status: 500 }
    );
  }
}
