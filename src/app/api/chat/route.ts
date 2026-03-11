import { NextRequest, NextResponse } from "next/server";
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";

export async function POST(request: NextRequest) {
  try {
    const { message, csvData } = await request.json();

    if (!message) {
      return NextResponse.json(
        { error: "Message is required" },
        { status: 400 }
      );
    }

    // Prepare CSV context (always use default or user-uploaded data)
    const csvContext = csvData
      ? `Current CSV Data:
Columns: ${csvData.headers.join(", ")}
Total rows: ${csvData.rows.length}
Sample data (first 2 rows):
${JSON.stringify(csvData.rows.slice(0, 2), null, 2)}`
      : "No data loaded";

    const systemPrompt = `You are an expert data analyst assistant. When the user asks a data analysis question:
1. Provide a clear, conversational response summarizing your approach and key findings.
2. Acknowledge the data and suggest analysis steps.
3. Be helpful and insightful.

If the question is purely conversational (not a data analysis request), respond naturally without suggesting analysis steps.

${csvContext}`;

    // Call OpenAI API
    const { text } = await generateText({
      model: openai("gpt-4o"),
      system: systemPrompt,
      prompt: message,
    });

    // Check if user is asking for analysis (heuristic)
    const shouldGeneratePlan =
      message.toLowerCase().includes("plot") ||
      message.toLowerCase().includes("analyze") ||
      message.toLowerCase().includes("chart") ||
      message.toLowerCase().includes("visualiz");

    let plan = null;
    if (shouldGeneratePlan && csvData) {
      // Generate a basic plan structure
      plan = {
        steps: [
          {
            step: 1,
            description: "Prepare and filter data",
            pythonCode: "import pandas as pd\n# Data preparation",
          },
          {
            step: 2,
            description: "Generate visualization",
            pythonCode: "import matplotlib.pyplot as plt\n# Create chart",
          },
          {
            step: 3,
            description: "Create conclusions",
            pythonCode: "# Analyze results",
          },
        ],
        userRequest: message,
      };
    }

    return NextResponse.json({
      response: text,
      plan: plan,
    });
  } catch (error) {
    console.error("Chat API error:", error);
    return NextResponse.json(
      { error: "Failed to process chat request" },
      { status: 500 }
    );
  }
}
