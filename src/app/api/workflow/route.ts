import { NextRequest, NextResponse } from "next/server";
import { createWorkflow } from "@/server/workflow";

// Store sessions in memory (in production, use a proper session store)
const sessions = new Map<string, any>();

export async function POST(request: NextRequest) {
  try {
    const { action, csvData, userMessage, sessionId, plan } =
      await request.json();

    if (!action) {
      return NextResponse.json({ error: "Action is required" }, { status: 400 });
    }

    const currentSessionId = sessionId || `session_${Date.now()}`;
    let workflow = sessions.get(currentSessionId);

    switch (action) {
      case "start": {
        if (!csvData || !userMessage) {
          return NextResponse.json(
            { error: "CSV data and user message are required" },
            { status: 400 }
          );
        }

        // Create new workflow
        workflow = createWorkflow(csvData, userMessage);
        sessions.set(currentSessionId, workflow);

        // Generate plan
        await workflow.run();

        return NextResponse.json({
          sessionId: currentSessionId,
          step: workflow.getCurrentStep(),
          state: workflow.getState(),
        });
      }

      case "approve": {
        if (!workflow) {
          return NextResponse.json(
            { error: "Session not found" },
            { status: 404 }
          );
        }

        // User approved the plan, continue execution
        workflow.setUserApproval();
        await workflow.run();

        return NextResponse.json({
          sessionId: currentSessionId,
          step: workflow.getCurrentStep(),
          state: workflow.getState(),
        });
      }

      case "update_plan": {
        if (!workflow) {
          return NextResponse.json(
            { error: "Session not found" },
            { status: 404 }
          );
        }

        if (!plan) {
          return NextResponse.json(
            { error: "Plan is required" },
            { status: 400 }
          );
        }

        // Update plan with user edits
        workflow.updatePlan(JSON.stringify(plan));

        return NextResponse.json({
          sessionId: currentSessionId,
          step: workflow.getCurrentStep(),
          state: workflow.getState(),
        });
      }

      case "cancel": {
        if (!workflow) {
          return NextResponse.json(
            { error: "Session not found" },
            { status: 404 }
          );
        }

        workflow.cancel();

        return NextResponse.json({
          sessionId: currentSessionId,
          step: workflow.getCurrentStep(),
          state: workflow.getState(),
        });
      }

      case "status": {
        if (!workflow) {
          return NextResponse.json(
            { error: "Session not found" },
            { status: 404 }
          );
        }

        return NextResponse.json({
          sessionId: currentSessionId,
          step: workflow.getCurrentStep(),
          state: workflow.getState(),
        });
      }

      default:
        return NextResponse.json(
          { error: "Unknown action" },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error("Workflow API error:", error);
    return NextResponse.json(
      { error: "Failed to process workflow request" },
      { status: 500 }
    );
  }
}
