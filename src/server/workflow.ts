import { WorkflowState } from "./workflowTypes";
import { planNode } from "./nodes/planNode";
import { executeNode } from "./nodes/executeNode";
import { reportNode } from "./nodes/reportNode";

export type WorkflowStep = "plan" | "execute" | "report" | "complete" | "error";

interface WorkflowStateWithStep extends WorkflowState {
  currentStep: WorkflowStep;
}

/**
 * Simple workflow orchestrator (replaces LangGraph for MVP)
 * Manages the state transitions through Plan -> Interrupt -> Execute -> Report
 */
export class WorkflowOrchestrator {
  private state: WorkflowStateWithStep;

  constructor(initialState: WorkflowState) {
    this.state = {
      ...initialState,
      currentStep: "plan",
    };
  }

  getState(): WorkflowState {
    const { currentStep, ...rest } = this.state;
    return rest;
  }

  getCurrentStep(): WorkflowStep {
    return this.state.currentStep;
  }

  async transitionPlan(): Promise<void> {
    const result = await planNode(this.state);
    this.state = { ...this.state, ...result, currentStep: "execute" };

    if (result.errorMessage) {
      this.state.currentStep = "error";
    }
  }

  async transitionExecute(): Promise<void> {
    if (!this.state.plan || this.state.plan.steps.length === 0) {
      this.state.currentStep = "error";
      this.state.errorMessage = "No plan to execute";
      return;
    }

    // Check if we've completed all steps
    if (this.state.currentStepIndex >= this.state.plan.steps.length) {
      this.state.currentStep = "report";
      return;
    }

    const result = await executeNode(this.state);
    this.state = { ...this.state, ...result };

    if (result.errorMessage) {
      this.state.currentStep = "error";
    } else if (this.state.plan && this.state.currentStepIndex >= this.state.plan.steps.length) {
      this.state.currentStep = "report";
    }
    // Otherwise stay in execute for next step
  }

  async transitionReport(): Promise<void> {
    const result = await reportNode(this.state);
    this.state = { ...this.state, ...result, currentStep: "complete" };

    if (result.errorMessage) {
      this.state.currentStep = "error";
    }
  }

  setUserApproval(): void {
    // User approved the plan, move to execution
    if (this.state.currentStep === "plan") {
      this.state.currentStep = "execute";
    }
  }

  cancel(): void {
    this.state.cancelled = true;
  }

  updatePlan(planText: string): void {
    // Allow user to edit plan before execution
    if (this.state.currentStep === "plan" && this.state.plan) {
      try {
        const parsed = JSON.parse(planText);
        if (parsed.steps && Array.isArray(parsed.steps)) {
          this.state.plan.steps = parsed.steps;
        }
      } catch (e) {
        this.state.errorMessage = "Invalid plan format";
      }
    }
  }

  async run(): Promise<void> {
    while (this.state.currentStep !== "complete" && this.state.currentStep !== "error") {
      switch (this.state.currentStep) {
        case "plan":
          await this.transitionPlan();
          // Pause here for user approval
          return;
        case "execute":
          await this.transitionExecute();
          break;
        case "report":
          await this.transitionReport();
          break;
        default:
          return;
      }

      if (this.state.cancelled) {
        this.state.currentStep = "error";
        this.state.errorMessage = "Execution cancelled";
        break;
      }
    }
  }
}

/**
 * Create a new workflow session
 */
export function createWorkflow(
  csvData: any,
  userMessage: string
): WorkflowOrchestrator {
  return new WorkflowOrchestrator({
    csvData,
    userMessage,
    plan: null,
    currentStepIndex: 0,
    executedSteps: [],
    retryCount: 0,
    cancelled: false,
    finalResults: null,
  });
}
