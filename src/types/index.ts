// CSV Data types
export interface CSVData {
  headers: string[];
  rows: Record<string, string | number>[];
}

// Plan types
export interface PlanStep {
  step: number;
  description: string;
  pythonCode: string;
}

export interface Plan {
  steps: PlanStep[];
  userRequest: string;
}

// Execution types
export interface ExecutionResult {
  stepIndex: number;
  output: string;
  error?: string;
  chartSpec?: ChartSpec;
}

export interface StepResult {
  step: number;
  description: string;
  pythonCode: string;
  output: string;
  error?: string;
  chartSpec?: ChartSpec;
}

// Chart types
export interface ChartSpec {
  type: "scatter" | "line" | "bar" | "composed";
  data: any[];
  title: string;
  xAxis: string;
  yAxis: string;
  series?: string[];
}

// Workflow state types
export interface WorkflowState {
  csvData: CSVData | null;
  userMessage: string;
  plan: Plan | null;
  currentStepIndex: number;
  executedSteps: StepResult[];
  retryCount: number;
  cancelled: boolean;
  finalResults: {
    chartSpec?: ChartSpec;
    conclusions?: string[];
  } | null;
  workflowState: "idle" | "generating_plan" | "waiting_approval" | "executing" | "complete" | "error";
}

// Template types
export interface Template {
  id: string;
  name: string;
  plan: Plan;
  createdAt: number;
  updatedAt: number;
}

// Chat types
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}
