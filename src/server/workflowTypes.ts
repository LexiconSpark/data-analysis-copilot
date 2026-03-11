import { CSVData, Plan, StepResult } from "@/types";

export interface WorkflowState {
  // Input data
  csvData: CSVData | null;
  userMessage: string;

  // Plan & execution
  plan: Plan | null;
  currentStepIndex: number;
  executedSteps: StepResult[];
  retryCount: number;

  // Control flow
  cancelled: boolean;
  errorMessage?: string;

  // Results
  finalResults: {
    chartSpec?: any;
    conclusions?: string[];
  } | null;
}

export const WorkflowStateSchema = {
  csvData: null,
  userMessage: "",
  plan: null,
  currentStepIndex: 0,
  executedSteps: [],
  retryCount: 0,
  cancelled: false,
  errorMessage: undefined,
  finalResults: null,
} as const;
