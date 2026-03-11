"use client";

import { useState, useCallback, useRef } from "react";
import { CSVData, Plan } from "@/types";

interface WorkflowResponse {
  sessionId: string;
  step: "plan" | "execute" | "report" | "complete" | "error";
  state: {
    plan?: Plan;
    executedSteps?: any[];
    finalResults?: any;
    errorMessage?: string;
  };
}

export function useWorkflow() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState<
    "idle" | "plan" | "execute" | "report" | "complete" | "error"
  >("idle");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [executedSteps, setExecutedSteps] = useState<any[]>([]);
  const [finalResults, setFinalResults] = useState<any>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const startAnalysis = useCallback(
    async (csvData: CSVData, userMessage: string) => {
      setIsLoading(true);
      setErrorMessage(null);
      abortControllerRef.current = new AbortController();

      try {
        const response = await fetch("/api/workflow", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "start",
            csvData,
            userMessage,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) throw new Error("Failed to start analysis");

        const data: WorkflowResponse = await response.json();
        setSessionId(data.sessionId);
        setCurrentStep(data.step);

        if (data.state.plan) {
          setPlan(data.state.plan);
        }
        if (data.state.errorMessage) {
          setErrorMessage(data.state.errorMessage);
          setCurrentStep("error");
        }
      } catch (error) {
        if (error instanceof Error && error.name !== "AbortError") {
          setErrorMessage(error.message);
          setCurrentStep("error");
        }
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const approvePlan = useCallback(async () => {
    if (!sessionId) return;

    setIsLoading(true);
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch("/api/workflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "approve",
          sessionId,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) throw new Error("Failed to approve plan");

      const data: WorkflowResponse = await response.json();
      setCurrentStep(data.step);

      if (data.state.executedSteps) {
        setExecutedSteps(data.state.executedSteps);
      }
      if (data.state.finalResults) {
        setFinalResults(data.state.finalResults);
      }
      if (data.state.errorMessage) {
        setErrorMessage(data.state.errorMessage);
        setCurrentStep("error");
      }
    } catch (error) {
      if (error instanceof Error && error.name !== "AbortError") {
        setErrorMessage(error.message);
        setCurrentStep("error");
      }
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  const updatePlan = useCallback(
    async (updatedPlan: Plan) => {
      if (!sessionId) return;

      setIsLoading(true);

      try {
        const response = await fetch("/api/workflow", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "update_plan",
            sessionId,
            plan: updatedPlan,
          }),
        });

        if (!response.ok) throw new Error("Failed to update plan");

        const data: WorkflowResponse = await response.json();
        setPlan(data.state.plan || updatedPlan);
      } catch (error) {
        if (error instanceof Error) {
          setErrorMessage(error.message);
        }
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId]
  );

  const cancel = useCallback(async () => {
    if (!sessionId) return;

    abortControllerRef.current?.abort();
    setIsLoading(false);

    try {
      await fetch("/api/workflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "cancel",
          sessionId,
        }),
      });
    } catch (error) {
      console.error("Cancel error:", error);
    }

    setCurrentStep("error");
    setErrorMessage("Analysis cancelled");
  }, [sessionId]);

  return {
    // State
    sessionId,
    currentStep,
    plan,
    executedSteps,
    finalResults,
    errorMessage,
    isLoading,

    // Actions
    startAnalysis,
    approvePlan,
    updatePlan,
    cancel,
  };
}
