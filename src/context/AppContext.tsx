"use client";

import React, { createContext, useCallback, useState } from "react";
import {
  CSVData,
  Plan,
  StepResult,
  WorkflowState,
  ChatMessage,
  Template,
} from "@/types";

interface AppContextType {
  // State
  csvData: CSVData | null;
  plan: Plan | null;
  executedSteps: StepResult[];
  retryCount: number;
  cancelled: boolean;
  workflowState: WorkflowState["workflowState"];
  chatMessages: ChatMessage[];
  activeTab: "templates" | "plan" | "code";
  finalResults: {
    chartSpec?: any;
    conclusions?: string[];
  } | null;
  templates: Template[];

  // Actions
  setCsvData: (data: CSVData | null) => void;
  setPlan: (plan: Plan | null) => void;
  setExecutedSteps: (steps: StepResult[]) => void;
  addExecutedStep: (step: StepResult) => void;
  setRetryCount: (count: number) => void;
  setCancelled: (cancelled: boolean) => void;
  setWorkflowState: (state: WorkflowState["workflowState"]) => void;
  addChatMessage: (message: ChatMessage) => void;
  setChatMessages: (messages: ChatMessage[]) => void;
  setActiveTab: (tab: "templates" | "plan" | "code") => void;
  setFinalResults: (results: any) => void;
  addTemplate: (template: Template) => void;
  removeTemplate: (id: string) => void;
  loadTemplate: (template: Template) => void;
}

export const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [csvData, setCsvData] = useState<CSVData | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [executedSteps, setExecutedSteps] = useState<StepResult[]>([]);
  const [retryCount, setRetryCount] = useState(0);
  const [cancelled, setCancelled] = useState(false);
  const [workflowState, setWorkflowState] = useState<WorkflowState["workflowState"]>("idle");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [activeTab, setActiveTab] = useState<"templates" | "plan" | "code">(
    "templates"
  );
  const [finalResults, setFinalResults] = useState<any>(null);
  const [templates, setTemplates] = useState<Template[]>(() => {
    // Load templates from localStorage on mount
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem("templates");
      return stored ? JSON.parse(stored) : [];
    }
    return [];
  });

  const addExecutedStep = useCallback((step: StepResult) => {
    setExecutedSteps((prev) => [...prev, step]);
  }, []);

  const addChatMessage = useCallback((message: ChatMessage) => {
    setChatMessages((prev) => [...prev, message]);
  }, []);

  const addTemplate = useCallback((template: Template) => {
    setTemplates((prev) => {
      const updated = [...prev, template];
      if (typeof window !== "undefined") {
        localStorage.setItem("templates", JSON.stringify(updated));
      }
      return updated;
    });
  }, []);

  const removeTemplate = useCallback((id: string) => {
    setTemplates((prev) => {
      const updated = prev.filter((t) => t.id !== id);
      if (typeof window !== "undefined") {
        localStorage.setItem("templates", JSON.stringify(updated));
      }
      return updated;
    });
  }, []);

  const loadTemplate = useCallback((template: Template) => {
    setPlan(template.plan);
    setActiveTab("plan");
  }, []);

  const value: AppContextType = {
    csvData,
    plan,
    executedSteps,
    retryCount,
    cancelled,
    workflowState,
    chatMessages,
    activeTab,
    finalResults,
    templates,

    setCsvData,
    setPlan,
    setExecutedSteps,
    addExecutedStep,
    setRetryCount,
    setCancelled,
    setWorkflowState,
    addChatMessage,
    setChatMessages,
    setActiveTab,
    setFinalResults,
    addTemplate,
    removeTemplate,
    loadTemplate,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useAppContext() {
  const context = React.useContext(AppContext);
  if (context === undefined) {
    throw new Error("useAppContext must be used within AppProvider");
  }
  return context;
}
