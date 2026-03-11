"use client";

import React, { useEffect } from "react";
import { GridLayout } from "@/components/GridLayout";
import { ChatPanel } from "@/components/ChatPanel";
import { TabsContainer } from "@/components/TabsContainer";
import { TemplatesTab } from "@/components/TemplatesTab";
import { PlanTab } from "@/components/PlanTab";
import { CodeTab } from "@/components/CodeTab";
import { ResultsPanel } from "@/components/ResultsPanel";
import { CSVStatusPanel } from "@/components/CSVStatusPanel";
import { useAppContext } from "@/context/AppContext";
import { CSVData, Plan } from "@/types";
import { DEFAULT_CSV_DATA } from "@/utils/defaultData";

export default function Home() {
  const {
    csvData,
    setCsvData,
    chatMessages,
    addChatMessage,
    plan,
    setPlan,
    activeTab,
    setActiveTab,
    executedSteps,
    addExecutedStep,
    templates,
    addTemplate,
    removeTemplate,
    loadTemplate,
    finalResults,
    setFinalResults,
  } = useAppContext();

  const [isLoading, setIsLoading] = React.useState(false);
  const [sessionId, setSessionId] = React.useState<string | null>(null);

  // Load default data on first render
  useEffect(() => {
    if (!csvData) {
      setCsvData(DEFAULT_CSV_DATA);
    }
  }, [csvData, setCsvData]);

  const handleSendMessage = async (message: string) => {
    if (!csvData) return;

    // Add user message
    addChatMessage({
      role: "user",
      content: message,
      timestamp: Date.now(),
    });

    setIsLoading(true);

    try {
      // Call chat API
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          csvData,
        }),
      });

      if (!response.ok) throw new Error("Chat request failed");

      const data = await response.json();

      // Add assistant response
      addChatMessage({
        role: "assistant",
        content: data.response,
        timestamp: Date.now(),
      });

      // If plan was generated, activate Plan tab and start workflow
      if (data.plan) {
        setPlan(data.plan);
        setActiveTab("plan");

        // Start the workflow to generate the plan
        const workflowResponse = await fetch("/api/workflow", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "start",
            csvData,
            userMessage: message,
          }),
        });

        if (workflowResponse.ok) {
          const workflowResult = await workflowResponse.json();
          setSessionId(workflowResult.sessionId);
        }
      }
    } catch (error) {
      console.error("Chat error:", error);
      addChatMessage({
        role: "assistant",
        content: "Error: Failed to process your request. Please try again.",
        timestamp: Date.now(),
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleApprovePlan = async () => {
    if (!plan || !csvData || !sessionId) return;

    setActiveTab("code");
    setIsLoading(true);

    try {
      // Approve plan and start execution
      const response = await fetch("/api/workflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "approve",
          sessionId,
        }),
      });

      if (!response.ok) throw new Error("Failed to approve and execute plan");

      const result = await response.json();
      const state = result.state;

      // Update context with executed steps and results
      if (state.executedSteps && state.executedSteps.length > 0) {
        state.executedSteps.forEach((step: any) => {
          if (!executedSteps.some((s) => s.step === step.step)) {
            addExecutedStep(step);
          }
        });
      }

      if (state.finalResults) {
        setFinalResults(state.finalResults);
      }

      // Poll for completion if still executing
      if (result.step !== "complete" && result.step !== "error") {
        // Continue polling
        let pollCount = 0;
        const pollInterval = setInterval(async () => {
          try {
            const statusResponse = await fetch("/api/workflow", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                action: "status",
                sessionId,
              }),
            });

            if (statusResponse.ok) {
              const statusResult = await statusResponse.json();
              const statusState = statusResult.state;

              // Update executed steps
              if (statusState.executedSteps && statusState.executedSteps.length > 0) {
                statusState.executedSteps.forEach((step: any) => {
                  if (!executedSteps.some((s) => s.step === step.step)) {
                    addExecutedStep(step);
                  }
                });
              }

              if (statusState.finalResults) {
                setFinalResults(statusState.finalResults);
              }

              // Stop polling if complete or error
              if (statusResult.step === "complete" || statusResult.step === "error") {
                clearInterval(pollInterval);
                setIsLoading(false);
              }
            }

            pollCount++;
            if (pollCount > 30) {
              clearInterval(pollInterval); // Stop after 30 seconds
              setIsLoading(false);
            }
          } catch (pollError) {
            console.error("Poll error:", pollError);
            clearInterval(pollInterval);
            setIsLoading(false);
          }
        }, 1000); // Poll every second
      } else {
        setIsLoading(false);
      }
    } catch (error) {
      console.error("Execution error:", error);
      addChatMessage({
        role: "assistant",
        content: "Error: Failed to execute plan. Please try again.",
        timestamp: Date.now(),
      });
      setIsLoading(false);
    }
  };

  const handleSaveTemplate = (name: string) => {
    if (!plan) return;

    const template = {
      id: Date.now().toString(),
      name,
      plan,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    addTemplate(template);
  };


  // Render tab content
  const templatesContent = (
    <TemplatesTab
      templates={templates}
      onLoadTemplate={loadTemplate}
      onDeleteTemplate={removeTemplate}
      onNewAnalysis={() => {
        setPlan({ steps: [], userRequest: "" });
        setActiveTab("plan");
      }}
    />
  );

  const planContent = (
    <PlanTab
      plan={plan}
      onApprovePlan={handleApprovePlan}
      onSaveTemplate={handleSaveTemplate}
    />
  );

  const codeContent = <CodeTab steps={executedSteps} />;

  const resultsContent = (
    <ResultsPanel
      chartSpec={finalResults?.chartSpec}
      conclusions={finalResults?.conclusions}
    />
  );

  return (
    <GridLayout
      topLeft={
        <ChatPanel
          messages={chatMessages}
          csvData={csvData}
          onUploadCSV={setCsvData}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
        />
      }
      topRight={
        <TabsContainer
          activeTab={activeTab}
          onTabChange={setActiveTab}
          templates={templatesContent}
          plan={planContent}
          code={codeContent}
        />
      }
      bottomLeft={<CSVStatusPanel csvData={csvData} onUploadCSV={setCsvData} isDefault={csvData === DEFAULT_CSV_DATA} />}
      bottomRight={
        <ResultsPanel
          chartSpec={finalResults?.chartSpec}
          conclusions={finalResults?.conclusions}
        />
      }
    />
  );
}
