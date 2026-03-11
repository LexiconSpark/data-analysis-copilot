"use client";

import { useState } from "react";
import { Plan } from "@/types";

interface PlanTabProps {
  plan: Plan | null;
  onApprovePlan: () => void;
  onSaveTemplate: (name: string) => void;
  isApproving?: boolean;
}

export function PlanTab({
  plan,
  onApprovePlan,
  onSaveTemplate,
  isApproving = false,
}: PlanTabProps) {
  const [templateName, setTemplateName] = useState("");
  const [showSaveTemplate, setShowSaveTemplate] = useState(false);

  if (!plan || plan.steps.length === 0) {
    return (
      <div className="flex items-center justify-center text-center text-gray-500">
        <p>No plan generated yet. Submit an analysis request to generate a plan.</p>
      </div>
    );
  }

  const handleSaveTemplate = () => {
    if (templateName.trim()) {
      onSaveTemplate(templateName);
      setTemplateName("");
      setShowSaveTemplate(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Steps */}
      <div className="flex-1 overflow-y-auto border-b border-gray-300 p-4">
        <p className="mb-3 text-xs font-semibold text-gray-600">Analysis Plan — {plan.steps.length} Steps</p>
        <div className="space-y-2">
          {plan.steps.map((step, idx) => (
            <div key={idx} className="rounded border border-gray-200 bg-gray-50 p-3 hover:bg-gray-100 transition-colors">
              <div className="flex items-start gap-3">
                <span className="inline-flex h-6 w-6 min-w-fit items-center justify-center rounded-full bg-blue-600 text-xs font-bold text-white">
                  {step.step}
                </span>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900">{step.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="border-t border-gray-300 p-3">
        {showSaveTemplate ? (
          <div className="mb-3 space-y-2">
            <input
              type="text"
              value={templateName}
              onChange={(e) => setTemplateName(e.target.value)}
              placeholder="Template name..."
              className="w-full rounded border border-gray-300 px-2 py-1 text-sm focus:border-blue-500 focus:outline-none"
            />
            <div className="flex gap-2">
              <button
                onClick={handleSaveTemplate}
                className="flex-1 rounded bg-green-600 px-2 py-1 text-sm font-medium text-white hover:bg-green-700"
              >
                Save
              </button>
              <button
                onClick={() => {
                  setShowSaveTemplate(false);
                  setTemplateName("");
                }}
                className="flex-1 rounded border border-gray-300 px-2 py-1 text-sm hover:bg-gray-50"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={() => setShowSaveTemplate(true)}
            className="w-full rounded border border-gray-300 px-2 py-1 text-sm font-medium hover:bg-gray-50"
          >
            Save as Template
          </button>
        )}

        <button
          onClick={onApprovePlan}
          disabled={isApproving}
          className="mt-2 w-full rounded bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700 disabled:bg-gray-400"
        >
          {isApproving ? "Approving..." : "Approve Plan"}
        </button>
      </div>
    </div>
  );
}
