"use client";

import React from "react";
import { StepResult } from "@/types";

interface CodeTabProps {
  steps: StepResult[];
}

export function CodeTab({ steps }: CodeTabProps) {
  if (steps.length === 0) {
    return (
      <div className="flex items-center justify-center text-center text-gray-500">
        <p>No code executed yet. Approve a plan to see execution steps.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4 overflow-y-auto p-4">
      {steps.map((step, idx) => (
        <div key={idx} className="rounded border border-gray-300 p-4 bg-white">
          {/* Step Header */}
          <div className="mb-3 flex items-center gap-2">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-blue-600 text-sm font-bold text-white">
              {step.step}
            </span>
            <div>
              <p className="font-medium text-gray-900">{step.description}</p>
            </div>
          </div>

          {/* Python Code Section */}
          <div className="mb-3">
            <p className="mb-1 text-xs font-semibold text-gray-600">Python Code:</p>
            <div className="rounded bg-gray-900 p-3">
              <pre className="overflow-x-auto font-mono text-xs text-gray-100">
                <code>{step.pythonCode}</code>
              </pre>
            </div>
          </div>

          {/* Execution Result */}
          <div className="mb-3">
            <p className="mb-1 text-xs font-semibold text-gray-600">Execution Result:</p>
            {step.error ? (
              <div className="rounded border border-red-300 bg-red-50 p-3 text-xs text-red-700">
                <p className="mb-1 font-medium">❌ Error</p>
                <pre className="overflow-x-auto whitespace-pre-wrap">{step.error}</pre>
              </div>
            ) : (
              <div className="rounded border border-green-300 bg-green-50 p-3 text-xs text-green-700">
                <p className="mb-1 font-medium">✓ Success</p>
                <pre className="overflow-x-auto whitespace-pre-wrap">
                  {step.output.substring(0, 500)}
                  {step.output.length > 500 ? "..." : ""}
                </pre>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
