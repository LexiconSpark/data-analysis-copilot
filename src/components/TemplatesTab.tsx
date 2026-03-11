"use client";

import React from "react";
import { Template } from "@/types";

interface TemplatesTabProps {
  templates: Template[];
  onLoadTemplate: (template: Template) => void;
  onDeleteTemplate: (id: string) => void;
  onNewAnalysis: () => void;
}

export function TemplatesTab({
  templates,
  onLoadTemplate,
  onDeleteTemplate,
  onNewAnalysis,
}: TemplatesTabProps) {
  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900">Saved Templates</h2>
        <button
          onClick={onNewAnalysis}
          className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          New Analysis
        </button>
      </div>

      {templates.length === 0 ? (
        <div className="flex flex-1 items-center justify-center text-center">
          <div className="text-gray-500">
            <p className="mb-2">No templates saved yet</p>
            <p className="text-sm">
              Save your first analysis plan as a template to reuse it
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-2 overflow-y-auto">
          {templates.map((template) => (
            <div
              key={template.id}
              className="flex items-center justify-between rounded border border-gray-300 p-3"
            >
              <div>
                <p className="font-medium text-gray-900">{template.name}</p>
                <p className="text-xs text-gray-500">
                  {template.plan.steps.length} steps
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => onLoadTemplate(template)}
                  className="text-sm text-blue-600 hover:underline"
                >
                  Load
                </button>
                <button
                  onClick={() => onDeleteTemplate(template.id)}
                  className="text-sm text-red-600 hover:underline"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
