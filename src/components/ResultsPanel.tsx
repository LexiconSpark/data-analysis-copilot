"use client";

import React from "react";
import { ChartSpec } from "@/types";
import { ChartRenderer } from "./ChartRenderer";

interface ResultsPanelProps {
  chartSpec?: ChartSpec;
  conclusions?: string[];
  isLoading?: boolean;
}

export function ResultsPanel({
  chartSpec,
  conclusions,
  isLoading = false,
}: ResultsPanelProps) {
  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <div className="mb-4 h-8 w-8 animate-spin rounded-full border-4 border-gray-300 border-t-blue-600 mx-auto"></div>
          <p className="text-gray-600">Generating analysis...</p>
        </div>
      </div>
    );
  }

  if (!chartSpec && (!conclusions || conclusions.length === 0)) {
    return (
      <div className="flex h-full items-center justify-center text-center text-gray-500">
        <p>No results yet. Approve and execute a plan to see analysis results.</p>
      </div>
    );
  }

  const handleDownload = () => {
    if (!chartSpec) return;
    // Download functionality would be implemented here
    console.log("Download chart:", chartSpec);
  };

  return (
    <div className="flex h-full w-full flex-col overflow-auto p-4">
      {/* Chart */}
      {chartSpec && (
        <div className="mb-4 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-gray-900">{chartSpec.title}</h3>
            <button
              onClick={handleDownload}
              className="rounded bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-700"
            >
              Download
            </button>
          </div>
          <div className="flex-1 rounded border border-gray-300 bg-white p-4">
            <ChartRenderer spec={chartSpec} />
          </div>
        </div>
      )}

      {/* Conclusions */}
      {conclusions && conclusions.length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="font-semibold text-gray-900">Conclusions</h3>
          <ul className="space-y-2">
            {conclusions.map((conclusion, idx) => (
              <li
                key={idx}
                className="flex gap-2 text-xs text-gray-700 leading-relaxed"
              >
                <span className="mt-1 h-2 w-2 flex-shrink-0 rounded-full bg-blue-600"></span>
                <span>{conclusion}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
