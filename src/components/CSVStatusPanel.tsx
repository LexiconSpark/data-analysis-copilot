"use client";

import React, { useState } from "react";
import { CSVData } from "@/types";

interface CSVStatusPanelProps {
  csvData: CSVData | null;
  onUploadCSV?: (data: CSVData) => void;
  isDefault?: boolean;
}

export function CSVStatusPanel({
  csvData,
  onUploadCSV,
  isDefault = true,
}: CSVStatusPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!csvData) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-gray-500">
        <p>No CSV data loaded</p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden bg-white">
      {/* Header */}
      <div className="border-b border-gray-300 p-3">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-gray-600">
              {isDefault ? "Default CSV" : "Custom CSV"}
            </p>
            <p className="text-sm font-semibold text-gray-900">
              {csvData.rows.length} rows × {csvData.headers.length} columns
            </p>
          </div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="rounded px-2 py-1 text-xs text-blue-600 hover:bg-blue-50"
          >
            {isExpanded ? "Hide" : "View"}
          </button>
        </div>
      </div>

      {/* Expandable Data Preview */}
      {isExpanded && (
        <div className="flex-1 overflow-auto">
          <div className="p-3">
            {/* Column Headers */}
            <div className="mb-2 grid gap-2" style={{ gridTemplateColumns: `repeat(${csvData.headers.length}, 1fr)` }}>
              {csvData.headers.map((header) => (
                <div
                  key={header}
                  className="text-xs font-bold text-gray-700 bg-gray-100 p-1 rounded"
                >
                  {header}
                </div>
              ))}
            </div>

            {/* Data Rows (show first 10) */}
            <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${csvData.headers.length}, 1fr)` }}>
              {csvData.rows.slice(0, 10).map((row, rowIdx) => (
                <React.Fragment key={rowIdx}>
                  {csvData.headers.map((header) => (
                    <div
                      key={`${rowIdx}-${header}`}
                      className="text-xs text-gray-700 p-1 border border-gray-200 rounded"
                    >
                      {row[header]}
                    </div>
                  ))}
                </React.Fragment>
              ))}
            </div>

            {csvData.rows.length > 10 && (
              <p className="mt-2 text-xs text-gray-500">
                +{csvData.rows.length - 10} more rows
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
