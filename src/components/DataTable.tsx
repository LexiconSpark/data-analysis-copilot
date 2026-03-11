"use client";

import React, { useState } from "react";
import { CSVData } from "@/types";

interface DataTableProps {
  csvData: CSVData | null;
  onCellChange?: (rowIndex: number, column: string, value: string | number) => void;
}

export function DataTable({ csvData, onCellChange }: DataTableProps) {
  const [editingCell, setEditingCell] = useState<{
    row: number;
    col: string;
  } | null>(null);

  if (!csvData) {
    return (
      <div className="flex h-full items-center justify-center text-center text-gray-500">
        <p>Upload a CSV file to view data</p>
      </div>
    );
  }

  const { headers, rows } = csvData;

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-white">
      {/* Data Info */}
      <div className="border-b border-gray-300 bg-gray-50 px-4 py-2 text-xs text-gray-600">
        {rows.length} rows × {headers.length} columns
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full border-collapse text-xs">
          <thead className="sticky top-0 bg-gray-100">
            <tr>
              <th className="border border-gray-300 bg-gray-100 px-2 py-1 text-left font-medium text-gray-700">
                #
              </th>
              {headers.map((header) => (
                <th
                  key={header}
                  className="border border-gray-300 bg-gray-100 px-2 py-1 text-left font-medium text-gray-700"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIdx) => (
              <tr key={rowIdx} className="hover:bg-blue-50">
                <td className="border border-gray-300 bg-gray-50 px-2 py-1 font-medium text-gray-500">
                  {rowIdx + 1}
                </td>
                {headers.map((header) => {
                  const value = row[header];
                  const isEditing =
                    editingCell?.row === rowIdx && editingCell?.col === header;

                  return (
                    <td
                      key={`${rowIdx}-${header}`}
                      className="border border-gray-300 px-2 py-1"
                      onClick={() =>
                        setEditingCell({ row: rowIdx, col: header })
                      }
                    >
                      {isEditing ? (
                        <input
                          type="text"
                          value={String(value)}
                          onChange={(e) => {
                            onCellChange?.(rowIdx, header, e.target.value);
                          }}
                          onBlur={() => setEditingCell(null)}
                          className="w-full border border-blue-500 px-1 py-0"
                          autoFocus
                        />
                      ) : (
                        <span className="cursor-text text-gray-900">
                          {value}
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
