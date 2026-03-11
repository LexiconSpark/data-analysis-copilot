"use client";

import React, { useRef, useEffect } from "react";
import Papa from "papaparse";
import { CSVData, ChatMessage } from "@/types";
import { DEFAULT_CSV_DATA } from "@/utils/defaultData";

interface ChatPanelProps {
  messages: ChatMessage[];
  csvData: CSVData | null;
  onUploadCSV: (data: CSVData) => void;
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
}

export function ChatPanel({
  messages,
  csvData,
  onUploadCSV,
  onSendMessage,
  isLoading = false,
}: ChatPanelProps) {
  const [inputValue, setInputValue] = React.useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = () => {
    if (inputValue.trim()) {
      onSendMessage(inputValue);
      setInputValue("");
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results: any) => {
          if (results.data && results.data.length > 0) {
            const headers = Object.keys(results.data[0]);
            const rows = results.data as Record<string, string | number>[];
            onUploadCSV({ headers, rows });
          }
        },
        error: (error: any) => {
          console.error("CSV parse error:", error);
        },
      });
    }
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const isDefault = csvData === DEFAULT_CSV_DATA;

  return (
    <div className="flex h-full w-full flex-col">
      {/* CSV Status Section */}
      <div className="border-b border-gray-300 bg-gray-50 p-3">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-gray-600">
              {csvData ? (
                <>
                  CSV loaded: <span className="font-semibold">{csvData.rows.length} rows × {csvData.headers.length} columns</span>
                  {isDefault && <span className="ml-2 text-gray-500">(default)</span>}
                </>
              ) : (
                <span className="text-gray-500">No CSV loaded</span>
              )}
            </p>
          </div>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="rounded px-2 py-1 text-xs text-blue-600 hover:bg-blue-50"
          >
            Upload CSV
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-center text-gray-500">
            <p>
              {csvData
                ? "No messages yet. Start typing an analysis request..."
                : "Upload a CSV file to begin your analysis"}
            </p>
          </div>
        ) : (
          <>
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`mb-3 flex ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-xs rounded px-3 py-2 text-sm ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-gray-200 text-gray-900"
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-300 p-3">
        <div className="flex gap-2">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Describe your analysis request..."
            className="flex-1 resize-none rounded border border-gray-300 p-2 text-sm text-gray-900 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            rows={2}
            disabled={isLoading || !csvData}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !inputValue.trim() || !csvData}
            className="rounded bg-blue-600 px-4 text-white hover:bg-blue-700 disabled:bg-gray-400"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
