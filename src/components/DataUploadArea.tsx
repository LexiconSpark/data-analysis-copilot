"use client";

import React, { useRef, useState } from "react";
import Papa from "papaparse";
import { CSVData } from "@/types";

interface DataUploadAreaProps {
  onUpload: (data: CSVData) => void;
  fileName?: string;
}

export function DataUploadArea({ onUpload, fileName }: DataUploadAreaProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      processFile(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  };

  const processFile = (file: File) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.data && Array.isArray(results.data)) {
          const headers = Object.keys((results.data[0] as Record<string, any>) || {});
          const csvData: CSVData = {
            headers,
            rows: results.data as Record<string, string | number>[],
          };
          onUpload(csvData);
        }
      },
      error: (error) => {
        console.error("CSV parsing error:", error);
      },
    });
  };

  return (
    <div
      className={`flex h-full w-full flex-col items-center justify-center gap-4 border-2 border-dashed p-6 transition-colors ${
        isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-gray-50"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv"
        onChange={handleFileSelect}
        className="hidden"
      />

      {fileName ? (
        <div className="text-center">
          <p className="text-sm font-medium text-gray-700">Loaded: {fileName}</p>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="mt-2 text-sm text-blue-600 hover:underline"
          >
            Upload another file
          </button>
        </div>
      ) : (
        <>
          <p className="text-center text-sm text-gray-700">
            Drag and drop a CSV file here or click to browse
          </p>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
          >
            Browse Files
          </button>
        </>
      )}
    </div>
  );
}
