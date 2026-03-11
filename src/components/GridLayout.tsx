import React from "react";

interface GridLayoutProps {
  topLeft: React.ReactNode;
  topRight: React.ReactNode;
  bottomLeft: React.ReactNode;
  bottomRight: React.ReactNode;
}

export function GridLayout({
  topLeft,
  topRight,
  bottomLeft,
  bottomRight,
}: GridLayoutProps) {
  return (
    <div className="h-screen w-screen overflow-hidden bg-white">
      <div className="grid h-full w-full grid-cols-[40%_60%] grid-rows-[60%_40%] gap-px border border-gray-300">
        {/* Top-Left: Chat Interface */}
        <div className="border-r border-b border-gray-300 overflow-hidden bg-white">
          {topLeft}
        </div>

        {/* Top-Right: Tabs Container */}
        <div className="border-b border-gray-300 overflow-hidden bg-white">
          {topRight}
        </div>

        {/* Bottom-Left: CSV Data Table */}
        <div className="border-r border-gray-300 overflow-hidden bg-white">
          {bottomLeft}
        </div>

        {/* Bottom-Right: Analysis Results */}
        <div className="overflow-hidden bg-white">{bottomRight}</div>
      </div>
    </div>
  );
}
