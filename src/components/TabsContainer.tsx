"use client";

import React from "react";

interface TabsContainerProps {
  activeTab: "templates" | "plan" | "code";
  onTabChange: (tab: "templates" | "plan" | "code") => void;
  templates: React.ReactNode;
  plan: React.ReactNode;
  code: React.ReactNode;
}

export function TabsContainer({
  activeTab,
  onTabChange,
  templates,
  plan,
  code,
}: TabsContainerProps) {
  const tabs = [
    { id: "templates", label: "Templates" },
    { id: "plan", label: "Plan" },
    { id: "code", label: "Code" },
  ] as const;

  const contentMap: Record<string, React.ReactNode> = {
    templates,
    plan,
    code,
  };

  return (
    <div className="flex h-full w-full flex-col">
      {/* Tab Navigation */}
      <div className="flex border-b border-gray-300 bg-gray-50">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? "border-b-2 border-blue-600 text-blue-600"
                : "text-gray-600 hover:text-gray-900"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-auto p-4">
        {contentMap[activeTab]}
      </div>
    </div>
  );
}
