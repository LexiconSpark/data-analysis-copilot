"use client";

import React from "react";
import {
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  BarChart,
  Bar,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { ChartSpec } from "@/types";

interface ChartRendererProps {
  spec: ChartSpec;
}

export function ChartRenderer({ spec }: ChartRendererProps) {
  if (!spec.data || spec.data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        No data to display
      </div>
    );
  }

  const commonProps = {
    data: spec.data,
    margin: { top: 5, right: 30, left: 0, bottom: 5 },
  };

  switch (spec.type) {
    case "scatter":
      return (
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey={spec.xAxis}
              name={spec.xAxis}
              label={{ value: spec.xAxis, position: "right", offset: 10 }}
            />
            <YAxis
              type="number"
              dataKey={spec.yAxis}
              name={spec.yAxis}
              label={{ value: spec.yAxis, angle: -90, position: "insideLeft" }}
            />
            <Tooltip />
            {spec.series && spec.series.length > 0 ? (
              spec.series.map((series, idx) => (
                <Scatter
                  key={idx}
                  name={series}
                  dataKey={series}
                  fill={`hsl(${(idx * 360) / spec.series!.length}, 70%, 50%)`}
                />
              ))
            ) : (
              <Scatter name="Data" dataKey={spec.yAxis} fill="hsl(210, 70%, 50%)" />
            )}
          </ScatterChart>
        </ResponsiveContainer>
      );

    case "line":
      return (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey={spec.xAxis}
              label={{ value: spec.xAxis, position: "right", offset: 10 }}
            />
            <YAxis
              label={{ value: spec.yAxis, angle: -90, position: "insideLeft" }}
            />
            <Tooltip />
            <Legend />
            {spec.series && spec.series.length > 0 ? (
              spec.series.map((series, idx) => (
                <Line
                  key={idx}
                  type="monotone"
                  dataKey={series}
                  stroke={`hsl(${(idx * 360) / spec.series!.length}, 70%, 50%)`}
                  dot={false}
                />
              ))
            ) : (
              <Line type="monotone" dataKey={spec.yAxis} stroke="hsl(210, 70%, 50%)" />
            )}
          </LineChart>
        </ResponsiveContainer>
      );

    case "bar":
      return (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey={spec.xAxis}
              label={{ value: spec.xAxis, position: "right", offset: 10 }}
            />
            <YAxis
              label={{ value: spec.yAxis, angle: -90, position: "insideLeft" }}
            />
            <Tooltip />
            <Legend />
            {spec.series && spec.series.length > 0 ? (
              spec.series.map((series, idx) => (
                <Bar
                  key={idx}
                  dataKey={series}
                  fill={`hsl(${(idx * 360) / spec.series!.length}, 70%, 50%)`}
                />
              ))
            ) : (
              <Bar dataKey={spec.yAxis} fill="hsl(210, 70%, 50%)" />
            )}
          </BarChart>
        </ResponsiveContainer>
      );

    case "composed":
      return (
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey={spec.xAxis}
              label={{ value: spec.xAxis, position: "right", offset: 10 }}
            />
            <YAxis
              label={{ value: spec.yAxis, angle: -90, position: "insideLeft" }}
            />
            <Tooltip />
            <Legend />
            {spec.series && spec.series.length > 0 ? (
              spec.series.map((series, idx) =>
                idx === 0 ? (
                  <Bar
                    key={idx}
                    dataKey={series}
                    fill={`hsl(${(idx * 360) / spec.series!.length}, 70%, 50%)`}
                  />
                ) : (
                  <Line
                    key={idx}
                    type="monotone"
                    dataKey={series}
                    stroke={`hsl(${(idx * 360) / spec.series!.length}, 70%, 50%)`}
                  />
                )
              )
            ) : (
              <Bar dataKey={spec.yAxis} fill="hsl(210, 70%, 50%)" />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      );

    default:
      return (
        <div className="flex items-center justify-center h-64 text-gray-500">
          Unsupported chart type: {spec.type}
        </div>
      );
  }
}
