"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

function MiniLineChart({ data, color }: { data: number[]; color: string }) {
  const ref = useRef<SVGSVGElement | null>(null);
  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    const width = 200;
    const height = 60;
    const x = d3.scaleLinear().domain([0, data.length - 1]).range([4, width - 4]);
    const y = d3
      .scaleLinear()
      .domain([Math.min(...data), Math.max(...data)])
      .range([height - 6, 6]);
    const line = d3
      .line<number>()
      .x((_, i) => x(i))
      .y((d) => y(d))
      .curve(d3.curveMonotoneX);
    svg
      .append("path")
      .datum(data)
      .attr("d", line)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 2);
  }, [data, color]);
  return <svg ref={ref} viewBox="0 0 200 60" className="w-full" />;
}

function MiniBarChart({ data, color }: { data: number[]; color: string }) {
  const ref = useRef<SVGSVGElement | null>(null);
  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    const width = 200;
    const height = 60;
    const x = d3
      .scaleBand()
      .domain(data.map((_, i) => String(i)))
      .range([4, width - 4])
      .padding(0.2);
    const y = d3.scaleLinear().domain([0, Math.max(...data)]).range([height - 6, 6]);
    svg
      .selectAll("rect")
      .data(data)
      .join("rect")
      .attr("x", (_, i) => x(String(i)) ?? 0)
      .attr("width", x.bandwidth())
      .attr("y", (d) => y(d))
      .attr("height", (d) => height - 6 - y(d))
      .attr("fill", color)
      .attr("rx", 1);
  }, [data, color]);
  return <svg ref={ref} viewBox="0 0 200 60" className="w-full" />;
}

const LAYERS = [
  {
    id: "routing",
    name: "Routing",
    theorem: "Route-Graph Receiver Theorem",
    summary: "The main model never sees profile content — only which profile-shape has closed similar questions before.",
    chart: "line" as const,
    data: [8, 5, 3, 2, 2, 2, 2, 2, 2, 2],
    color: "#58E6D9",
  },
  {
    id: "allocation",
    name: "Allocation & Sync",
    theorem: "Federated Phase-Lock Floor Theorem",
    summary: "A fixed budget is water-filled across the most informative profiles, then their responses are phase-locked before being combined.",
    chart: "bar" as const,
    data: [1.0, 1.0, 0.6, 0, 0],
    color: "#B63E96",
  },
  {
    id: "elimination",
    name: "Elimination",
    theorem: "Elimination-Answering Receiver Theorem",
    summary: "Each profile rules out candidates rather than asserting an answer, and reports what's left over — never a decoded value.",
    chart: "line" as const,
    data: [10, 8, 6, 5, 4, 4, 3, 3, 2, 2],
    color: "#f5f5f5",
  },
];

export function LayerCards() {
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <div className="grid gap-4 md:grid-cols-3">
      {LAYERS.map((layer) => {
        const isOpen = expanded === layer.id;
        return (
          <button
            key={layer.id}
            onClick={() => setExpanded(isOpen ? null : layer.id)}
            className="rounded-lg border border-white/10 bg-white/5 p-4 text-left transition hover:border-primaryDark/50"
          >
            <h4 className="text-sm font-semibold text-light">{layer.name}</h4>
            <p className="mt-1 text-xs text-primaryDark">{layer.theorem}</p>
            {isOpen && (
              <>
                <p className="mt-2 text-xs text-light/70">{layer.summary}</p>
                <div className="mt-3">
                  {layer.chart === "line" ? (
                    <MiniLineChart data={layer.data} color={layer.color} />
                  ) : (
                    <MiniBarChart data={layer.data} color={layer.color} />
                  )}
                </div>
              </>
            )}
            {!isOpen && <p className="mt-2 text-xs text-light/40">Click to expand</p>}
          </button>
        );
      })}
    </div>
  );
}
