"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

interface Stage {
  id: string;
  label: string;
  detail: string;
  theorem: string;
  color: string;
}

const STAGES: Stage[] = [
  {
    id: "user",
    label: "User",
    detail: "Adds papers, CSVs, JSON, and links to their own profile.",
    theorem: "Acquisition (Accountable Compilation, Constr. 4.1)",
    color: "#f5f5f5",
  },
  {
    id: "profile",
    label: "Profile",
    detail: "Trains a local receiver once, at its own extremal regime. Never leaves the profile.",
    theorem: "Inclusion Theorem",
    color: "#58E6D9",
  },
  {
    id: "route",
    label: "Route",
    detail: "The main model consults a route graph of closure-shapes — never content — to pick candidate profiles.",
    theorem: "Route-Graph Receiver Theorem (§Selection)",
    color: "#B63E96",
  },
  {
    id: "waterfill",
    label: "Water-fill",
    detail: "A fixed query budget is divided across the selected profiles by marginal informativeness.",
    theorem: "Water-Filling / Cascade Theorem (§Allocation)",
    color: "#B63E96",
  },
  {
    id: "query",
    label: "Query N profiles",
    detail: "Selected profiles are queried concurrently, each exposing only a column — never a decoded value.",
    theorem: "Negation-Only Exposure",
    color: "#58E6D9",
  },
  {
    id: "extinction",
    label: "Extinction-lock check",
    detail: "Each pair is checked for categorical indistinguishability — a binary condition, not a graded threshold.",
    theorem: "Partition Extinction (§Combination)",
    color: "#f4a261",
  },
  {
    id: "federate",
    label: "Federate locked subset",
    detail: "Only extinction-locked receivers are combined. Unlocked ones are excluded, never averaged in.",
    theorem: "Extinction-Locked Federation Floor",
    color: "#B63E96",
  },
  {
    id: "report",
    label: "Answer or decline",
    detail: "A floor-bounded answer with its residual honestly stated — or a decline, never a silent guess.",
    theorem: "Seam Theorem",
    color: "#f5f5f5",
  },
];

export function ArchitectureFlowDiagram() {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [active, setActive] = useState<Stage | null>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 900;
    const stageWidth = width / STAGES.length;
    const height = 180;
    const boxW = stageWidth - 24;
    const boxH = 70;
    const y = height / 2 - boxH / 2;

    // connecting arrows
    const defs = svg.append("defs");
    defs
      .append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9)
      .attr("refY", 5)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto-start-reverse")
      .append("path")
      .attr("d", "M 0 0 L 10 5 L 0 10 z")
      .attr("fill", "#666");

    for (let i = 0; i < STAGES.length - 1; i++) {
      const x1 = i * stageWidth + boxW + 12;
      const x2 = (i + 1) * stageWidth + 12;
      svg
        .append("line")
        .attr("x1", x1)
        .attr("y1", height / 2)
        .attr("x2", x2 - 4)
        .attr("y2", height / 2)
        .attr("stroke", "#666")
        .attr("stroke-width", 1.5)
        .attr("marker-end", "url(#arrow)");
    }

    const groups = svg
      .selectAll("g.stage")
      .data(STAGES)
      .join("g")
      .attr("class", "stage")
      .attr("transform", (_, i) => `translate(${i * stageWidth + 12},${y})`)
      .style("cursor", "pointer")
      .on("mouseenter", (_event, d) => setActive(d))
      .on("click", (_event, d) => setActive(d));

    groups
      .append("rect")
      .attr("width", boxW)
      .attr("height", boxH)
      .attr("rx", 8)
      .attr("fill", (d) => d.color)
      .attr("fill-opacity", 0.18)
      .attr("stroke", (d) => d.color)
      .attr("stroke-width", 1.5);

    groups
      .append("text")
      .attr("x", boxW / 2)
      .attr("y", boxH / 2)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("font-size", 11)
      .attr("font-weight", 600)
      .attr("fill", "#f5f5f5")
      .style("pointer-events", "none")
      .each(function (d) {
        const words = d.label.split(" ");
        const el = d3.select(this);
        if (words.length > 1) {
          el.text("");
          el.append("tspan").attr("x", boxW / 2).attr("dy", "-0.3em").text(words.slice(0, Math.ceil(words.length / 2)).join(" "));
          el.append("tspan").attr("x", boxW / 2).attr("dy", "1.2em").text(words.slice(Math.ceil(words.length / 2)).join(" "));
        } else {
          el.text(d.label);
        }
      });
  }, []);

  return (
    <div className="rounded-lg border border-white/10 bg-white/5 p-5">
      <svg ref={svgRef} viewBox="0 0 900 180" className="w-full" />
      <div className="mt-3 min-h-[4.5rem] rounded border border-white/10 bg-black/20 p-3">
        {active ? (
          <>
            <p className="text-sm font-semibold" style={{ color: active.color }}>
              {active.label}
            </p>
            <p className="mt-1 text-xs text-light/70">{active.detail}</p>
            <p className="mt-1 text-xs text-primaryDark">{active.theorem}</p>
          </>
        ) : (
          <p className="text-xs text-light/40">Hover or click a stage for detail.</p>
        )}
      </div>
    </div>
  );
}
