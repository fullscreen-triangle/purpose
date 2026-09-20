"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

type NodeKind = "main" | "profile" | "content" | "route";

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  kind: NodeKind;
  detail: string;
}

interface GraphLink {
  source: string;
  target: string;
  label: string;
}

const CONTENT_STORE_NODES: GraphNode[] = [
  { id: "big-model", label: "One Big Model", kind: "main", detail: "A single model trained on, or retrieving from, everyone's raw documents." },
  { id: "doc-1", label: "Doc A", kind: "content", detail: "Raw content, embedded directly." },
  { id: "doc-2", label: "Doc B", kind: "content", detail: "Raw content, embedded directly." },
  { id: "doc-3", label: "Doc C", kind: "content", detail: "Raw content, embedded directly." },
  { id: "doc-4", label: "Doc D", kind: "content", detail: "Raw content, embedded directly." },
];

const CONTENT_STORE_LINKS: GraphLink[] = [
  { source: "doc-1", target: "big-model", label: "content in" },
  { source: "doc-2", target: "big-model", label: "content in" },
  { source: "doc-3", target: "big-model", label: "content in" },
  { source: "doc-4", target: "big-model", label: "content in" },
];

const FEDERATION_NODES: GraphNode[] = [
  { id: "main", label: "Main Model", kind: "main", detail: "Holds a route graph of closure-shapes, never any profile's content." },
  { id: "p1", label: "Profile A", kind: "profile", detail: "Its own model, trained on its own documents, never shared." },
  { id: "p2", label: "Profile B", kind: "profile", detail: "Its own model, trained on its own documents, never shared." },
  { id: "p3", label: "Profile C", kind: "profile", detail: "Its own model, trained on its own documents, never shared." },
  { id: "p4", label: "Profile D", kind: "profile", detail: "Its own model, trained on its own documents, never shared." },
];

const FEDERATION_LINKS: GraphLink[] = [
  { source: "main", target: "p1", label: "which profile, not what's in it" },
  { source: "main", target: "p2", label: "which profile, not what's in it" },
  { source: "main", target: "p3", label: "which profile, not what's in it" },
  { source: "main", target: "p4", label: "which profile, not what's in it" },
];

const COLORS: Record<NodeKind, string> = {
  main: "#B63E96",
  profile: "#58E6D9",
  content: "#888888",
  route: "#f5f5f5",
};

function GraphPane({
  title,
  subtitle,
  nodes: nodeData,
  links: linkData,
}: {
  title: string;
  subtitle: string;
  nodes: GraphNode[];
  links: GraphLink[];
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hovered, setHovered] = useState<GraphNode | null>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 360;
    const height = 320;

    const nodes: GraphNode[] = nodeData.map((n) => ({ ...n }));
    const links = linkData.map((l) => ({ ...l }));

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3
          .forceLink<GraphNode, d3.SimulationLinkDatum<GraphNode>>(links as unknown as d3.SimulationLinkDatum<GraphNode>[])
          .id((d) => (d as GraphNode).id)
          .distance(110),
      )
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide(34));

    const link = svg
      .append("g")
      .attr("stroke", "#555")
      .attr("stroke-opacity", 0.5)
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", 1.4);

    const node = svg
      .append("g")
      .selectAll<SVGCircleElement, GraphNode>("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d) => (d.kind === "main" ? 22 : 16))
      .attr("fill", (d) => COLORS[d.kind])
      .attr("stroke", "#1b1b1b")
      .attr("stroke-width", 1.5)
      .style("cursor", "pointer")
      .on("mouseenter", (_event, d) => setHovered(d))
      .on("mouseleave", () => setHovered(null))
      .call(
        d3
          .drag<SVGCircleElement, GraphNode>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }),
      );

    const label = svg
      .append("g")
      .selectAll("text")
      .data(nodes)
      .join("text")
      .text((d) => d.label)
      .attr("font-size", 10)
      .attr("fill", "#e8e8e8")
      .attr("text-anchor", "middle")
      .attr("dy", (d) => (d.kind === "main" ? 36 : 30))
      .style("pointer-events", "none");

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as unknown as GraphNode).x ?? 0)
        .attr("y1", (d) => (d.source as unknown as GraphNode).y ?? 0)
        .attr("x2", (d) => (d.target as unknown as GraphNode).x ?? 0)
        .attr("y2", (d) => (d.target as unknown as GraphNode).y ?? 0);

      node.attr("cx", (d) => d.x ?? 0).attr("cy", (d) => d.y ?? 0);
      label.attr("x", (d) => d.x ?? 0).attr("y", (d) => d.y ?? 0);
    });

    return () => {
      simulation.stop();
    };
  }, [nodeData, linkData]);

  return (
    <div className="flex-1 rounded-lg border border-white/10 bg-white/5 p-4">
      <h3 className="text-sm font-semibold text-light">{title}</h3>
      <p className="mt-1 text-xs text-light/60">{subtitle}</p>
      <svg ref={svgRef} viewBox="0 0 360 320" className="mt-2 w-full" />
      <div className="mt-2 h-8 text-xs text-primaryDark">
        {hovered ? `${hovered.label}: ${hovered.detail}` : "Hover or drag a node."}
      </div>
    </div>
  );
}

export function ArchitectureGraph() {
  return (
    <div className="flex flex-col gap-4 md:flex-row">
      <GraphPane
        title="Content-store RAG"
        subtitle="Everyone's raw documents flow into one shared model."
        nodes={CONTENT_STORE_NODES}
        links={CONTENT_STORE_LINKS}
      />
      <GraphPane
        title="This architecture"
        subtitle="Only routing decisions cross the boundary — never content."
        nodes={FEDERATION_NODES}
        links={FEDERATION_LINKS}
      />
    </div>
  );
}
