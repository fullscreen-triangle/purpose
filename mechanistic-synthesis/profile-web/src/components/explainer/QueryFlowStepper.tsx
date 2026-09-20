"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

const STEPS = [
  { title: "1. Query arrives", body: "A question comes in to the main model." },
  { title: "2. Route", body: "The route graph picks which profiles have closed similar questions before — by shape, never by content." },
  { title: "3. Water-fill the budget", body: "A fixed query budget is divided across the selected profiles by how informative each one is." },
  { title: "4. Each profile eliminates", body: "Every profile rules out what the answer definitely isn't, rather than guessing what it is." },
  { title: "5. Phase-lock and combine", body: "Profiles that agree lock into phase; only the locked ones are federated into one answer." },
  { title: "6. Report, honestly", body: "The answer comes back with its residual stated — or a decline, never a silent guess." },
] as const;

// Illustrative receiver values (informativeness), used identically by both
// the water-fill bar chart and the Kuramoto phase circle so the two panels
// depict one consistent scenario as the stepper advances.
const RECEIVERS = [
  { id: "A", value: 3.1, cost: 1.0 },
  { id: "B", value: 2.4, cost: 1.2 },
  { id: "C", value: 1.6, cost: 0.8 },
  { id: "D", value: 0.9, cost: 1.4 },
  { id: "E", value: 0.5, cost: 1.0 },
] as const;

function waterFillAllocation(budget: number) {
  const density = RECEIVERS.map((r) => ({ ...r, density: r.value / r.cost }));
  density.sort((a, b) => b.density - a.density);
  let remaining = budget;
  return density.map((r) => {
    if (remaining <= 0) return { ...r, allocation: 0 };
    if (r.cost <= remaining) {
      remaining -= r.cost;
      return { ...r, allocation: 1 };
    }
    const frac = remaining / r.cost;
    remaining = 0;
    return { ...r, allocation: frac };
  });
}

function WaterFillChart({ budget, active }: { budget: number; active: boolean }) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const allocation = useMemo(() => waterFillAllocation(budget), [budget]);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    const width = 320;
    const height = 160;
    const margin = { top: 10, right: 10, bottom: 24, left: 28 };

    const x = d3
      .scaleBand()
      .domain(allocation.map((d) => d.id))
      .range([margin.left, width - margin.right])
      .padding(0.25);
    const y = d3.scaleLinear().domain([0, 1]).range([height - margin.bottom, margin.top]);

    svg
      .append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x))
      .attr("font-size", 10)
      .attr("color", "#aaa");

    svg
      .append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%")))
      .attr("font-size", 10)
      .attr("color", "#aaa");

    svg
      .selectAll("rect")
      .data(allocation)
      .join("rect")
      .attr("x", (d) => x(d.id) ?? 0)
      .attr("width", x.bandwidth())
      .attr("y", (d) => y(d.allocation))
      .attr("height", (d) => y(0) - y(d.allocation))
      .attr("fill", (d) => (active ? "#58E6D9" : "#4a7a96"))
      .attr("rx", 2);
  }, [allocation, active]);

  return <svg ref={svgRef} viewBox="0 0 320 160" className="w-full" />;
}

function KuramotoCircle({ locked, active }: { locked: boolean; active: boolean }) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setTick((t) => t + 1), 120);
    return () => clearInterval(id);
  }, [active]);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    const size = 200;
    const r = 70;
    const cx = size / 2;
    const cy = size / 2;

    svg
      .append("circle")
      .attr("cx", cx)
      .attr("cy", cy)
      .attr("r", r)
      .attr("fill", "none")
      .attr("stroke", "#444")
      .attr("stroke-width", 1);

    const n = RECEIVERS.length;
    const targetPhase = -Math.PI / 2;
    const lockProgress = locked ? Math.min(1, tick / 12) : 0;

    const phases = RECEIVERS.map((_, i) => {
      const spread = ((i - (n - 1) / 2) / n) * Math.PI * 1.4;
      const scatter = locked ? spread * (1 - lockProgress) : spread + Math.sin(tick / 4 + i) * 0.15;
      return targetPhase + scatter;
    });

    svg
      .selectAll("circle.node")
      .data(phases)
      .join("circle")
      .attr("class", "node")
      .attr("cx", (p) => cx + r * Math.cos(p))
      .attr("cy", (p) => cy + r * Math.sin(p))
      .attr("r", 7)
      .attr("fill", locked && lockProgress > 0.85 ? "#58E6D9" : "#B63E96")
      .attr("stroke", "#1b1b1b")
      .attr("stroke-width", 1);

    // order-parameter arrow
    const meanX = d3.mean(phases, (p) => Math.cos(p)) ?? 0;
    const meanY = d3.mean(phases, (p) => Math.sin(p)) ?? 0;
    const R = Math.sqrt(meanX * meanX + meanY * meanY);
    svg
      .append("line")
      .attr("x1", cx)
      .attr("y1", cy)
      .attr("x2", cx + r * R * (meanX / (R || 1)))
      .attr("y2", cy + r * R * (meanY / (R || 1)))
      .attr("stroke", "#f5f5f5")
      .attr("stroke-width", 2)
      .attr("marker-end", "url(#arrow)");

    svg
      .append("text")
      .attr("x", cx)
      .attr("y", size - 6)
      .attr("text-anchor", "middle")
      .attr("font-size", 10)
      .attr("fill", "#aaa")
      .text(`R = ${R.toFixed(2)}`);
  }, [tick, locked]);

  return <svg ref={svgRef} viewBox="0 0 200 200" className="mx-auto w-40" />;
}

export function QueryFlowStepper() {
  const [step, setStep] = useState(0);
  const [budget, setBudget] = useState(4.0);

  const current = STEPS[step] ?? STEPS[0];

  return (
    <div className="rounded-lg border border-white/10 bg-white/5 p-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-light">{current.title}</h3>
        <div className="flex gap-2">
          <button
            onClick={() => setStep((s) => Math.max(0, s - 1))}
            disabled={step === 0}
            className="rounded border border-white/20 px-2 py-1 text-xs text-light/80 disabled:opacity-30"
          >
            Back
          </button>
          <button
            onClick={() => setStep((s) => Math.min(STEPS.length - 1, s + 1))}
            disabled={step === STEPS.length - 1}
            className="rounded bg-primary px-2 py-1 text-xs text-white disabled:opacity-30"
          >
            Next
          </button>
        </div>
      </div>
      <p className="mt-1 text-xs text-light/60">{current.body}</p>

      {step === 2 && (
        <div className="mt-4">
          <label className="text-xs text-light/60">
            Total budget: {budget.toFixed(1)}
            <input
              type="range"
              min={1}
              max={6}
              step={0.2}
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="ml-3 w-48 align-middle"
            />
          </label>
          <div className="mt-2">
            <WaterFillChart budget={budget} active />
          </div>
        </div>
      )}

      {step >= 4 && (
        <div className="mt-4">
          <KuramotoCircle locked={step >= 4} active={step === 4} />
          <p className="text-center text-xs text-light/50">
            {step === 4 ? "Locking in progress…" : "Locked and federated."}
          </p>
        </div>
      )}

      {step === 5 && (
        <div className="mt-4 rounded border border-primaryDark/40 bg-primaryDark/10 p-3 text-xs text-light/80">
          Answer certified at residual ≤ τ. Two profiles did not lock and were excluded, not averaged in.
        </div>
      )}

      <div className="mt-4 flex gap-1">
        {STEPS.map((_, i) => (
          <div
            key={i}
            className={`h-1 flex-1 rounded ${i <= step ? "bg-primaryDark" : "bg-white/10"}`}
          />
        ))}
      </div>
    </div>
  );
}
