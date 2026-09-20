"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";

// Real numbers from absicht/docs/distributed-domain-route-graph's own
// validation suite (validation/results/exp04_generate_test.json), not
// illustrative placeholders: of 30 trials where a profile's existing
// negation set was exhausted, 70% were resolved by generating and testing
// a fresh candidate column; 30% correctly declined rather than guessing;
// 0 trials silently reported an answer without satisfying one of the two.
const RESOLVED_RATE = 0.7;
const DECLINED_RATE = 0.3;
const DICHOTOMY_VIOLATIONS = 0;
const N_TRIALS = 30;

function DichotomyChart() {
  const ref = useRef<SVGSVGElement | null>(null);
  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    const width = 280;
    const height = 90;
    const data = [
      { label: "Resolved (found an answer)", value: RESOLVED_RATE, color: "#58E6D9" },
      { label: "Declined (said “don't know”)", value: DECLINED_RATE, color: "#B63E96" },
    ];
    const x = d3.scaleLinear().domain([0, 1]).range([90, width - 10]);
    const y = d3
      .scaleBand()
      .domain(data.map((d) => d.label))
      .range([4, height - 4])
      .padding(0.35);

    svg
      .selectAll("rect")
      .data(data)
      .join("rect")
      .attr("x", x(0))
      .attr("y", (d) => y(d.label) ?? 0)
      .attr("width", (d) => x(d.value) - x(0))
      .attr("height", y.bandwidth())
      .attr("fill", (d) => d.color)
      .attr("rx", 2);

    svg
      .selectAll("text.label")
      .data(data)
      .join("text")
      .attr("class", "label")
      .attr("x", 86)
      .attr("y", (d) => (y(d.label) ?? 0) + y.bandwidth() / 2 + 4)
      .attr("text-anchor", "end")
      .attr("font-size", 9)
      .attr("fill", "#ccc")
      .text((d) => d.label);

    svg
      .selectAll("text.value")
      .data(data)
      .join("text")
      .attr("class", "value")
      .attr("x", (d) => x(d.value) + 4)
      .attr("y", (d) => (y(d.label) ?? 0) + y.bandwidth() / 2 + 4)
      .attr("font-size", 10)
      .attr("fill", "#fff")
      .text((d) => `${(d.value * 100).toFixed(0)}%`);
  }, []);
  return <svg ref={ref} viewBox="0 0 280 90" className="w-full max-w-sm" />;
}

export function HonestyPanel() {
  return (
    <div className="rounded-lg border border-white/10 bg-white/5 p-5">
      <p className="text-sm text-light/80">
        Every profile, and the main model itself, will sometimes say{" "}
        <span className="text-primaryDark">&ldquo;I don&rsquo;t know&rdquo;</span> rather than
        guess. This is provable, not a limitation we&rsquo;re hiding: when a profile&rsquo;s
        existing knowledge can&rsquo;t rule out enough candidates to answer confidently, it either
        generates and tests a new candidate or declines — never a silent guess.
      </p>
      <div className="mt-4">
        <DichotomyChart />
      </div>
      <p className="mt-2 text-xs text-light/50">
        From {N_TRIALS} real trials in the accompanying paper&rsquo;s validation suite: when a
        profile&rsquo;s negation set was deliberately exhausted, generating a fresh candidate
        resolved it {Math.round(RESOLVED_RATE * 100)}% of the time; the rest correctly declined.
        Zero trials violated the dichotomy — {DICHOTOMY_VIOLATIONS} silent guesses.
      </p>
    </div>
  );
}
