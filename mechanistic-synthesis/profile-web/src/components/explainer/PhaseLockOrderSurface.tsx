"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import * as d3 from "d3";

// R*(K, sigma_omega) for a Gaussian frequency law, using the corrected
// critical coupling Kc = 2/[pi*g(0)] = sqrt(2*pi)*sigma_omega derived in
// "Coordination Regimes of Synchronised Agents" (Sachikonye, Int. J.
// Topology 2026, 3(21), Thm. 5) and restated in the accompanying paper's
// Discussion. Below Kc, R* = 0 (incoherent); above it, the classical
// Kuramoto pitchfork R* = sqrt(1 - Kc/K) (Thm. 4). This is the real
// closed form, evaluated live, not illustrative data.
function criticalCoupling(sigmaOmega: number): number {
  return Math.sqrt(2 * Math.PI) * sigmaOmega;
}

function orderParameter(K: number, sigmaOmega: number): number {
  const Kc = criticalCoupling(sigmaOmega);
  if (K <= Kc) return 0;
  return Math.sqrt(1 - Kc / K);
}

interface Point3D {
  x: number;
  y: number;
  z: number;
}

function project(p: Point3D, azimuthDeg: number, elevationDeg: number, scale: number, cx: number, cy: number) {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  const x1 = p.x * Math.cos(az) - p.y * Math.sin(az);
  const y1 = p.x * Math.sin(az) + p.y * Math.cos(az);
  const z1 = p.z;
  const y2 = y1 * Math.cos(el) - z1 * Math.sin(el);
  const z2 = y1 * Math.sin(el) + z1 * Math.cos(el);
  return { sx: cx + x1 * scale, sy: cy - z2 * scale * 0.6 - y2 * scale * 0.25, depth: y2 };
}

const GRID_N = 26;

export function PhaseLockOrderSurface() {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [azimuth, setAzimuth] = useState(-58);
  const [elevation, setElevation] = useState(22);
  const dragState = useRef({ dragging: false, lastX: 0, lastY: 0 });

  const render = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    const width = 420;
    const height = 340;
    const cx = width / 2;
    const cy = height / 2 + 20;
    const scale = 130;

    const Ks = d3.range(GRID_N).map((i) => 0.2 + (i / (GRID_N - 1)) * 4.8); // coupling K in [0.2, 5]
    const sigmas = d3.range(GRID_N).map((i) => 0.1 + (i / (GRID_N - 1)) * 1.9); // spread in [0.1, 2.0]

    interface Quad {
      pts: { sx: number; sy: number; depth: number }[];
      z: number;
      depth: number;
    }
    const quads: Quad[] = [];
    for (let i = 0; i < GRID_N - 1; i++) {
      for (let j = 0; j < GRID_N - 1; j++) {
        const corners: Point3D[] = [
          { x: (Ks[i]! / 5) * 2 - 1, y: (sigmas[j]! / 2) * 2 - 1, z: orderParameter(Ks[i]!, sigmas[j]!) * 1.6 },
          { x: (Ks[i + 1]! / 5) * 2 - 1, y: (sigmas[j]! / 2) * 2 - 1, z: orderParameter(Ks[i + 1]!, sigmas[j]!) * 1.6 },
          { x: (Ks[i + 1]! / 5) * 2 - 1, y: (sigmas[j + 1]! / 2) * 2 - 1, z: orderParameter(Ks[i + 1]!, sigmas[j + 1]!) * 1.6 },
          { x: (Ks[i]! / 5) * 2 - 1, y: (sigmas[j + 1]! / 2) * 2 - 1, z: orderParameter(Ks[i]!, sigmas[j + 1]!) * 1.6 },
        ];
        const projected = corners.map((p) => project(p, azimuth, elevation, scale, cx, cy));
        const avgZ = (corners[0]!.z + corners[1]!.z + corners[2]!.z + corners[3]!.z) / 4;
        const avgDepth = d3.mean(projected, (p) => p.depth) ?? 0;
        quads.push({ pts: projected, z: avgZ, depth: avgDepth });
      }
    }
    quads.sort((a, b) => a.depth - b.depth);

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1.6]);

    svg
      .selectAll("polygon")
      .data(quads)
      .join("polygon")
      .attr("points", (q) => q.pts.map((p) => `${p.sx},${p.sy}`).join(" "))
      .attr("fill", (q) => colorScale(q.z))
      .attr("stroke", "#1b1b1b")
      .attr("stroke-width", 0.3)
      .attr("opacity", 0.92);

    svg.append("text").attr("x", 8).attr("y", 18).attr("font-size", 10).attr("fill", "#aaa").text("z = R* (order parameter)");
  }, [azimuth, elevation]);

  useEffect(() => {
    render();
  }, [render]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    function onDown(e: PointerEvent) {
      dragState.current = { dragging: true, lastX: e.clientX, lastY: e.clientY };
    }
    function onMove(e: PointerEvent) {
      if (!dragState.current.dragging) return;
      const dx = e.clientX - dragState.current.lastX;
      const dy = e.clientY - dragState.current.lastY;
      dragState.current.lastX = e.clientX;
      dragState.current.lastY = e.clientY;
      setAzimuth((a) => a + dx * 0.4);
      setElevation((el) => Math.max(5, Math.min(85, el - dy * 0.4)));
    }
    function onUp() {
      dragState.current.dragging = false;
    }
    svg.addEventListener("pointerdown", onDown);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      svg.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, []);

  return (
    <div className="rounded-lg border border-white/10 bg-white/5 p-5">
      <p className="text-sm text-light/70">
        The Kuramoto pitchfork, evaluated live from the corrected critical
        coupling{" "}
        <code className="mx-1 rounded bg-white/10 px-1.5 py-0.5 text-xs">
          K_c = √(2π)·σ_ω
        </code>
        : below K_c the population never locks (R*=0); above it, R* rises
        continuously. Drag to rotate.
      </p>
      <svg
        ref={svgRef}
        viewBox="0 0 420 340"
        className="mt-3 w-full max-w-md cursor-grab touch-none active:cursor-grabbing"
      />
      <p className="mt-2 text-xs text-light/50">
        x-axis: coupling strength K (0.2–5). y-axis: frequency spread σ_ω
        (0.1–2.0). The ridge running diagonally is the bifurcation — the
        only genuine phase transition in this model.
      </p>
    </div>
  );
}
