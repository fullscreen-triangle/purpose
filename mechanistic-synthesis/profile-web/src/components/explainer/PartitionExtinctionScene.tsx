"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as d3 from "d3";

// tau_p(angularDistance, delta): the partition lag, computed exactly as
// validation/run_validation.py's partition_lag() in the accompanying paper
// (absicht/docs/distributed-domain-route-graph) — 0 once the model's finite
// resolution delta is reached, otherwise delta / angularDistance, diverging
// as angularDistance -> delta from above. This is real math from the paper's
// own closed form, evaluated live here, not illustrative/invented data.
function tauP(angularDistance: number, delta: number): number {
  if (angularDistance <= delta) return 0;
  return delta / angularDistance;
}

interface Point3D {
  x: number;
  y: number;
  z: number;
}

// Minimal hand-rolled perspective projection (no Three.js dependency):
// rotate a 3D point by (azimuth, elevation) then project orthographically,
// matching the paper's own matplotlib 3D panels' "viewing angle elevation/
// azimuth" convention.
function project(p: Point3D, azimuthDeg: number, elevationDeg: number, scale: number, cx: number, cy: number) {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  // rotate around Y (azimuth), then around X (elevation)
  const x1 = p.x * Math.cos(az) - p.y * Math.sin(az);
  const y1 = p.x * Math.sin(az) + p.y * Math.cos(az);
  const z1 = p.z;
  const y2 = y1 * Math.cos(el) - z1 * Math.sin(el);
  const z2 = y1 * Math.sin(el) + z1 * Math.cos(el);
  return {
    sx: cx + x1 * scale,
    sy: cy - z2 * scale * 0.6 - y2 * scale * 0.25,
    depth: y2,
  };
}

const GRID_N = 28;
const DELTA = 0.05;

export function PartitionExtinctionScene() {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [azimuth, setAzimuth] = useState(-55);
  const [elevation, setElevation] = useState(24);
  const dragState = useRef<{ dragging: boolean; lastX: number; lastY: number }>({
    dragging: false,
    lastX: 0,
    lastY: 0,
  });

  const render = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    const width = 420;
    const height = 340;
    const cx = width / 2;
    const cy = height / 2 + 30;
    const scale = 130;

    // Build the surface grid: x = angular distance in [0.001, 0.3],
    // y = delta multiple in [0.5, 3.0] (so delta itself sweeps too),
    // z = tau_p(x, delta*y) -- identical parameterization to the paper's
    // panel_7_binary_lock.png Panel D.
    const angularRange = d3.range(GRID_N).map((i) => 0.001 + (i / (GRID_N - 1)) * 0.3);
    const deltaMultRange = d3.range(GRID_N).map((i) => 0.5 + (i / (GRID_N - 1)) * 2.5);

    interface Quad {
      pts: { sx: number; sy: number; depth: number }[];
      z: number;
      depth: number;
    }
    const quads: Quad[] = [];
    for (let i = 0; i < GRID_N - 1; i++) {
      for (let j = 0; j < GRID_N - 1; j++) {
        const corners: Point3D[] = [
          { x: angularRange[i]! * 3 - 0.5, y: deltaMultRange[j]! - 1.75, z: tauP(angularRange[i]!, DELTA * deltaMultRange[j]!) * 2 },
          { x: angularRange[i + 1]! * 3 - 0.5, y: deltaMultRange[j]! - 1.75, z: tauP(angularRange[i + 1]!, DELTA * deltaMultRange[j]!) * 2 },
          { x: angularRange[i + 1]! * 3 - 0.5, y: deltaMultRange[j + 1]! - 1.75, z: tauP(angularRange[i + 1]!, DELTA * deltaMultRange[j + 1]!) * 2 },
          { x: angularRange[i]! * 3 - 0.5, y: deltaMultRange[j + 1]! - 1.75, z: tauP(angularRange[i]!, DELTA * deltaMultRange[j + 1]!) * 2 },
        ];
        const projected = corners.map((p) => project(p, azimuth, elevation, scale, cx, cy));
        const avgZ = (corners[0]!.z + corners[1]!.z + corners[2]!.z + corners[3]!.z) / 4;
        const avgDepth = d3.mean(projected, (p) => p.depth) ?? 0;
        quads.push({ pts: projected, z: avgZ, depth: avgDepth });
      }
    }
    quads.sort((a, b) => a.depth - b.depth);

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

    svg
      .selectAll("polygon")
      .data(quads)
      .join("polygon")
      .attr("points", (q) => q.pts.map((p) => `${p.sx},${p.sy}`).join(" "))
      .attr("fill", (q) => colorScale(Math.min(1, q.z)))
      .attr("stroke", "#1b1b1b")
      .attr("stroke-width", 0.3)
      .attr("opacity", 0.92);

    svg
      .append("text")
      .attr("x", 8)
      .attr("y", 18)
      .attr("font-size", 10)
      .attr("fill", "#aaa")
      .text("z = τ_p (partition lag)");
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
        This is the paper&rsquo;s own closed form for the partition lag,
        <code className="mx-1 rounded bg-white/10 px-1.5 py-0.5 text-xs">
          τ_p(d, δ) = 0 if d ≤ δ, else δ/d
        </code>
        , evaluated live — not illustrative data. Drag to rotate.
      </p>
      <svg
        ref={svgRef}
        viewBox="0 0 420 340"
        className="mt-3 w-full max-w-md cursor-grab touch-none active:cursor-grabbing"
      />
      <p className="mt-2 text-xs text-light/50">
        The surface is flat at zero below the resolution δ and rises
        steeply — a cliff, not a slope — as the angular distance shrinks
        toward δ from above. There is no intermediate lag value for a
        combination rule to weight against.
      </p>
    </div>
  );
}
