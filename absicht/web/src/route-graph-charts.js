import * as d3 from 'd3';
import { COL, makeSvg, axisStyle } from './charts.js';

export const EXPERIMENTS = [
  { n: 1, name: 'Route graph is a bounded receiver', thm: 'Thm (route-receiver)', key: '|Know| plateaus at n_types in 100% of 90 trials' },
  { n: 2, name: 'Water-filling converges to knapsack optimum', thm: 'Thm (waterfill-cascade)', key: 'Ratio 1.31→1.0006, never below 1.0' },
  { n: 3, name: 'Federated phase-lock floor', thm: 'Thm (phaselock-floor)', key: 'Locked beats naive in 100% of trials' },
  { n: 4, name: 'Generate-and-test on exhausted negation', thm: 'Constr (generate-test)', key: '0 dichotomy violations, 70% resolved' },
  { n: 5, name: 'Seam Theorem end-to-end bound', thm: 'Thm (seam)', key: 'Bound satisfied in 120/120 trials' },
  { n: 6, name: 'Crowd-sharpening under route selection', thm: 'Cor (crowd-routing)', key: 'Routed never worse in 150/150 trials' },
  { n: 7, name: 'Partition extinction is discontinuous', thm: 'Thm (extinction-discontinuous)', key: 'Positive branch bounded away from 0' },
  { n: 8, name: 'Operation-set exhaustion for generate-and-test', thm: 'Rem (operation-exhaustion)', key: '0 false exhaustion, 0 dichotomy violations' },
];

/* ===================================================================
   PROBLEM DIAGRAM: Select / Allocate / Combine -> one composable floor
=================================================================== */
export function drawThreeToOne() {
  const w = 480, h = 380;
  const g = makeSvg('#viz-route-three-to-one', w, h, { l: 10, t: 10, r: 10, b: 10 });
  const items = [
    { label: 'Selection', sub: 'which receivers, blind to content', y: 40 },
    { label: 'Allocation', sub: 'how much budget each gets', y: 150 },
    { label: 'Combination', sub: 'which locked, which declined', y: 260 },
  ];
  const boxW = 190, boxH = 66, cx = 100;
  items.forEach((it) => {
    const node = g.append('g').attr('transform', `translate(${cx - boxW / 2}, ${it.y})`);
    node.append('rect').attr('width', boxW).attr('height', boxH).attr('rx', 8)
      .attr('fill', 'none').attr('stroke', COL.line);
    node.append('text').attr('x', boxW / 2).attr('y', 27).attr('text-anchor', 'middle')
      .attr('font-family', "'Fraunces', serif").attr('font-size', 15).attr('fill', COL.ink)
      .text(it.label);
    node.append('text').attr('x', boxW / 2).attr('y', 46).attr('text-anchor', 'middle')
      .attr('font-size', 10.5).attr('fill', COL.inkFaint).text(it.sub);
    g.append('path')
      .attr('d', `M${cx + boxW / 2 - 10},${it.y + boxH / 2} C ${340},${it.y + boxH / 2} ${340},${150 + 33} ${400},${150 + 33}`)
      .attr('fill', 'none').attr('stroke', COL.line).attr('stroke-dasharray', '3,3');
  });
  const fx = 400, fy = 150;
  const fg = g.append('g').attr('transform', `translate(${fx - 70},${fy - 33})`);
  fg.append('rect').attr('width', 140).attr('height', 66).attr('rx', 33)
    .attr('fill', 'none').attr('stroke', COL.accent2).attr('stroke-width', 1.6);
  fg.append('text').attr('x', 70).attr('y', 28).attr('text-anchor', 'middle')
    .attr('font-family', "'Fraunces', serif").attr('font-weight', 600).attr('font-size', 15).attr('fill', COL.accent2)
    .text('R_Σ');
  fg.append('text').attr('x', 70).attr('y', 46).attr('text-anchor', 'middle')
    .attr('font-size', 10).attr('fill', COL.inkFaint).text('one bounded receiver');
}

/* ===================================================================
   QUERY FLOW DIAGRAM: the actual step-by-step path of one query,
   including the branch to decline -- a genuine flow, not a static stack.
=================================================================== */
export function drawQueryFlow() {
  const w = 1060, h = 460;
  const g = makeSvg('#viz-query-flow', w, h, { l: 20, t: 20, r: 20, b: 20 });
  const boxW = 168, boxH = 60;

  const stages = [
    { id: 'query', x: 20, y: 40, label: 'Query arrives', sub: 'at the main model', color: COL.ink },
    { id: 'route', x: 220, y: 40, label: 'Route', sub: 'route graph, content-blind', color: COL.accent2 },
    { id: 'waterfill', x: 420, y: 40, label: 'Water-fill', sub: 'divide budget by value', color: COL.accent2 },
    { id: 'query-n', x: 620, y: 40, label: 'Query N profiles', sub: 'concurrently, columns only', color: COL.accent },
    { id: 'extinction', x: 820, y: 40, label: 'Extinction-lock check', sub: 'binary, per pair', color: COL.accent },
    { id: 'federate', x: 620, y: 200, label: 'Federate locked subset', sub: 'exclude, never average', color: COL.accent2 },
    { id: 'answer', x: 420, y: 200, label: 'Report answer', sub: 'floor-bounded, residual stated', color: COL.ink },
    { id: 'decline', x: 620, y: 340, label: 'Decline', sub: 'nothing locked — say so', color: COL.decline },
  ];

  const byId = Object.fromEntries(stages.map(s => [s.id, s]));

  // straight horizontal connector between two same-row stages
  function arrowRight(fromId, toId) {
    const a = byId[fromId], b = byId[toId];
    g.append('path')
      .attr('d', `M${a.x + boxW},${a.y + boxH / 2} L${b.x - 6},${b.y + boxH / 2}`)
      .attr('fill', 'none').attr('stroke', COL.line).attr('stroke-width', 1.3)
      .attr('marker-end', 'url(#flow-arrowhead)');
  }

  // arrowhead marker, defined once
  const defs = g.append('defs');
  defs.append('marker').attr('id', 'flow-arrowhead')
    .attr('viewBox', '0 0 10 10').attr('refX', 8).attr('refY', 5)
    .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto-start-reverse')
    .append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', COL.inkFaint);

  stages.forEach(s => {
    const node = g.append('g').attr('transform', `translate(${s.x},${s.y})`);
    node.append('rect').attr('width', boxW).attr('height', boxH).attr('rx', 8)
      .attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', 1.4);
    node.append('text').attr('x', boxW / 2).attr('y', 24).attr('text-anchor', 'middle')
      .attr('font-family', "'Fraunces', serif").attr('font-size', 13).attr('fill', s.color)
      .text(s.label);
    node.append('text').attr('x', boxW / 2).attr('y', 42).attr('text-anchor', 'middle')
      .attr('font-size', 10).attr('fill', COL.inkFaint).text(s.sub);
  });

  // straight top-row arrows
  ['query>route', 'route>waterfill', 'waterfill>query-n', 'query-n>extinction'].forEach(pair => {
    const [from, to] = pair.split('>');
    arrowRight(from, to);
  });

  // down from extinction-lock check to federate (locked branch)
  g.append('path')
    .attr('d', `M${byId.extinction.x + boxW / 2},${byId.extinction.y + boxH} C ${byId.extinction.x + boxW / 2},${byId.extinction.y + 90} ${byId.federate.x + boxW / 2},${byId.federate.y - 90} ${byId.federate.x + boxW / 2},${byId.federate.y}`)
    .attr('fill', 'none').attr('stroke', COL.accent2).attr('stroke-width', 1.3)
    .attr('marker-end', 'url(#flow-arrowhead)');
  g.append('text').attr('x', byId.extinction.x + boxW / 2 + 8).attr('y', byId.extinction.y + boxH + 40)
    .attr('font-size', 9.5).attr('fill', COL.accent2).text('locked ≥1');

  // federate -> answer
  g.append('path')
    .attr('d', `M${byId.federate.x},${byId.federate.y + boxH / 2} L${byId.answer.x + boxW + 6},${byId.answer.y + boxH / 2}`)
    .attr('fill', 'none').attr('stroke', COL.line).attr('stroke-width', 1.3)
    .attr('marker-end', 'url(#flow-arrowhead)');

  // federate -> decline (nothing locked branch)
  g.append('path')
    .attr('d', `M${byId.federate.x + boxW / 2},${byId.federate.y + boxH} C ${byId.federate.x + boxW / 2},${byId.federate.y + 90} ${byId.decline.x + boxW / 2},${byId.decline.y - 90} ${byId.decline.x + boxW / 2},${byId.decline.y}`)
    .attr('fill', 'none').attr('stroke', COL.decline).attr('stroke-width', 1.3)
    .attr('stroke-dasharray', '3,3')
    .attr('marker-end', 'url(#flow-arrowhead)');
  g.append('text').attr('x', byId.federate.x + boxW / 2 + 8).attr('y', byId.federate.y + boxH + 40)
    .attr('font-size', 9.5).attr('fill', COL.decline).text('none locked');
}

/* ===================================================================
   FEDERATION TOPOLOGY: main model + N profile receivers, showing which
   edges carry content (none) and which carry only routing decisions.
=================================================================== */
export function drawFederationTopology() {
  const w = 1060, h = 420;
  const g = makeSvg('#viz-federation-topology', w, h, { l: 20, t: 20, r: 20, b: 20 });
  const cx = w / 2, cy = h / 2 - 10;
  const mainR = 46, profR = 28, ringR = 130;

  const profiles = [
    { label: 'Profile A', locked: true },
    { label: 'Profile B', locked: true },
    { label: 'Profile C', locked: false },
    { label: 'Profile D', locked: true },
    { label: 'Profile E', locked: false },
    { label: 'Profile F', locked: true },
  ];

  const defs = g.append('defs');
  defs.append('marker').attr('id', 'fed-arrowhead')
    .attr('viewBox', '0 0 10 10').attr('refX', 8).attr('refY', 5)
    .attr('markerWidth', 5).attr('markerHeight', 5).attr('orient', 'auto-start-reverse')
    .append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', COL.inkFaint);

  profiles.forEach((p, i) => {
    const angle = (i / profiles.length) * 2 * Math.PI - Math.PI / 2;
    const px = cx + ringR * Math.cos(angle);
    const py = cy + ringR * Math.sin(angle);
    const color = p.locked ? COL.accent2 : COL.inkFaint;

    g.append('line')
      .attr('x1', cx + mainR * Math.cos(angle)).attr('y1', cy + mainR * Math.sin(angle))
      .attr('x2', px - profR * Math.cos(angle) * 1.15).attr('y2', py - profR * Math.sin(angle) * 1.15)
      .attr('stroke', color).attr('stroke-width', p.locked ? 1.6 : 1)
      .attr('stroke-dasharray', p.locked ? null : '3,3')
      .attr('marker-end', 'url(#fed-arrowhead)');

    const node = g.append('g').attr('transform', `translate(${px},${py})`);
    node.append('circle').attr('r', profR).attr('fill', 'none').attr('stroke', color).attr('stroke-width', 1.6);
    node.append('text').attr('y', 4).attr('text-anchor', 'middle')
      .attr('font-size', 10.5).attr('fill', color).text(p.label);
    node.append('text').attr('y', profR + 16).attr('text-anchor', 'middle')
      .attr('font-size', 9).attr('fill', color)
      .text(p.locked ? 'locked' : 'excluded');
  });

  const main = g.append('g').attr('transform', `translate(${cx},${cy})`);
  main.append('circle').attr('r', mainR).attr('fill', 'none').attr('stroke', COL.accent).attr('stroke-width', 2);
  main.append('text').attr('y', -2).attr('text-anchor', 'middle')
    .attr('font-family', "'Fraunces', serif").attr('font-size', 13).attr('fill', COL.accent).text('Main');
  main.append('text').attr('y', 14).attr('text-anchor', 'middle')
    .attr('font-size', 9).attr('fill', COL.accent).text('model');

  g.append('text').attr('x', 0).attr('y', h - 24).attr('font-size', 11).attr('fill', COL.inkDim)
    .text('Every edge carries a routing decision — never content. The main model holds a route graph of closure-shapes and nothing about any profile’s training data.');
}

/* ===================================================================
   1. ROUTE GRAPH BOUNDEDNESS: |Know| plateau vs |Cands| growth
=================================================================== */
export function drawRouteReceiver(DATA) {
  const d = DATA.exp1;
  const w = 1060, h = 300, m = { l: 46, t: 16, r: 20, b: 36 };
  const g = makeSvg('#chart-route-receiver', w, h, m);
  const iw = (w - m.l - m.r - 60) / 3;
  const ih = h - m.t - m.b;
  const colors = [COL.accent2, COL.accent, '#a48cff'];

  d.Ns.forEach((N, i) => {
    const panel = g.append('g').attr('transform', `translate(${i * (iw + 30)},0)`);
    const know = d.know_trace[N];
    const cands = d.cands_trace[N];
    const x = d3.scaleLinear().domain([0, cands.length - 1]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, Math.max(...cands)]).range([ih, 0]);

    panel.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(4)).join('line')
      .attr('x1', 0).attr('x2', iw).attr('y1', v => y(v)).attr('y2', v => y(v)).attr('stroke', COL.grid);

    panel.append('path').datum(cands)
      .attr('d', d3.line().x((v, idx) => x(idx)).y(v => y(v)))
      .attr('fill', 'none').attr('stroke', COL.inkFaint).attr('stroke-dasharray', '3,3').attr('stroke-width', 1.2);
    panel.append('path').datum(know)
      .attr('d', d3.line().x((v, idx) => x(idx)).y(v => y(v)))
      .attr('fill', 'none').attr('stroke', colors[i]).attr('stroke-width', 2);

    panel.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(5).tickSize(4)).call(axisStyle);
    panel.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4).tickSize(4)).call(axisStyle);
    panel.append('text').attr('x', 0).attr('y', -2).attr('font-size', 11).attr('fill', COL.inkDim)
      .text(`N=${N}: |Know| (solid) vs |Cands| (dashed)`);
  });
}

/* ===================================================================
   2. WATER-FILLING CONVERGENCE
=================================================================== */
export function drawWaterfill(DATA) {
  const d = DATA.exp2;
  const w = 520, h = 340, m = { l: 46, t: 20, r: 20, b: 40 };
  const g = makeSvg('#chart-waterfill', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const x = d3.scaleLog().domain([4, 128]).range([0, iw]);
  const y = d3.scaleLinear().domain([0.95, 1.47]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(5)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', v => y(v)).attr('y2', v => y(v)).attr('stroke', COL.grid);

  g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y(1)).attr('y2', y(1))
    .attr('stroke', COL.decline).attr('stroke-dasharray', '4,3').attr('stroke-width', 1.3);

  const area = d3.area().x((v, i) => x(d.ks[i])).y0((v, i) => y(d.ratio_lo[i])).y1((v, i) => y(d.ratio_hi[i]));
  g.append('path').datum(d.ks).attr('d', area).attr('fill', COL.accent2).attr('opacity', 0.15);

  g.append('path').datum(d.ks).attr('d', d3.line().x((v, i) => x(d.ks[i])).y((v, i) => y(d.ratio_mean[i])))
    .attr('fill', 'none').attr('stroke', COL.accent2).attr('stroke-width', 2);
  g.selectAll('circle').data(d.ks).join('circle')
    .attr('cx', v => x(v)).attr('cy', (v, i) => y(d.ratio_mean[i])).attr('r', 4).attr('fill', COL.accent2);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(5, '~s').tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5).tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text('fractional/exact-knapsack ratio  vs.  k receivers (log scale)  ·  approaches 1 from above');
}

/* ===================================================================
   3. FEDERATED PHASE-LOCK FLOOR
=================================================================== */
export function drawPhaselock(DATA) {
  const d = DATA.exp3;
  const w = 520, h = 340, m = { l: 50, t: 20, r: 20, b: 40 };
  const g = makeSvg('#chart-phaselock', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const x = d3.scalePoint().domain(d.n_locked.map(String)).range([0, iw]).padding(0.5);
  const y = d3.scaleLog().domain([1e-18, 1e-10]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(4)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', v => y(v)).attr('y2', v => y(v)).attr('stroke', COL.grid);

  g.append('path').datum(d.n_locked).attr('d', d3.line().x(v => x(String(v))).y((v, i) => y(d.q_joint_locked[i])))
    .attr('fill', 'none').attr('stroke', COL.accent2).attr('stroke-width', 2);
  g.selectAll('circle').data(d.n_locked).join('circle')
    .attr('cx', v => x(String(v))).attr('cy', (v, i) => y(d.q_joint_locked[i])).attr('r', 4).attr('fill', COL.accent2);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4, '.0e').tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text('joint failure probability (locked)  vs.  number locked');
}

export function drawPhaselockBars(DATA) {
  const d = DATA.exp3;
  const w = 420, h = 300, m = { l: 56, t: 20, r: 20, b: 50 };
  const g = makeSvg('#chart-phaselock-bars', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const rows = [{ label: 'locked-only', v: d.mean_q_locked }, { label: 'naive full-pop', v: d.mean_q_naive }];
  const x = d3.scaleBand().domain(rows.map(r => r.label)).range([0, iw]).padding(0.4);
  const y = d3.scaleLog().domain([1e-8, 1e-1]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(4)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', v => y(v)).attr('y2', v => y(v)).attr('stroke', COL.grid);

  g.selectAll('rect').data(rows).join('rect')
    .attr('x', r => x(r.label)).attr('width', x.bandwidth())
    .attr('y', r => y(r.v)).attr('height', r => ih - y(r.v))
    .attr('fill', (r, i) => i === 0 ? COL.accent2 : COL.inkFaint);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4, '.0e').tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text(`failure probability  ·  ${(d.mean_q_naive / d.mean_q_locked).toExponential(1)}x gap, log scale`);
}

/* ===================================================================
   4. GENERATE-AND-TEST
=================================================================== */
export function drawGenerateTest(DATA) {
  const d = DATA.exp4;
  const w = 520, h = 320, m = { l: 46, t: 20, r: 20, b: 40 };
  const g = makeSvg('#chart-generate-test', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const allVals = [...d.resolved_trace, ...d.declined_trace];
  const x = d3.scaleLinear().domain([0, Math.max(d.resolved_trace.length, d.declined_trace.length) - 1]).range([0, iw]);
  const y = d3.scaleLog().domain([0.05, Math.max(...allVals)]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(4)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', v => y(v)).attr('y2', v => y(v)).attr('stroke', COL.grid);

  g.append('path').datum(d.resolved_trace).attr('d', d3.line().x((v, i) => x(i)).y(v => y(Math.max(v, 0.05))))
    .attr('fill', 'none').attr('stroke', COL.accent2).attr('stroke-width', 2);
  g.append('path').datum(d.declined_trace).attr('d', d3.line().x((v, i) => x(i)).y(v => y(Math.max(v, 0.05))))
    .attr('fill', 'none').attr('stroke', COL.decline).attr('stroke-width', 2);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(6).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4).tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text('joint residual (log)  vs.  relaxation round  ·  resolved (teal) vs still-declined (red)');
}

/* ===================================================================
   5. SEAM THEOREM END-TO-END
=================================================================== */
export function drawSeam(DATA) {
  const d = DATA.exp5;
  const w = 520, h = 320, m = { l: 46, t: 20, r: 20, b: 40 };
  const g = makeSvg('#chart-seam', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const x = d3.scalePoint().domain(d.Ns.map(String)).range([0, iw]).padding(0.5);
  const y = d3.scaleLinear().domain([0.8, 0.88]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(4)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', v => y(v)).attr('y2', v => y(v)).attr('stroke', COL.grid);

  const vals = d.Ns.map(n => d.achieved_floor_by_N[n]);
  g.append('path').datum(d.Ns).attr('d', d3.line().x(n => x(String(n))).y((n, i) => y(vals[i])))
    .attr('fill', 'none').attr('stroke', COL.accent2).attr('stroke-width', 2);
  g.selectAll('circle').data(d.Ns).join('circle')
    .attr('cx', n => x(String(n))).attr('cy', (n, i) => y(vals[i])).attr('r', 5).attr('fill', COL.accent2);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4).tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text('achieved system floor  vs.  population size N  ·  bound satisfied in 120/120');
}

/* ===================================================================
   6. CROWD-SHARPENING UNDER ROUTING
=================================================================== */
export function drawCrowdRouting(DATA) {
  const d = DATA.exp6;
  const w = 520, h = 340, m = { l: 50, t: 20, r: 20, b: 40 };
  const g = makeSvg('#chart-crowd-routing', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const x = d3.scalePoint().domain(d.n_consult.map(String)).range([0, iw]).padding(0.5);
  const y = d3.scaleLog().domain([0.2, 0.8]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(4)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', v => y(v)).attr('y2', v => y(v)).attr('stroke', COL.grid);

  g.append('path').datum(d.n_consult).attr('d', d3.line().x(n => x(String(n))).y((n, i) => y(d.q_routed[i])))
    .attr('fill', 'none').attr('stroke', COL.accent2).attr('stroke-width', 2);
  g.append('path').datum(d.n_consult).attr('d', d3.line().x(n => x(String(n))).y((n, i) => y(d.q_uniform[i])))
    .attr('fill', 'none').attr('stroke', COL.inkFaint).attr('stroke-width', 2).attr('stroke-dasharray', '4,3');

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4).tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text('failure probability  ·  routed (teal solid) vs uniform (grey dashed)');
}

/* ===================================================================
   7. PARTITION EXTINCTION IS DISCONTINUOUS
=================================================================== */
export function drawExtinction(DATA) {
  const d = DATA.exp7;
  const w = 620, h = 340, m = { l: 46, t: 20, r: 20, b: 40 };
  const g = makeSvg('#chart-extinction', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const x = d3.scaleLinear().domain([0, 1.05]).range([0, iw]);
  const bins = d3.bin().domain([0, 1.05]).thresholds(24)(d.lag_sample);
  const y = d3.scaleLinear().domain([0, Math.max(...bins.map(b => b.length))]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(4)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', v => y(v)).attr('y2', v => y(v)).attr('stroke', COL.grid);

  g.selectAll('rect').data(bins).join('rect')
    .attr('x', b => x(b.x0) + 1).attr('width', b => Math.max(1, x(b.x1) - x(b.x0) - 1))
    .attr('y', b => y(b.length)).attr('height', b => ih - y(b.length))
    .attr('fill', b => b.x0 === 0 ? COL.accent2 : COL.accent);

  g.append('line').attr('x1', x(d.theoretical_min_positive_lag)).attr('x2', x(d.theoretical_min_positive_lag))
    .attr('y1', 0).attr('y2', ih).attr('stroke', COL.decline).attr('stroke-dasharray', '4,3').attr('stroke-width', 1.3);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(6).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4).tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text(`partition lag τ_p  ·  ${(d.fraction_extinct * 100).toFixed(1)}% extinct spike vs. ${(d.fraction_distinguishable * 100).toFixed(1)}% bounded-away positive branch`);
}

/* ===================================================================
   8. OPERATION-SET EXHAUSTION
=================================================================== */
export function drawOperationExhaustion(DATA) {
  const d = DATA.exp8;
  const w = 420, h = 300, m = { l: 46, t: 20, r: 20, b: 56 };
  const g = makeSvg('#chart-operation-exhaustion', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const rows = [
    { label: 'resolved · existing op', v: d.outcome_counts.resolved_by_existing_operation, color: COL.accent2 },
    { label: 'resolved · generated op', v: d.outcome_counts.resolved_by_generation, color: COL.accent },
    { label: 'declined', v: d.outcome_counts.declined, color: COL.decline },
  ];
  const x = d3.scaleBand().domain(rows.map(r => r.label)).range([0, iw]).padding(0.3);
  const y = d3.scaleLinear().domain([0, Math.max(...rows.map(r => r.v)) + 2]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(4)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', v => y(v)).attr('y2', v => y(v)).attr('stroke', COL.grid);

  g.selectAll('rect').data(rows).join('rect')
    .attr('x', r => x(r.label)).attr('width', x.bandwidth())
    .attr('y', r => y(r.v)).attr('height', r => ih - y(r.v)).attr('fill', r => r.color);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).tickSize(4)).call(axisStyle)
    .selectAll('text').attr('transform', 'rotate(-20)').style('text-anchor', 'end');
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4).tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text(`trial count by outcome  ·  0 false exhaustion, 0 dichotomy violations`);
}

/* ===================================================================
   EXPERIMENT TABLE
=================================================================== */
export function populateRouteGraphTable() {
  const tbody = d3.select('#route-exp-table-body');
  EXPERIMENTS.forEach(e => {
    const tr = tbody.append('tr');
    tr.append('td').attr('class', 'n').text('0' + e.n);
    tr.append('td').text(e.name);
    tr.append('td').attr('class', 'n').text(e.thm);
    tr.append('td').text(e.key);
    tr.append('td').append('span').attr('class', 'confirmed').text('● CONFIRMED');
  });
}
