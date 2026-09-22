import * as d3 from 'd3';

export const EXPERIMENTS = [
  { n: 1, name: 'Inclusion holds downward', thm: 'Thm 3.1', key: 'Mean ratio 0.983, CI [0.976, 0.990]' },
  { n: 2, name: 'Non-inclusion upward, measured gap', thm: 'Thm 3.2', key: 'Δ > 0 in 100% of 120 trials' },
  { n: 3, name: 'Sample-complexity separation', thm: 'Thm 3.3', key: 'Ratio = N+1 exactly, err ≤ 1.5e-16' },
  { n: 4, name: 'Federation floor sub-minimum & multiplicative', thm: 'Thm 6.2 / 6.3', key: 'Sub-minimum in 100% of 150 trials' },
  { n: 5, name: 'Cascade allocation vs. knapsack optimum', thm: 'Thm 6.4', key: 'Greedy ≥ 0.658, guarantee 0.632' },
  { n: 6, name: 'Minimum certifying loop is three', thm: 'Thm 7.1 / 7.2', key: '80.1% drop at size-3 exactly' },
  { n: 7, name: 'Quiescence dichotomy', thm: 'Thm 8.2', key: '0 violations across 60 trials' },
  { n: 8, name: 'Route-audit detects false friends', thm: 'Thm 8.3', key: '100% detection vs. 0% central-only' },
  { n: 9, name: 'Verification-floor bound holds', thm: 'Thm 9.2', key: 'Bound satisfied in 100% of 150 trials' },
];

export const COL = {
  ink: '#eef0f3', inkDim: '#98a1b3', inkFaint: '#5c6577',
  accent: '#ff7a45', accent2: '#4fd1c5', decline: '#e0645f', line: '#242a35', grid: '#ffffff10',
};

export function makeSvg(sel, w, h, margin) {
  const el = d3.select(sel);
  el.selectAll('*').remove();
  const svg = el.append('svg')
    .attr('viewBox', `0 0 ${w} ${h}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', 'auto').style('display', 'block');
  return svg.append('g').attr('transform', `translate(${margin.l},${margin.t})`);
}

export function axisStyle(g) {
  g.selectAll('path').attr('stroke', COL.line);
  g.selectAll('line').attr('stroke', COL.line);
  g.selectAll('text').attr('fill', COL.inkDim);
}

/* ===================================================================
   1. THREE-TO-ONE DIAGRAM (problem section)
=================================================================== */
export function drawThreeToOne() {
  const w = 480, h = 380;
  const g = makeSvg('#viz-three-to-one', w, h, { l: 10, t: 10, r: 10, b: 10 });
  const items = [
    { label: 'Acquisition', sub: 'data coverage', y: 40 },
    { label: 'Composition', sub: 'held-out perplexity', y: 150 },
    { label: 'Verification', sub: 'answer plausibility', y: 260 },
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
    .attr('fill', 'none').attr('stroke', COL.accent).attr('stroke-width', 1.6);
  fg.append('text').attr('x', 70).attr('y', 28).attr('text-anchor', 'middle')
    .attr('font-family', "'Fraunces', serif").attr('font-weight', 600).attr('font-size', 15).attr('fill', COL.accent)
    .text('β(R)');
  fg.append('text').attr('x', 70).attr('y', 46).attr('text-anchor', 'middle')
    .attr('font-size', 10).attr('fill', COL.inkFaint).text('the error floor');
}

/* ===================================================================
   2. INCLUSION
=================================================================== */
export function drawInclusion(DATA) {
  const w = 1060, h = 300, m = { l: 46, t: 16, r: 20, b: 36 };
  const g = makeSvg('#chart-inclusion', w, h, m);
  const iw = (w - m.l - m.r - 60) / 2;
  const ih = h - m.t - m.b;

  const left = g.append('g');
  const ratios = DATA.exp1.ratios;
  const x1 = d3.scaleLinear().domain([0, ratios.length - 1]).range([0, iw]);
  const y1 = d3.scaleLinear().domain([0.90, 1.03]).range([ih, 0]);

  left.append('g').attr('class', 'grid').selectAll('line').data(y1.ticks(5)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', d => y1(d)).attr('y2', d => y1(d)).attr('stroke', COL.grid);

  left.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y1(1)).attr('y2', y1(1))
    .attr('stroke', COL.decline).attr('stroke-dasharray', '4,3').attr('stroke-width', 1.3);
  left.append('rect').attr('x', 0).attr('width', iw)
    .attr('y', y1(DATA.exp1.ci_hi)).attr('height', y1(DATA.exp1.ci_lo) - y1(DATA.exp1.ci_hi))
    .attr('fill', COL.accent2).attr('opacity', 0.12);
  left.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y1(DATA.exp1.mean)).attr('y2', y1(DATA.exp1.mean))
    .attr('stroke', COL.accent2).attr('stroke-width', 1.2);

  left.selectAll('circle').data(ratios).join('circle')
    .attr('cx', (d, i) => x1(i)).attr('cy', d => y1(d)).attr('r', 3.4)
    .attr('fill', COL.accent2).attr('opacity', 0.85);

  left.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x1).ticks(6).tickSize(4)).call(axisStyle);
  left.append('g').attr('class', 'axis').call(d3.axisLeft(y1).ticks(5).tickSize(4)).call(axisStyle);
  left.append('text').attr('x', 0).attr('y', -2).attr('font-size', 11).attr('fill', COL.inkDim).text('loss ratio, extremal→restricted');

  const right = g.append('g').attr('transform', `translate(${iw + 60},0)`);
  const radii = ['0.3', '0.5', '0.7', '0.9'];
  const x2 = d3.scalePoint().domain(radii).range([0, iw]).padding(0.5);
  const y2 = d3.scaleLinear().domain([0.20, 0.44]).range([ih, 0]);

  right.append('g').attr('class', 'grid').selectAll('line').data(y2.ticks(5)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', d => y2(d)).attr('y2', d => y2(d)).attr('stroke', COL.grid);

  radii.forEach(r => {
    const vals = DATA.exp2.by_radius[r];
    const cx = x2(r);
    right.selectAll(`.pt-${r.replace('.', '_')}`).data(vals).join('circle')
      .attr('cx', () => cx + (Math.random() - 0.5) * 20)
      .attr('cy', d => y2(d)).attr('r', 2.6)
      .attr('fill', COL.accent).attr('opacity', 0.55);
    const mean = d3.mean(vals);
    right.append('rect').attr('x', cx - 11).attr('y', y2(mean) - 2.5).attr('width', 22).attr('height', 5)
      .attr('fill', COL.ink);
  });
  right.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y2(0)).attr('y2', y2(0));

  right.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x2).tickSize(4)).call(axisStyle);
  right.append('g').attr('class', 'axis').call(d3.axisLeft(y2).ticks(5).tickSize(4)).call(axisStyle);
  right.append('text').attr('x', 0).attr('y', -2).attr('font-size', 11).attr('fill', COL.inkDim).text('gap Δ by restriction box radius');
}

/* ===================================================================
   3. SAMPLE COMPLEXITY
=================================================================== */
export function drawSampleComplexity(DATA) {
  const w = 520, h = 340, m = { l: 44, t: 20, r: 24, b: 40 };
  const g = makeSvg('#chart-samplecomplexity', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const N = DATA.exp3.N;
  const x = d3.scaleLinear().domain([1, 8]).range([0, iw]);
  const y = d3.scaleLinear().domain([0, 10]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(5)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d)).attr('stroke', COL.grid);

  const lineT = d3.line().x(d => x(d)).y(d => y(d + 1));
  g.append('path').datum(d3.range(1, 8.01, 0.1))
    .attr('d', lineT).attr('fill', 'none').attr('stroke', COL.decline)
    .attr('stroke-dasharray', '5,3').attr('stroke-width', 1.4);

  const lineE = d3.line().x((d, i) => x(N[i])).y(d => y(d));
  g.append('path').datum(DATA.exp3.empirical).attr('d', lineE)
    .attr('fill', 'none').attr('stroke', COL.accent2).attr('stroke-width', 2);
  g.selectAll('circle').data(DATA.exp3.empirical).join('circle')
    .attr('cx', (d, i) => x(N[i])).attr('cy', d => y(d)).attr('r', 4)
    .attr('fill', COL.accent2);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(8).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5).tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text('sample-complexity ratio  vs.  N nested restrictions  ·  matches N+1 to 1.5e-16');
}

/* ===================================================================
   4. FEDERATION FLOOR
=================================================================== */
export function drawFederation(DATA) {
  const w = 1060, h = 280, m = { l: 48, t: 18, r: 24, b: 40 };
  const g = makeSvg('#chart-federation', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const sizes = DATA.exp4.sizes;
  const x = d3.scalePoint().domain(sizes).range([0, iw]).padding(0.5);
  const y = d3.scaleLinear().domain([0.84, 0.94]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(5)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d)).attr('stroke', COL.grid);

  g.append('path').datum(sizes).attr('d', d3.area()
    .x(d => x(d)).y0(d => y(DATA.exp4.lo[sizes.indexOf(d)])).y1(d => y(DATA.exp4.hi[sizes.indexOf(d)])))
    .attr('fill', COL.accent2).attr('opacity', 0.15);

  g.append('path').datum(sizes).attr('d', d3.line().x(d => x(d)).y(d => y(DATA.exp4.mean[sizes.indexOf(d)])))
    .attr('fill', 'none').attr('stroke', COL.accent2).attr('stroke-width', 2);
  g.selectAll('circle').data(sizes).join('circle')
    .attr('cx', d => x(d)).attr('cy', d => y(DATA.exp4.mean[sizes.indexOf(d)])).attr('r', 5)
    .attr('fill', COL.accent2);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5).tickSize(4).tickFormat(d3.format('.2f'))).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text('normalised federation floor β(F)/β(R₁)  vs.  federation size n');
}

/* ===================================================================
   5. CASCADE KNAPSACK
=================================================================== */
export function drawCascade(DATA) {
  const w = 520, h = 340, m = { l: 46, t: 20, r: 20, b: 40 };
  const g = makeSvg('#chart-cascade', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const budgets = DATA.exp5.budgets;
  const x = d3.scalePoint().domain(budgets).range([0, iw]).padding(0.6);
  const y = d3.scaleLinear().domain([0.6, 1.02]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(5)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d)).attr('stroke', COL.grid);

  g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y(DATA.exp5.bound)).attr('y2', y(DATA.exp5.bound))
    .attr('stroke', COL.decline).attr('stroke-dasharray', '5,3').attr('stroke-width', 1.4);
  g.append('text').attr('x', iw - 4).attr('y', y(DATA.exp5.bound) - 6).attr('text-anchor', 'end')
    .attr('font-size', 10).attr('fill', COL.decline).text('(1-1/e) = 0.632 guarantee');

  g.append('path').datum(budgets).attr('d', d3.area()
    .x(d => x(d)).y0(d => y(DATA.exp5.lo[budgets.indexOf(d)])).y1(d => y(DATA.exp5.hi[budgets.indexOf(d)])))
    .attr('fill', COL.accent).attr('opacity', 0.15);
  g.append('path').datum(budgets).attr('d', d3.line().x(d => x(d)).y(d => y(DATA.exp5.mean[budgets.indexOf(d)])))
    .attr('fill', 'none').attr('stroke', COL.accent).attr('stroke-width', 2);
  g.selectAll('.dot').data(budgets).join('circle')
    .attr('cx', d => x(d)).attr('cy', d => y(DATA.exp5.mean[budgets.indexOf(d)])).attr('r', 5)
    .attr('fill', COL.accent);
  g.selectAll('.mn').data(budgets).join('circle')
    .attr('cx', d => x(d)).attr('cy', d => y(DATA.exp5.min_observed[budgets.indexOf(d)])).attr('r', 3)
    .attr('fill', 'none').attr('stroke', COL.ink).attr('stroke-width', 1.2);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5).tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text('greedy / optimal ratio  vs.  budget B  ·  ring = worst trial observed');
}

/* ===================================================================
   6. MINIMUM LOOP
=================================================================== */
export function drawMinLoop(DATA) {
  const w = 520, h = 340, m = { l: 46, t: 20, r: 20, b: 40 };
  const g = makeSvg('#chart-minloop', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const sizes = DATA.exp6.sizes;
  const x = d3.scalePoint().domain(sizes).range([0, iw]).padding(0.6);
  const y = d3.scaleLinear().domain([0, 6.6]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(5)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d)).attr('stroke', COL.grid);
  g.append('line').attr('x1', x(3)).attr('x2', x(3)).attr('y1', 0).attr('y2', ih)
    .attr('stroke', COL.inkFaint).attr('stroke-dasharray', '2,3');

  g.append('path').datum(sizes).attr('d', d3.area()
    .x(d => x(d)).y0(d => y(DATA.exp6.certified_lo[sizes.indexOf(d)])).y1(d => y(DATA.exp6.certified_hi[sizes.indexOf(d)])))
    .attr('fill', COL.decline).attr('opacity', 0.13);
  g.append('path').datum(sizes).attr('d', d3.line().x(d => x(d)).y(d => y(DATA.exp6.certified_mean[sizes.indexOf(d)])))
    .attr('fill', 'none').attr('stroke', COL.decline).attr('stroke-width', 2.2);
  g.selectAll('circle').data(sizes).join('circle')
    .attr('cx', d => x(d)).attr('cy', d => y(DATA.exp6.certified_mean[sizes.indexOf(d)])).attr('r', 5)
    .attr('fill', COL.decline);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5).tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text('certified error  vs.  check-graph size  ·  dotted line = size 3');
}

/* ===================================================================
   7. QUIESCENCE DICHOTOMY
=================================================================== */
export function drawQuiescence(DATA) {
  const w = 1060, h = 280, m = { l: 52, t: 18, r: 24, b: 38 };
  const g = makeSvg('#chart-quiescence', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const conv = DATA.exp7.convergent, nonconv = DATA.exp7.nonconvergent;
  const maxR = Math.max(conv.length, nonconv.length) - 1;
  const x = d3.scaleLinear().domain([0, maxR]).range([0, iw]);
  const y = d3.scaleLog().domain([0.01, 1]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(4)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d)).attr('stroke', COL.grid);

  const line = d3.line().x((d, i) => x(i)).y(d => y(d));
  g.append('path').datum(nonconv).attr('d', line).attr('fill', 'none').attr('stroke', COL.decline).attr('stroke-width', 2);
  g.selectAll('.nc').data(nonconv).join('circle').attr('cx', (d, i) => x(i)).attr('cy', d => y(d)).attr('r', 3).attr('fill', COL.decline);

  g.append('path').datum(conv).attr('d', line).attr('fill', 'none').attr('stroke', COL.accent2).attr('stroke-width', 2);
  g.selectAll('.c').data(conv).join('circle').attr('cx', (d, i) => x(i)).attr('cy', d => y(d)).attr('r', 4).attr('fill', COL.accent2);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(8).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4, '~g').tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim).text('joint residual Δ(r), log scale, vs. relaxation round r');

  const lg = g.append('g').attr('transform', `translate(${iw - 190},10)`);
  lg.append('circle').attr('cx', 0).attr('cy', 0).attr('r', 4).attr('fill', COL.accent2);
  lg.append('text').attr('x', 10).attr('y', 4).attr('font-size', 11).attr('fill', COL.inkDim).text('quiescent (converges)');
  lg.append('circle').attr('cx', 0).attr('cy', 18).attr('r', 4).attr('fill', COL.decline);
  lg.append('text').attr('x', 10).attr('y', 22).attr('font-size', 11).attr('fill', COL.inkDim).text('declined (stalls > 0)');
}

/* ===================================================================
   8. ROUTE AUDIT
=================================================================== */
export function drawRouteAudit(DATA) {
  const w = 1060, h = 300, m = { l: 50, t: 18, r: 24, b: 40 };
  const g = makeSvg('#chart-routeaudit', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const x = d3.scaleLinear().domain([0, 0.02]).range([0, iw]);
  const y = d3.scaleLinear().domain([0, 0.72]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(6)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d)).attr('stroke', COL.grid);

  g.selectAll('.ff').data(d3.zip(DATA.exp8.central, DATA.exp8.provoked)).join('circle')
    .attr('cx', d => x(d[0])).attr('cy', d => y(d[1])).attr('r', 4.4)
    .attr('fill', COL.decline).attr('opacity', 0.85);
  g.selectAll('.ctrl').data(d3.zip(DATA.exp8.ctrl_central, DATA.exp8.ctrl_provoked)).join('circle')
    .attr('cx', d => x(d[0])).attr('cy', d => y(d[1])).attr('r', 4.4)
    .attr('fill', COL.accent2).attr('opacity', 0.85);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(5).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(6).tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text('provoked-column gap  vs.  central-column gap');

  const lg = g.append('g').attr('transform', `translate(${iw - 200},10)`);
  lg.append('circle').attr('cx', 0).attr('cy', 0).attr('r', 4).attr('fill', COL.decline);
  lg.append('text').attr('x', 10).attr('y', 4).attr('font-size', 11).attr('fill', COL.inkDim).text('false-friend (constructed)');
  lg.append('circle').attr('cx', 0).attr('cy', 18).attr('r', 4).attr('fill', COL.accent2);
  lg.append('text').attr('x', 10).attr('y', 22).attr('font-size', 11).attr('fill', COL.inkDim).text('genuine agreement (control)');
}

/* ===================================================================
   9. VERIFICATION FLOOR
=================================================================== */
export function drawVerifFloor(DATA) {
  const w = 1060, h = 280, m = { l: 48, t: 18, r: 24, b: 40 };
  const g = makeSvg('#chart-verifloor', w, h, m);
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const etas = DATA.exp9.etas;
  const x = d3.scaleLinear().domain([0, 0.5]).range([0, iw]);
  const y = d3.scaleLinear().domain([0.76, 0.92]).range([ih, 0]);

  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(5)).join('line')
    .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d)).attr('stroke', COL.grid);

  g.append('path').datum(etas).attr('d', d3.area()
    .x(d => x(d)).y0(d => y(DATA.exp9.lo[etas.indexOf(d)])).y1(d => y(DATA.exp9.hi[etas.indexOf(d)])))
    .attr('fill', COL.accent).attr('opacity', 0.15);
  g.append('path').datum(etas).attr('d', d3.line().x(d => x(d)).y(d => y(DATA.exp9.mean[etas.indexOf(d)])))
    .attr('fill', 'none').attr('stroke', COL.accent).attr('stroke-width', 2);
  g.selectAll('circle').data(etas).join('circle')
    .attr('cx', d => x(d)).attr('cy', d => y(DATA.exp9.mean[etas.indexOf(d)])).attr('r', 5)
    .attr('fill', COL.accent);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(6).tickSize(4)).call(axisStyle);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5).tickSize(4)).call(axisStyle);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', COL.inkDim)
    .text('β(R_AB), normalised by τ = β(R_A)+β(R_B)  vs.  disagreement rate η_AB');
}

/* ===================================================================
   EXPERIMENT TABLE
=================================================================== */
export function populateTable() {
  const tbody = d3.select('#exp-table-body');
  EXPERIMENTS.forEach(e => {
    const tr = tbody.append('tr');
    tr.append('td').attr('class', 'n').text('0' + e.n);
    tr.append('td').text(e.name);
    tr.append('td').attr('class', 'n').text(e.thm);
    tr.append('td').text(e.key);
    tr.append('td').append('span').attr('class', 'confirmed').text('● CONFIRMED');
  });
}
