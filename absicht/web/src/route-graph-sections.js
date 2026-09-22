export const ROUTE_GRAPH_SECTIONS_HTML = `
<a class="skiplink" href="#main">Skip to content</a>

<div id="rail"><div id="rail-fill"></div></div>
<nav id="toc" aria-label="Section navigation">
  <a href="#hero" data-label="Start"></a>
  <a href="#problem" data-label="The Problem"></a>
  <a href="#selection" data-label="I · Selection"></a>
  <a href="#allocation" data-label="II · Allocation"></a>
  <a href="#combination" data-label="III · Combination"></a>
  <a href="#elimination" data-label="IV · Elimination"></a>
  <a href="#architecture" data-label="Architecture"></a>
  <a href="#validation" data-label="Validation"></a>
  <a href="#discussion" data-label="Scope"></a>
</nav>

<main id="main">

<!-- ============================= HERO ============================= -->
<section id="hero">
  <div id="hero-bg"></div>
  <div class="wrap content">
    <div class="kicker mono">FEDERATED EXTENSION · MANY OPAQUE RECEIVERS, QUERIED LIVE</div>
    <h1 class="title">Distributed<br><em>Domain Route Graph</em></h1>
    <p class="subtitle">Accountable Compilation composes a <em>given</em> set of receivers.
    This extends it to a population queried live: which receivers to consult, how
    to divide a budget across them, and — the new result — a binary, derived
    stopping criterion for when they agree, replacing a hand-tuned tolerance.</p>

    <div class="hero-meta">
      <div>Presented by <b>Kundai Farai Sachikonye</b></div>
      <div><b>8</b> computational experiments · <b>8</b> confirmed, <b>0</b> failed</div>
      <div><b>3</b> composed source papers · <b>1</b> derived stopping criterion</div>
    </div>

    <div class="stat-row">
      <div class="stat"><div class="num tab" data-count="100">0</div><div class="lbl">% bound satisfaction, Seam Theorem, 120 trials</div></div>
      <div class="stat"><div class="num">10⁻⁸</div><div class="lbl">locked joint failure prob., vs. 0.018 naive</div></div>
      <div class="stat"><div class="num">0</div><div class="lbl">intermediate lag values — binary, not graded</div></div>
      <div class="stat"><div class="num tab" data-count="0">0</div><div class="lbl">dichotomy violations across every experiment</div></div>
    </div>

    <p style="margin-top:28px; font-size:13.5px;"><a href="./index.html">← Accountable Compilation, the base paper</a></p>
  </div>
</section>

<!-- ============================= THE PROBLEM ============================= -->
<section id="problem">
  <div class="wrap">
    <div class="eyebrow">The problem, once N is large and live</div>
    <div class="grid2" style="margin-top:22px;">
      <div class="prose">
        <h2 style="font-size:2.1rem;">Accountable Compilation composes a given set. It never says which set.</h2>
        <p style="color:var(--ink-dim); margin-top:18px;">
          Federation, Cascade, and Verification-Floor all presuppose the
          receivers to compose are already fixed. Three questions arise the
          moment a population is large and queried live rather than
          assembled once, offline: which receivers to consult without ever
          inspecting their content; how to divide a fixed budget across more
          candidates than it can fully afford; and how to combine several
          concurrent responses without silently averaging in an outlier.
        </p>
        <p style="color:var(--ink-dim); margin-top:14px;">
          Two independent manuscripts by the same author — never citing one
          another, never citing the receiver formalism — turn out to answer
          exactly these three questions, and a third supplies something
          neither had: a <b style="color:var(--ink)">derived</b>, not
          hand-tuned, criterion for when two receivers agree.
        </p>
      </div>
      <div id="viz-route-three-to-one"></div>
    </div>
  </div>
</section>

<!-- ============================= PART I: SELECTION ============================= -->
<section id="selection">
  <div class="wrap">
    <div class="eyebrow">Part I · Selection</div>
    <h2 style="font-size:2.3rem; margin-top:14px; max-width:22ch;">The route graph never holds content — and that makes it a receiver too</h2>
    <p class="prose" style="color:var(--ink-dim); margin-top:16px;">
      A <b style="color:var(--ink)">route graph</b> records, for each closed
      question-shape, <i>which</i> catalyst closed it — never the content
      that closed it. Blindness to source and blindness to truth are already
      proved for this object; what's new here is that the route graph
      itself satisfies the receiver definition, so it can be federated,
      cascade-routed, and verified by the unmodified machinery already in
      place.
    </p>

    <div class="theorem">
      <div class="t-label">Theorem — The route graph is a bounded receiver</div>
      <p>Its knowledge framework (observed closure-shapes) stays bounded by
      the number of syntactic types while the number of queries processed
      grows unboundedly — the same counting argument that gives every other
      receiver in this framework its positive floor.</p>
    </div>

    <div class="figure">
      <div id="chart-route-receiver"></div>
      <div class="cap">|Know| (solid) vs |Cands| (dashed) at three population sizes, one representative trial each. |Know| plateaus at the number of syntactic types (<b>2</b>, <b>4</b>, <b>8</b>) within the first few queries while |Cands| keeps growing linearly — boundedness reached in <b>100%</b> of 90 trials.</div>
    </div>
  </div>
</section>

<!-- ============================= PART II: ALLOCATION ============================= -->
<section id="allocation">
  <div class="wrap">
    <div class="eyebrow">Part II · Allocation</div>
    <h2 style="font-size:2.1rem; margin-top:14px;">Water-filling is the Cascade Theorem's continuous limit</h2>
    <div class="grid2" style="margin-top:24px;">
      <div class="prose">
        <p style="color:var(--ink-dim);">
          The Cascade Theorem solves a <i>hard</i> 0/1 admission decision.
          Water-filling divides one shared budget continuously across
          concurrently-consulted receivers. These are the same optimisation
          problem in two regimes, connected by a limiting argument: as the
          number of receivers grows and per-item cost shrinks relative to
          budget, the fractional relaxation's value — which can only match
          or exceed the 0/1 optimum, never fall below it — converges to it.
        </p>
        <div class="theorem">
          <div class="t-label">Theorem — Water-filling solves the linear-relaxation cascade</div>
          <p>The continuous relaxation is solved by a bang-bang water-fill:
          full admission above a shared shadow price, zero below it — the
          classical fractional-knapsack greedy rule, upper-bounding the
          exact 0-1 optimum at every k.</p>
        </div>
      </div>
      <div class="figure" style="margin-top:0;">
        <div id="chart-waterfill"></div>
        <div class="cap">Fractional/exact ratio, 180 trials across k∈{4,...,128}. Falls monotonically from <b>1.311</b> at k=4 to <b>1.0006</b> at k=128 — approaching 1 strictly from above, never below, exactly as the containment argument requires.</div>
      </div>
    </div>
  </div>
</section>

<!-- ============================= PART III: COMBINATION ============================= -->
<section id="combination">
  <div class="wrap">
    <div class="eyebrow">Part III · Combination</div>
    <h2 style="font-size:2.3rem; margin-top:14px; max-width:26ch;">Kuramoto phase-lock gives a floor for live, synchronised federations</h2>
    <p class="prose" style="color:var(--ink-dim); margin-top:16px;">
      A population of concurrently-queried receivers is modelled as a
      society graph, each carrying an answer phase, driven toward Kuramoto
      phase-lock. Federate <i>only</i> the locked subset — never a weighted
      average over everyone queried.
    </p>

    <div class="theorem">
      <div class="t-label">Theorem — Federated phase-lock floor</div>
      <p>A locked sub-federation's floor is bounded by the minimum
      individual locked floor; the joint failure probability over locked
      members — a product, not a mean — decreases geometrically as more
      receivers lock, while unlocked receivers contribute no term to either
      bound at all.</p>
    </div>

    <div class="grid2" style="margin-top:24px;">
      <div class="figure" style="margin-top:0;">
        <div id="chart-phaselock"></div>
        <div class="cap">Joint failure probability of the locked sub-population, log scale, vs. number locked. Falls from <b>3.7×10⁻¹¹</b> at 6 locked to <b>3.9×10⁻¹⁸</b> at 10.</div>
      </div>
      <div class="figure" style="margin-top:0;">
        <div id="chart-phaselock-bars"></div>
        <div class="cap">Locked-only failure probability vs. the naive full-population mean (dragged upward by corrupted, unlocked members present) — a five-order-of-magnitude gap, locked-only lower in <b>100%</b> of trials.</div>
      </div>
    </div>

    <h3 style="font-size:1.6rem; margin-top:56px; max-width:28ch;">But "locked" was a hand-tuned tolerance. Partition extinction derives it instead.</h3>
    <p class="prose" style="color:var(--ink-dim); margin-top:14px;">
      A third, independent manuscript — <i>Coordination Regimes of
      Synchronised Agents</i> — supplies exactly the missing derivation.
      Under a finite-resolution axiom, categorical distinguishability is
      <b style="color:var(--ink)">binary</b>: two things are either
      distinguishable or they are not, with no partial credit. The lag of a
      distinguishing operation — not the raw, genuinely continuous phase
      distance it's computed from — jumps discontinuously to exactly zero
      at lock. There was never an intermediate state for a graded
      combination rule to weight against.
    </p>

    <div class="theorem" style="border-left-color:var(--accent2);">
      <div class="t-label" style="color:var(--accent2);">Theorem — Partition extinction is discontinuous</div>
      <p>As a control parameter drives a pair from distinguishable to
      phase-locked, the partition lag τ<sub>p</sub> is positive while
      distinguishable and exactly 0 once locked, with no intermediate value
      at any point — distinguishability is binary, so there is no state of
      "partial" distinction for τ<sub>p</sub> to graduate through.</p>
    </div>

    <div class="figure">
      <div id="chart-extinction"></div>
      <div class="cap">Sample of 500 partition-lag observations. A spike at exactly <b>0</b> (extinct) and a separate positive population (distinguishable) with none falling below the closed-form minimum (red dashed) — the discontinuity is a property of the derived lag, not of the raw Kuramoto phase distance it's built from.</div>
    </div>

    <div class="theorem">
      <div class="t-label">Theorem — Extinction-locked federation floor</div>
      <p>Exclusion of unlocked receivers is now <i>forced</i>, not merely
      the better design choice: because the lag takes only two values with
      nothing between them, any combination rule that is a non-trivial
      function of it on an interval must be constant on each branch — which
      is exactly the exclude/include rule, not a coincidence.</p>
    </div>

    <div class="card" style="margin-top:32px; max-width:680px;">
      <p style="margin:0; font-size:14.5px; color:var(--ink-dim);">
      What's <b style="color:var(--ink)">not</b> imported: no Cooper-pairing,
      no Bose–Einstein condensation, no claim that the model's operational
      R-bands are phase transitions — the source paper itself retracts that
      framing for its own model. Only the information-theoretic content
      (binary distinguishability, discontinuous lag) crosses over.</p>
    </div>
  </div>
</section>

<!-- ============================= PART IV: ELIMINATION & SEAM ============================= -->
<section id="elimination">
  <div class="wrap">
    <div class="eyebrow">Part IV · Elimination and the seam</div>
    <h2 style="font-size:2.1rem; margin-top:14px; max-width:26ch;">A receiver exposes only what it rules out — and generates a fresh operation, never a guess, when its toolkit is exhausted</h2>

    <div class="opaque-demo">
      <div class="receiver-node"><b>R<sub>i</sub></b><span>local profile receiver</span></div>
      <div class="opaque-gap"><span class="lbl mono">exposes only a column — never Know_i</span></div>
      <div class="receiver-node"><b>coordinator</b><span>main model</span></div>
    </div>

    <div class="theorem" style="margin-top:36px;">
      <div class="t-label">Construction — Generate-and-test on exhausted negation</div>
      <p>Exhaustion means the existing, finite set of partition operations
      has all been tried and none achieves extinction — not that a round
      cap merely expired. Generating a fresh operation is then the only
      available next move, using only the receiver's own machinery.</p>
    </div>

    <div class="grid2" style="margin-top:24px;">
      <div class="figure" style="margin-top:0;">
        <div id="chart-generate-test"></div>
        <div class="cap">Joint residual vs. relaxation round, log scale. Resolved (teal) reaches tolerance after regeneration; declined (red) plateaus, bounded away from zero — <b>0</b> dichotomy violations across 30 trials.</div>
      </div>
      <div class="figure" style="margin-top:0;">
        <div id="chart-operation-exhaustion"></div>
        <div class="cap">Outcome counts under an explicit, finite, enumerable operation-set (not a round-cap proxy): 12 resolved by an existing operation, 10 by a generated one, 8 correctly declined — false-exhaustion rate exactly <b>0</b>, correct-exhaustion rate exactly <b>1</b>.</div>
      </div>
    </div>

    <h3 style="font-size:1.6rem; margin-top:56px;">The Seam Theorem: the whole assembly is again a bounded receiver</h3>
    <div class="theorem">
      <div class="t-label">Theorem — Seam</div>
      <p>Route, water-fill, phase-lock, combine — composed end to end — is
      itself a bounded receiver, with floor bounded by the minimum locked
      individual floor plus a decline measure. There is no distinguished
      top level: the main model is an ordinary receiver, exempt from no
      further instance of federation, cascade routing, or verification.</p>
    </div>

    <div class="figure">
      <div id="chart-seam"></div>
      <div class="cap">End-to-end achieved floor vs. population size N∈{4,8,16,24}, 120 trials. Bound satisfied in <b>100%</b> of trials at every N, with <b>0</b> system-wide declines observed.</div>
    </div>
  </div>
</section>

<!-- ============================= ARCHITECTURE ============================= -->
<section id="architecture">
  <div class="wrap">
    <div class="eyebrow">Assembled architecture</div>
    <h2 style="font-size:2.1rem; margin-top:14px;">How a query actually moves through the system</h2>
    <p class="prose" style="color:var(--ink-dim); margin-top:14px;">
      Route, water-fill, query, extinction-lock, federate, then report or
      decline — with the decline branch a first-class outcome, not an
      afterthought. Every stage below is an ordinary receiver operation
      under the Seam Theorem, so the whole round-trip inherits one
      composable floor rather than three incommensurable ones.
    </p>
    <div class="figure" style="margin-top:24px;">
      <div id="viz-query-flow"></div>
      <div class="cap">The full query round-trip, including the branch to decline when nothing locks — this is a genuine flow, not a static layer stack: which stage runs next depends on what the extinction-lock check finds.</div>
    </div>

    <h3 style="font-size:1.6rem; margin-top:48px;">The federation itself: one main model, many opaque profiles</h3>
    <div class="figure" style="margin-top:14px;">
      <div id="viz-federation-topology"></div>
      <div class="cap">Every edge between the main model and a profile carries a routing decision — never content. Locked profiles (teal, solid) are federated into the answer; excluded profiles (grey, dashed) contributed nothing this round, and the main model never inspected why.</div>
    </div>

    <h3 style="font-size:1.6rem; margin-top:48px;">Four layers, one composable floor</h3>
    <div class="layer-diagram">
      <div class="layer">
        <div><div class="name">Layer 1 — Selection</div><div style="font-size:13px; color:var(--ink-dim);">Route graph, content-blind by construction, itself a bounded receiver</div></div>
        <div class="gov">Route-Graph Receiver Theorem</div>
      </div>
      <div class="layer-arrow">↓</div>
      <div class="layer">
        <div><div class="name">Layer 2 — Allocation</div><div style="font-size:13px; color:var(--ink-dim);">Fixed budget water-filled across selected profiles by marginal informativeness</div></div>
        <div class="gov">Water-Filling / Cascade</div>
      </div>
      <div class="layer-arrow">↓</div>
      <div class="layer">
        <div><div class="name">Layer 3 — Combination</div><div style="font-size:13px; color:var(--ink-dim);">Extinction-locked receivers federated; unlocked ones excluded, forced by a binary discontinuity</div></div>
        <div class="gov">Federated Phase-Lock · Partition Extinction</div>
      </div>
      <div class="layer-arrow">↓</div>
      <div class="layer" style="border-color:var(--accent);">
        <div><div class="name" style="color:var(--accent);">Layer 4 — Elimination and report</div><div style="font-size:13px; color:var(--ink-dim);">Negation-only exposure; generate-and-test on exhausted operation-sets; answer or decline</div></div>
        <div class="gov">Seam Theorem</div>
      </div>
    </div>
  </div>
</section>

<!-- ============================= VALIDATION ============================= -->
<section id="validation">
  <div class="wrap">
    <div class="eyebrow">Validation suite</div>
    <h2 style="font-size:2.1rem; margin-top:14px;">Eight seeded computational experiments. Eight confirmed. Zero failed.</h2>
    <p class="prose" style="color:var(--ink-dim); margin-top:14px;">
      The first version of the binary-lock experiment was a tautology — a
      clamp function that structurally could never emit an intermediate
      value, regardless of the underlying dynamics. Fixed to test the
      theorem's actual claim (the derived lag, not raw phase distance) once
      the tautology was caught during validation, the same discipline this
      framework's earlier suite already established.
    </p>
    <table class="exp-table">
      <thead><tr><th>#</th><th>Experiment</th><th>Theorem</th><th>Key result</th><th>Status</th></tr></thead>
      <tbody id="route-exp-table-body"></tbody>
    </table>
  </div>
</section>

<!-- ============================= SCOPE ============================= -->
<section id="discussion">
  <div class="wrap grid2">
    <div class="prose">
      <div class="eyebrow">What this does not import</div>
      <h2 style="font-size:1.9rem; margin-top:14px;">Structural correspondence, not physical identity</h2>
      <p style="color:var(--ink-dim); margin-top:16px; font-size:14.5px;">
        Partition extinction's binary distinguishability and discontinuous
        lag are imported. No claim here depends on exchange statistics,
        Cooper pairing, or Bose–Einstein condensation — two extinction-locked
        receivers are functionally equivalent on the categorical observables
        the coordination dynamics act on, nothing about the ontological
        status of a physical particle.
      </p>
      <p style="color:var(--ink-dim); margin-top:12px; font-size:14.5px;">
        Nor are the source paper's five operational R-bands imported as
        phase transitions — its own Proposition 1 retracts exactly that
        claim for its own model, and this framework does not reintroduce it.
      </p>
    </div>
    <div class="prose">
      <div class="eyebrow">Open problems</div>
      <div class="two-col-list" style="grid-template-columns:1fr;">
        <div><b>1</b> A population-level route-audit — locking provoked, not merely central, phases — is stated but not yet developed into its own floor bound.</div>
        <div><b>2</b> Whether the Seam Theorem's bound is affected when the frequency-law density g is estimated from a finite receiver sample rather than known exactly.</div>
        <div><b>3</b> Whether observable-commutation extends from a pairwise opaque pair to a full locked population simultaneously.</div>
      </div>
    </div>
  </div>
</section>

<footer>
  <div class="wrap fline">
    <div>Distributed Domain Route Graph — Federated Extension of Accountable Compilation</div>
    <div class="mono">validation/results/summary.json · 8/8 confirmed · <a href="./index.html">base paper →</a></div>
  </div>
</footer>

</main>
`;
