export const SECTIONS_HTML = `
<a class="skiplink" href="#main">Skip to content</a>

<div id="rail"><div id="rail-fill"></div></div>
<nav id="toc" aria-label="Section navigation">
  <a href="#hero" data-label="Start"></a>
  <a href="#problem" data-label="The Problem"></a>
  <a href="#acquisition" data-label="I · Acquisition"></a>
  <a href="#composition" data-label="II · Composition"></a>
  <a href="#verification" data-label="III · Verification"></a>
  <a href="#new-result" data-label="IV · New Result"></a>
  <a href="#architecture" data-label="Architecture"></a>
  <a href="#validation" data-label="Validation"></a>
  <a href="#discussion" data-label="Scope"></a>
</nav>

<main id="main">

<!-- ============================= HERO ============================= -->
<section id="hero">
  <div id="hero-bg"></div>
  <div class="wrap content">
    <div class="kicker mono">RESEARCH NOTE · DOMAIN-SPECIFIC MODEL SYNTHESIS</div>
    <h1 class="title">Accountable<br><em>Compilation</em></h1>
    <p class="subtitle">A unified framework for producing, composing, and cross-verifying
    domain-specific language models — governed, end to end, by a single
    provably well-behaved quantity: a bounded receiver's irreducible error floor.</p>

    <div class="hero-meta">
      <div>Presented by <b>Kundai Farai Sachikonye</b></div>
      <div><b>9</b> computational experiments · <b>9</b> confirmed, <b>0</b> failed</div>
      <div><b>6</b> theorems · <b>1</b> impossibility result</div>
    </div>

    <div class="stat-row">
      <div class="stat"><div class="num tab" data-count="150">0</div><div class="lbl">trials in the federation-floor suite</div></div>
      <div class="stat"><div class="num tab" data-count="100">0</div><div class="lbl">% false-friend detection, route-audit</div></div>
      <div class="stat"><div class="num">0.632</div><div class="lbl">worst-case knapsack guarantee, never breached</div></div>
      <div class="stat"><div class="num">3</div><div class="lbl">minimum size of a certifying check-loop</div></div>
    </div>
  </div>
</section>

<!-- ============================= THE PROBLEM ============================= -->
<section id="problem">
  <div class="wrap">
    <div class="eyebrow">The problem, posed wrongly</div>
    <div class="grid2" style="margin-top:22px;">
      <div class="prose">
        <h2 style="font-size:2.1rem;">Three teams, three toolchains, three objectives — stitched together after the fact.</h2>
        <p style="color:var(--ink-dim); margin-top:18px;">
          Building a domain-specific model is usually split into three
          independent engineering decisions: gather and clean data, fine-tune
          a base model, check the outputs at inference time. Each is owned by
          a different team, optimising a different proxy — data coverage,
          held-out perplexity, answer plausibility.
        </p>
        <p style="color:var(--ink-dim); margin-top:14px;">
          This paper argues the three are not independent. They are three
          views of one underlying quantity: the <b style="color:var(--ink)">irreducible error floor</b>
          of a bounded system resolving a query against an incompletely known
          domain. Training-data selection is floor minimisation by choice of
          task distribution. Composition is floor minimisation by choice of
          receiver set. Verification is floor minimisation by choice of
          check structure.
        </p>
      </div>
      <div id="viz-three-to-one"></div>
    </div>
  </div>
</section>

<!-- ============================= PART I: ACQUISITION ============================= -->
<section id="acquisition">
  <div class="wrap">
    <div class="eyebrow">Part I · Acquisition</div>
    <h2 style="font-size:2.3rem; margin-top:14px; max-width:20ch;">Where competence should be trained</h2>
    <p class="prose" style="color:var(--ink-dim); margin-top:16px;">
      Every task domain has an <b style="color:var(--ink)">extremal regime</b>: the
      sub-domain that exercises every degree of freedom of the domain's action
      space, under a single scalar objective — simultaneously the hardest
      instance along every axis and the cleanest in its feedback.
    </p>

    <div class="theorem">
      <div class="t-label">Theorem 3.1 — Inclusion</div>
      <p>A model &epsilon;-competent on the extremal regime is automatically
      &epsilon;-competent on every restriction — without further training —
      by a non-expansive projection argument.</p>
      <div class="ref">Proved via metric-projection non-expansiveness onto nested admissible sets.</div>
    </div>
    <div class="theorem" style="border-left-color:var(--decline);">
      <div class="t-label" style="color:var(--decline);">Theorem 3.2 — Non-inclusion upward</div>
      <p>The converse fails: a model competent on a restriction is <em>not</em>
      competent on the extremal regime, and the gap &Delta; is bounded away
      from zero whenever the uncovered region has positive measure.</p>
    </div>

    <div class="figure">
      <div id="chart-inclusion"></div>
      <div class="cap"><b>Left</b> — loss ratio J(restricted)&nbsp;/&nbsp;J(extremal) across 30 trials transferring downward, no retraining: mean <b>0.983</b>, 95% CI [0.976, 0.990], entirely at or below the no-cost reference. <b>Right</b> — the out-of-support gap &Delta; measured when training goes the other way, at four restriction widths: strictly positive in <b>100%</b> of 120 trials, shrinking as the restriction widens toward the full extremal regime.</div>
    </div>
  </div>
</section>

<section style="padding-block:88px; border-bottom:1px solid var(--line-soft);">
  <div class="wrap grid2">
    <div class="prose">
      <div class="eyebrow">Sample complexity</div>
      <h2 style="font-size:1.9rem; margin-top:14px;">One training run pays once. Per-restriction training pays N+1 times.</h2>
      <p style="color:var(--ink-dim); margin-top:16px;">
        Because the Inclusion Theorem transfers competence downward for free,
        a domain need only train once, at its extremal regime, to cover every
        nested restriction a deployment will ever see. Training each
        restriction separately pays the full sample-complexity cost N+1
        times over — a lower bound realised exactly, not asymptotically, in
        the validation suite.
      </p>
      <div class="theorem">
        <div class="t-label">Domain Contract</div>
        <p style="font-size:14.5px;">A domain need only declare a typed
        operation vocabulary <i>O</i>, a query embedding &psi;, an
        extremal-regime sampler &sigma;* , and a verifier <i>Verd</i> — no
        labelled corpus required.</p>
      </div>
    </div>
    <div id="chart-samplecomplexity"></div>
  </div>
</section>

<!-- ============================= PART II: COMPOSITION ============================= -->
<section id="composition">
  <div class="wrap">
    <div class="eyebrow">Part II · Composition</div>
    <h2 style="font-size:2.3rem; margin-top:14px; max-width:22ch;">How several models should be combined</h2>
    <p class="prose" style="color:var(--ink-dim); margin-top:16px;">
      A <b style="color:var(--ink)">bounded receiver</b> is a map from queries
      to candidate answers whose internal state, being finite, forces a
      strictly positive worst-case residual — the <b style="color:var(--ink)">floor</b>.
      No receiver escapes this by getting bigger; it only pushes the floor
      lower. Combining receivers pushes it lower still.
    </p>

    <div class="theorem">
      <div class="t-label">Theorem 6.2 — Federation</div>
      <p>The floor of a federation (union of candidate sets) is at most the
      minimum floor of its constituents — strictly less whenever the
      federation is non-redundant.</p>
    </div>

    <div class="figure">
      <div id="chart-federation"></div>
      <div class="cap">Normalised federation floor across sizes n&nbsp;=&nbsp;1&ndash;5, 150 trials. Falls monotonically from <b>0.916</b> at n=1 to <b>0.863</b> at n=5; the sub-minimum bound &beta;(F) &le; min<sub>i</sub> &beta;(R<sub>i</sub>) held in <b>100%</b> of trials.</div>
    </div>
  </div>
</section>

<section style="padding-block:88px; border-bottom:1px solid var(--line-soft);">
  <div class="wrap grid2 rev">
    <div id="chart-cascade"></div>
    <div class="prose">
      <div class="eyebrow">Cascade routing</div>
      <h2 style="font-size:1.9rem; margin-top:14px;">Allocating a query budget across receivers is an exact knapsack problem.</h2>
      <div class="theorem">
        <div class="t-label">Theorem 6.4 — Cascade</div>
        <p>Minimising the federation's floor under a budget constraint is a
        0–1 knapsack in v<sub>i</sub> = log( &Omega; / (&Omega; &minus; &beta;(R<sub>i</sub>)) ) — exact via DP
        in O(kB), and within (1&minus;1/e) &approx; 0.632 of optimal by the
        greedy value-density rule.</p>
      </div>
      <p style="color:var(--ink-dim); margin-top:14px; font-size:14.5px;">
        Across five budgets and 150 trials, greedy routing averaged
        <b style="color:var(--ink)">95.9–98.5%</b> of the exact DP optimum.
        The single worst trial observed anywhere in the suite still cleared
        the guarantee, <b style="color:var(--ink)">0.658</b> against
        <b style="color:var(--ink)">0.632</b>.
      </p>
    </div>
  </div>
</section>

<section style="padding-block:88px; border-bottom:1px solid var(--line-soft);">
  <div class="wrap">
    <div class="eyebrow">Certification topology</div>
    <h2 style="font-size:1.9rem; margin-top:14px; max-width:30ch;">No chain of pairwise checks can certify below its weakest unverified link. A closed loop of three can.</h2>
    <div class="grid2" style="margin-top:24px;">
      <div class="prose">
        <div class="theorem" style="border-left-color:var(--decline);">
          <div class="t-label" style="color:var(--decline);">Theorem 7.1 — No acyclic certification</div>
          <p>Any DAG of pairwise checks has a terminal receiver with no
          outgoing check — its floor bounds the whole certification from
          below, however long the chain.</p>
        </div>
        <div class="theorem">
          <div class="t-label">Theorem 7.2 — A three-cycle suffices</div>
          <p>A directed 3-cycle of non-expansive mutual checks is a
          contraction with a unique fixed point — a jointly agreed output no
          single receiver could unilaterally alter.</p>
        </div>
      </div>
      <div class="figure" style="margin-top:0;">
        <div id="chart-minloop"></div>
        <div class="cap">Certified error vs. check-graph size, one corrupted member (p=0.4), 30 trials/size. An <b>80.1%</b> drop lands exactly at the size-2 → size-3 transition, and nowhere else.</div>
      </div>
    </div>
  </div>
</section>

<!-- ============================= PART III: VERIFICATION ============================= -->
<section id="verification">
  <div class="wrap">
    <div class="eyebrow">Part III · Verification without disclosure</div>
    <h2 style="font-size:2.3rem; margin-top:14px; max-width:26ch;">Two systems that share no decoding map — how do they certify agreement?</h2>
    <p class="prose" style="color:var(--ink-dim); margin-top:16px;">
      An <b style="color:var(--ink)">opaque pair</b> is two receivers whose
      internal knowledge states admit no shared decoding: neither can be
      asked about the other's representation — only about its
      <i>output</i>, and what that output itself provokes.
    </p>

    <div class="opaque-demo">
      <div class="receiver-node"><b>R<sub>A</sub></b><span>trained adapter</span></div>
      <div class="opaque-gap"><span class="lbl mono">no shared decoding map h</span></div>
      <div class="receiver-node"><b>R<sub>B</sub></b><span>knowledge source</span></div>
    </div>

    <div class="theorem" style="margin-top:36px;">
      <div class="t-label">Construction — Four-column comparison</div>
      <p>Compare not just each receiver's <b>central</b> answer, but each
      receiver's own <b>provoked</b> follow-up on its own answer — computed
      using only that receiver's own machinery. Iterate to a fixed point.</p>
    </div>

    <div class="figure">
      <div id="chart-quiescence"></div>
      <div class="cap">Joint residual &Delta;<sup>(r)</sup> over relaxation rounds, log scale. One representative quiescent trace (teal) reaches tolerance; one declined trace (red) plateaus at <b>0.576</b>, bounded away from zero. Across 60 trials: <b>0</b> dichotomy violations — every instance either converges or provably does not, no third outcome.</div>
    </div>

    <div class="theorem" style="border-left-color:var(--decline); margin-top:40px;">
      <div class="t-label" style="color:var(--decline);">Theorem 8.3 — Route-Audit</div>
      <p>Two receivers can agree centrally while their provoked follow-ups
      concern entirely different underlying conditions — a "false friend"
      no central-only comparison can see.</p>
    </div>

    <div class="figure">
      <div id="chart-routeaudit"></div>
      <div class="cap">Provoked-column gap vs. central-column gap, 30 constructed false-friends (red) against 30 genuine-agreement controls (teal). Both populations sit near zero centrally; only the false friends separate on the provoked axis. Route-audit detection: <b>100%</b>. Central-only detection: <b>0%</b>.</div>
    </div>
  </div>
</section>

<!-- ============================= PART IV: NEW RESULT ============================= -->
<section id="new-result">
  <div class="wrap">
    <div class="eyebrow">Part IV · The paper's new result</div>
    <h2 style="font-size:2.3rem; margin-top:14px; max-width:24ch;">Verification is not an oracle standing outside the system. It is a receiver.</h2>
    <p class="prose" style="color:var(--ink-dim); margin-top:16px;">
      Without this result, an architecture combining acquisition, federation,
      and cross-checking has three incommensurable design surfaces: a
      training budget, a routing budget, and an unaccounted verification step
      assumed free and infallible.
    </p>

    <div class="theorem">
      <div class="t-label">Theorem 9.2 — Verification-Floor</div>
      <p>The four-column relaxation, run to quiescence or declared
      non-convergence, is itself a bounded receiver, with floor
      &beta;(R<sub>AB</sub>) &le; &beta;(R<sub>A</sub>) + &beta;(R<sub>B</sub>) + &eta;<sub>AB</sub>, where
      &eta;<sub>AB</sub> is the measured disagreement rate.</p>
    </div>

    <div class="figure">
      <div id="chart-verifloor"></div>
      <div class="cap">Verification-receiver floor, normalised by the bound &tau; = &beta;(R<sub>A</sub>)+&beta;(R<sub>B</sub>), across five disagreement rates &eta;<sub>AB</sub>, 150 trials. The bound held in <b>100%</b> of trials at every rate; at &eta;<sub>AB</sub>=0 the achieved floor sat <b>85.6%</b> below the bound — validated, but not observed to be tight.</div>
    </div>

    <div class="card" style="margin-top:32px; max-width:640px;">
      <p style="margin:0; font-size:14.5px; color:var(--ink-dim);">
      <b style="color:var(--ink)">Corollary 9.4</b> — a tree of opaque pairs,
      leaves acquired as in Part I, each internal node a verification
      receiver, has a well-defined root floor computable bottom-up — and that
      root may itself be federated, cascade-routed, and further verified by
      the unmodified machinery of Part II.
      </p>
    </div>
  </div>
</section>

<!-- ============================= ARCHITECTURE ============================= -->
<section id="architecture">
  <div class="wrap">
    <div class="eyebrow">Assembled architecture</div>
    <h2 style="font-size:2.1rem; margin-top:14px;">Accountable Compilation — three layers, one governing quantity</h2>
    <div class="layer-diagram">
      <div class="layer">
        <div><div class="name">Layer 1 — Acquisition</div><div style="font-size:13px; color:var(--ink-dim);">One low-rank adapter per sub-domain, trained once at its extremal regime</div></div>
        <div class="gov">Inclusion Theorem</div>
      </div>
      <div class="layer-arrow">↓</div>
      <div class="layer">
        <div><div class="name">Layer 2 — Composition</div><div style="font-size:13px; color:var(--ink-dim);">Federate, knapsack-route under budget, certify via a ≥3-cycle check graph</div></div>
        <div class="gov">Federation · Cascade · Min-Loop</div>
      </div>
      <div class="layer-arrow">↓</div>
      <div class="layer">
        <div><div class="name">Layer 3 — Cross-domain verification</div><div style="font-size:13px; color:var(--ink-dim);">Opaque pairs certified by four-column relaxation, not surface comparison</div></div>
        <div class="gov">Quiescence · Route-Audit</div>
      </div>
      <div class="layer-arrow">↓</div>
      <div class="layer" style="border-color:var(--accent);">
        <div><div class="name" style="color:var(--accent);">Single reported reliability figure</div><div style="font-size:13px; color:var(--ink-dim);">Root floor, computed bottom-up across the whole deployment</div></div>
        <div class="gov">Verification-Floor Theorem</div>
      </div>
    </div>
  </div>
</section>

<!-- ============================= VALIDATION ============================= -->
<section id="validation">
  <div class="wrap">
    <div class="eyebrow">Validation suite</div>
    <h2 style="font-size:2.1rem; margin-top:14px;">Nine seeded computational experiments. Nine confirmed. Zero failed.</h2>
    <p class="prose" style="color:var(--ink-dim); margin-top:14px;">
      Every experiment is an explicit, seeded simulation over finite
      structures — no external trained language model — so every reported
      number is an exact or statistically characterised property of the
      mathematical objects the theorems are proved about.
    </p>
    <table class="exp-table">
      <thead><tr><th>#</th><th>Experiment</th><th>Theorem</th><th>Key result</th><th>Status</th></tr></thead>
      <tbody id="exp-table-body"></tbody>
    </table>
  </div>
</section>

<!-- ============================= SCOPE ============================= -->
<section id="discussion">
  <div class="wrap grid2">
    <div class="prose">
      <div class="eyebrow">Scope conditions</div>
      <h2 style="font-size:1.9rem; margin-top:14px;">What this framework does not claim</h2>
      <p style="color:var(--ink-dim); margin-top:16px; font-size:14.5px;">
        The framework applies to task domains admitting an extremal regime
        and restrictions satisfying the non-expansive projection axiom.
        Domains where a "harder" instance qualitatively changes the problem,
        rather than restricting it, fall outside the Inclusion Theorem's
        hypotheses.
      </p>
      <p style="color:var(--ink-dim); margin-top:12px; font-size:14.5px;">
        The Federation and Cascade Theorems assume independence of resolution
        failures; correlated receivers turn the multiplicative law into an
        upper bound, not an equality — consistent with the <b style="color:var(--ink)">0.118</b>
        mean absolute error observed in Experiment 4.
      </p>
      <p style="color:var(--ink-dim); margin-top:12px; font-size:14.5px;">
        The Verification-Floor bound is validated but not shown tight:
        Experiment 9 found the achieved floor comfortably inside the bound,
        not approaching it. Closing that gap is left open.
      </p>
    </div>
    <div class="prose">
      <div class="eyebrow">Open problems</div>
      <div class="two-col-list" style="grid-template-columns:1fr;">
        <div><b>1</b> Characterise exactly which domains satisfy the non-expansive projection axiom, beyond the convex and locally-Lipschitz sufficient cases.</div>
        <div><b>2</b> Which follow-up policies make the route-audit maximally discriminating, not merely sufficient to exhibit some detectable instance?</div>
        <div><b>3</b> The optimal <i>shape</i> of a verification-receiver tree given a fixed leaf set and a fixed total verification budget — a further knapsack-style optimisation.</div>
      </div>
    </div>
  </div>
</section>

<footer>
  <div class="wrap fline">
    <div>Accountable Compilation — Research Note, Domain-Specific Model Synthesis</div>
    <div class="mono">validation/results/summary.json · 9/9 confirmed</div>
  </div>
</footer>

</main>
`;
