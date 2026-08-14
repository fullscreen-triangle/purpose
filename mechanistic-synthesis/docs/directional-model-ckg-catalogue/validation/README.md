# Validation suite — *The Directional Pair*

Executable checks for every load-bearing claim in
`directional-model-ckg-catalogue.tex`.

```bash
python run_all.py           # 32 experiments, ~12 s, writes results/*.json
```

Exit code is `0` iff every experiment passes.

## Design

Every object in the paper is a finite weighted graph and every quantity is a
minimum cut, a reachability, a domination, or a convex optimum. The suite
therefore needs no external numerical dependency: `core.py` supplies an exact
Edmonds–Karp max-flow / min-cut backend over the standard library alone.

Each experiment seeds its own RNG from base seed 42. Verified: two consecutive
runs produce **byte-identical** JSON across all 38 result files.

## Files

| file | contents |
|---|---|
| `core.py` | contact graphs, exact min-cut, record, term maps, reachability, domination |
| `exp_foundations.py` | E01–E06 — Part I |
| `exp_identity.py` | E07–E11 — Part II |
| `exp_directional.py` | E12–E17 — Part III |
| `exp_probing.py` | E18–E26 — Part IV |
| `exp_separation.py` | E27–E32 — Part V |
| `run_all.py` | master runner, writes `results/master_results.json` |

## Experiments

### Part I — Foundations

| id | claim | shape |
|---|---|---|
| E01 | positive floor: σ(v) ≥ β > 0 | bound, 60 graphs / 484 separations |
| E02 | σ is a relabelling invariant; the label is not | identity, max discrepancy 0 |
| E03 | identity is a region — minimiser side is not a singleton | structural, 7 sizes |
| E04 | record is monotone; un-committing advances it | bound, 40 walks × 30 steps |
| E05 | self-blunting: capacity falls, record rises | identity, 30 instruments |
| E06 | no traceless, no unchanged instrument | search, 200 trials, 0 found |

### Part II — Identity

| id | claim | shape |
|---|---|---|
| E07 | Theseus: χ preserved in all three cases, record + termination separate them | structural |
| E08 | χ alone is not an identity criterion | 40 trials, 100% collision |
| E09 | snapshot/restore yields a distinct individual | 30 trials |
| E10 | no consistent self-verification | **exhaustive over all 65 536 total verifiers** |
| E11 | distinct consumers register distinct cells, each correct | 40 trials |

### Part III — The Directional Pair

| id | claim | shape |
|---|---|---|
| E12 | cell(v) = T(v, v, Π_rest) — the catalogue is the table at rest | identity, 50 graphs |
| E13 | process-side outputs are cuts of the same type | identity, 200 outputs |
| E14 | path opacity — endpoint invariants do not distinguish interiors | identity |
| E15 | representation mobility — mean exact, components may be inadmissible | 200 trials |
| E16 | the pair blunts where neither half alone does | 40 trials, 3 regimes |
| E17 | no staleness, no re-presentation | 40 trials × 5 queries |

### Part IV — Construction and Probing

| id | claim | shape |
|---|---|---|
| E18 | correct for **any** term map (5 adversarial kinds) | 60 maps |
| E19 | floor monotone under refinement — ground-truth-free quality readout | 40 chains |
| E20 | κ(γ₁⋄…⋄γₙ) = 1 − ∏(1−κᵢ) | identity, 400 chains, err < 1e−12 |
| E21 | saturation iff Σκᵢ diverges | 4 canonical sequences |
| E22 | diversify beats repeating the weakest | 300 trials |
| E23 | coherence needs a ≥3-cycle | **exhaustive on 2- and 3-member digraphs** + 800 random |
| E24 | selection is a 0/1 knapsack; greedy within 1 − c_max/B | 150 instances vs exact DP |
| E25 | water-filling: KKT, sharp dropout, price monotonicities | 120 convex instances |
| E26 | seek ≠ nec; nec ⊆ reach; necessity = domination | diamond ×10, chain ×7, 60 random |

### Part V — Closure and the Separation

| id | claim | shape |
|---|---|---|
| E27 | closure strictly stronger than a confidence threshold | 5 two-cluster graphs |
| E28 | convergent or contested closure, exhaustive and exclusive | 300 runs |
| E29 | **retrieval cannot express admissibility** — the witness pair | 7 sizes |
| E30 | attributes do not close the gap | 200 bounded patterns |
| E31 | **corpus-determined maps cannot express admissibility** | 500 arbitrary Λ |
| E32 | the pair *does* express it, and separates the witness | 200 trials |

## The separation witness

E29/E31 construct the paper's central witness and reproduce it exactly:

```
n_y = 4:  assertions = 7          contact relations IDENTICAL
          σ(v₀,x*)   = 2.0 = 2.0  pair alignment EQUAL
          system floor  2.0 vs 1.0
          accountable   True vs False        ← verdicts DIFFER
```

Two graphs indistinguishable to any retrieval query, and to all 500 tested
corpus-determined maps, differ in the admissibility verdict — because the
threshold is a minimum over *every* item, including the yᵢ that lie on no path
between v₀ and x*.

## Notes on two boundary cases

Both are recorded in the JSON rather than passed over:

- **E26** — the domination criterion characterises necessity for *non-seed*
  items. A seed is load-bearing by definition; where it reaches only itself it
  dominates nothing while remaining necessary. Seeds are counted separately
  (`seed_boundary_cases_excluded`), not silently included.
- **E19** — floor monotonicity is a *monotonicity* result, not a calibration. A
  rising floor indicates refinement; a high floor is not by itself evidence of a
  good term map. Optimising it directly would be an error.

## Result format

One JSON per experiment (`results/e01.json` … `results/e32.json`), one per part,
plus `results/master_results.json`. Each carries the claim it tests, the grid
it ran on, the measured quantities, a `verdict`, and a truncated sample of rows.
