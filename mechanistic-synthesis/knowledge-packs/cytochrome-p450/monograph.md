# Categorical Mechanics of Cytochrome P450

**A Complete Description of Substrate Oxidation in Bounded Phase Space**

Author: Kundai Farai Sachikonye
Affiliation: AIMe Registry for Artificial Intelligence / Technical University of Munich
Status: In progress (6 thesis papers + 1 methods paper complete; ~14 total planned)

---

## Central Thesis

Cytochrome P450 — its construction, its seven-state catalytic cycle, its
complete reaction repertoire, its 57-member human isoform family, its
drug-metabolism pharmacology, and its full spectroscopic atlas — can be
expressed as a **single recursively-decomposable S-expression** $\xi_{\mathrm{P450}}$,
evaluated under appropriate receivers using the framework's instrument suite.

Every published P450 result (PDB structures, BRENDA kinetics, ChEMBL activity,
pharmacogenomic outcomes) is recovered as a special-case
$\mathrm{eval}_{\mathcal{R}}(\text{sub-expression of } \xi_{\mathrm{P450}})$.
New predictions follow from sub-expressions whose receivers have not yet been
instantiated experimentally.

**Scope:** ~14 papers, ~7 parts, comprehensive cross-validation against ~16
academic databases, all framework instruments exercised at least once, all 57
human CYP isoforms covered, ~30 published reactions characterized, full
catalytic cycle predicted at fs/pm resolution.

---

## Status Summary

| Paper | Title | Status |
|---|---|---|
| 1 | The S-Expression Algebra for Biomolecules | **DONE** ✓ |
| 2 | The P450 Address Manifold + CYP3A4 Fold | **DONE** ✓ |
| 2.5 | GLB-Based Structural Input to the Receiver (methods) | **DONE** ✓ |
| 3 | The Resting and Substrate-Bound States of CYP3A4 | **DONE** ✓ |
| 4 | **Observing Electron Transfer through CPR (HEADLINE)** | **DONE** ✓ |
| 5 | Compound I Formation via O-O Heterolysis (downstream chemistry) | **DONE** ✓ |
| 6 | C-H Activation (Hydroxylation, Epoxidation, Rebound) | **DONE** ✓ |
| 7 | Heteroatom Oxidation and Dealkylation | planned |
| 8 | Atypical Reactions Atlas | planned |
| 9 | The 57 Human Isoforms as Variants | planned |
| 10 | Polymorphisms, DDIs, and Inhibitor Design | planned |
| 11 | Membrane Anchoring and Partner Coupling | planned |
| 12 | The Seven States as a Closed Orbit (Synthesis) | planned |
| 13 | Complete Spectroscopic Atlas | planned |
| 14 | Database-Wide Validation (Recovery Benchmark) | planned |

Each completed paper has: full LaTeX manuscript, references.bib, validation
suite (8 Python scripts), validation results (JSON), figure panels (8 PNGs
with 3D charts), and LaTeX captions file.

---

## Monograph Outline

### Part I — Foundations (1 paper)

#### **Paper 1: The S-Expression Algebra for Biomolecules** ✓ DONE

**Path:** `publications/foundations/expression-algebra-for-biomolecules/`

**Thesis:** The unconstrained-substructures calculus instantiates concretely on
biomolecules. Defines $\xi_{\mathrm{protein}}$, the receiver $\mathcal{R}_{\mathrm{bio}}$, the
conversion functors $\Phi^{\mathfrak{O} \leftrightarrow \mathfrak{C} \leftrightarrow \mathfrak{P}}$
for biomolecular quantities, the floor $\mathfrak{S}_{\mathrm{floor}}(\mathcal{R}_{\mathrm{bio}})$,
and the leaf-to-instrument assignment protocol.

**Resolves:** the τ-assignment ambiguity and the Δs=0 / spin-crossover question
(both fold into the receiver specification via the two-tier chirality coordinate).

**Validation:** 8/8 PASS (Floor Theorem 3.7×10⁻⁴, capacity formula C(n)=2n²,
cycle closure under S₃, amino-acid coordinates, τ-assignment rule, spin-crossover
catalytic states, Kuramoto sync, morphism chain).

**Output:** Foundation paper for all subsequent monograph papers.

---

### Part II — The P450 Manifold and CYP3A4 (2 papers)

#### **Paper 2: The Cytochrome P450 Address Manifold + CYP3A4 Fold** ✓ DONE

**Path:** `publications/manifold/p450-address-manifold-cyp3a4-fold/`

**Note:** Originally proposed as two papers (sequence manifold + CYP3A4 folding);
merged because the four claims — family clustering at k=3, isoform separation
at k=6, allelic variation at k=9, and CYP3A4 native fold — are truncations of
the same evaluation $\mathrm{eval}_{\mathcal{R}}(\xi_{\mathrm{P450}}^{\mathrm{family}})$
at different recursion depths.

**Thesis:** Apply the Empty Dictionary and ternary encoding to all ~400,000
known P450 sequences (UniProt). Show that the 18-family taxonomic structure
(CYP1–CYP51) corresponds to clustering at trit-depth k=3, that the 57 human
isoforms separate at k=6, and that allelic variants (CYP2D6$^*$1 through $^*$131)
separate at k=9. At full residue-and-structural depth, the same evaluation
folds CYP3A4 (UniProt P08684, 503 residues) against the unliganded crystal
structure (PDB 1TQN). Folding completes in $O(\log_3 N) \approx 6$ categorical
steps with $r \to 0.87$.

**Data:** UniProt, CYPED, Pfam (PF00067), InterPro (IPR001128), PDB 1TQN.

**Validation:** 8/8 PASS (address encoding, manifold density, family clustering,
isoform separation, CYP2D6 allele resolution, CYP3A4 address assembly, Kuramoto
folding trajectory, contact map vs 1TQN).

**Compares against:** AlphaFold2, MD simulation, BLAST/HMMER classification.

#### **Paper 2.5: GLB-Based Structural Input to the Receiver** ✓ DONE *(methods)*

**Path:** `publications/foundations/glb-structural-input/`

**Type:** Methods companion paper. Documents `cytochrome/glb/levinthal_glb/`,
the Python package that bridges binary glTF (GLB) 3D structural files to the
receiver $\mathcal{R}_{\mathrm{bio}}$.

**Thesis:** Public PDB-derived GLB files (Sketchfab, Mol* exports) are how
modern web-deployed structural biology distributes 3D coordinates. By walking
the GLB scene graph and mapping CPK baseColorFactor values to chemical
elements, atomistic ball-and-stick GLBs yield per-atom positions and
elements at no parsing cost — providing a real-PDB-grounded validation
upgrade path for the existing synthetic CYP3A4 validation suites in
Papers 2--5. The package establishes the **five GLB roles** taxonomy
(calibration references, initial conditions, validation targets,
interactive probes, trajectory waypoints).

**Headline result:** On the productive test GLB, the parser auto-detects
Fe coordination at canonical CYP450 distances — Fe-N$_\text{porphyrin}$ at
2.01--2.04~Å, Fe-S$_\text{thiolate}$ at 2.228~Å, Fe-O$_\text{axial}$ at
1.814~Å. The 1.814~Å Fe-O specifically identifies the GLB as modelling
**state 4 (oxy-complex Fe$^{II}$-O$_2$)** of the catalytic cycle —
between Fe-O$_2$ (1.80~Å) and Fe-OOH (1.85~Å), two apertures before
Compound~I (1.65~Å, Paper~5).

**Limitations:** Two of three test GLBs are ribbon-only (smoothed
surface meshes); they yield no atomic resolution. Future work: PDB-companion
RCSB API lookups for ribbon-only GLBs.

**Validation:** 8/8 PASS (parser smoke, CPK decoder, artifact filtering,
Fe coordination shell, state-4 identification, morphism chain, S-entropy
address, five-roles taxonomy).

**Imports:** `levinthal_glb` package (parser, cpk, structure, s_entropy,
rbio modules) at `cytochrome/glb/`.

#### **Paper 3: The Resting and Substrate-Bound States of CYP3A4** ✓ DONE

**Path:** `publications/equilibrium-states/cyp3a4-resting-substrate-bound/`

**Thesis:** Characterise states 1 (resting Fe³⁺-H₂O low-spin) and 2
(substrate-bound Fe³⁺ high-spin) of the catalytic cycle as receiver evaluations,
with worked validation against PDB 1TQN and PDB 1W0E. The 1→2 transition is
**one categorical aperture** ($d_C = 1$) — water displacement, substrate insertion,
and spin-crossover are simultaneous facets of one partition reorganisation, with
explicit partition-depth change $\Delta \mathcal{M} = 0.92$. The +120 mV redox
shift gating CPR-mediated electron acceptance emerges from $\Delta \mathcal{M}$
alone via $\Delta E_{1/2} = (k_B T/e) \cdot n_{\mathrm{eff}} \cdot \Delta \mathcal{M} \cdot \ln b$.

**Imports closed-form functors** $F_{OC}, F_{CB}, F_{BO}$ from triple-isomorphism
architecture, **variance–free-energy identity** $F = k_B T \cdot \sigma^2(\phi)$,
**heme-pocket capacitor** ($C \approx 5.7 \times 10^{-20}$ F, $U \approx 1.4$ eV,
$\tau_{RC} \approx 60$ ps), and **electrostatic chamber** confinement
($|e\Delta\phi|/k_B T \approx 7$).

**Headline numbers:** $\Delta E_{1/2} = 122$ mV (computed) vs 120 mV (Daff 1997
measurement) — 1.7% deviation.

**Validation:** 8/8 PASS (closed-form functors, resting coherent regime,
heme capacitor, water variance free-energy, spin-crossover ΔM, substrate-bound
locked regime, chamber confinement, redox shift derivation).

---

### Part III — The Catalytic Cycle (3 papers)

#### **Paper 4: Observing Electron Transfer through Cytochrome P450 Reductase** — **HEADLINE PAPER**

**Path:** `publications/catalytic-cycle/multi-hop-et-chain/`

**Thesis (the monograph's centre of gravity):** With the protein constructed by
Papers 1–3 (receiver, sequence/fold manifold, resting/substrate-bound states)
and grounded in real PDB coordinates by Paper 2.5, the four-cofactor electron
transfer chain NADPH → FAD → FMN → heme Fe³⁺ is **observed**, not simulated.
The observation is performed by a five-layer instrument stack already specified
in the source papers — a stack that is hardware all the way down. The headline
deliverable is per-frame visualisations of |ψ(r,t)|² across the four hops at
femtosecond resolution, with the Marcus reorganisation energy λ extracted as
hologram observable #5 of the same pipeline that produces the visualisations.

**The five-layer instrument stack:**

| Layer | Apparatus | Source paper |
|---|---|---|
| 5 | 5-pass GPU hologram pipeline → 6 observables incl. Marcus λ | `superimposed-multi-modal-holograms.tex` |
| 4 | Harmonic Molecular Resonator (cycle-rank cross-validation) | `harmonic-molecular-resonator.tex` |
| 3 | Ensemble Strobes (W_Sk fs / W_St ns / W_Se long) | `ensemble-strobes.tex` |
| 2 | Triple-Equivalence Theorem (calibration certificate) | `spectroscopic-derivation-of-elements.tex` |
| 1 | Categorical Spectrometer (CPU/bus/LED/refresh oscillators) | `hyperfine-transitions.tex` |

**Categorical predictions (from $d_C{=}4$, no fitted parameters):**
$k_{\mathrm{cat}}/K_M \sim 10^6$ M⁻¹s⁻¹; per-hop intrinsic rate $\sim 10^9$ s⁻¹;
Newton's-cradle non-identity (NADPH electron and heme-arriving electron are
categorically continuous but not materially identical — an isotope-tracking
falsifier).

**Apparatus-as-experiment validation:** counting-anomaly self-selection
identifies the four cofactor centres as active atoms (precedent: 100% accuracy
on azurin Cu site, [atomic-ternary-spectrometers.tex](publication/atomic-spectrometers/atomic-ternary-spectrometers.tex));
cycle-rank loops in the harmonic molecular resonator give independent
cross-validation channels; Marcus λ recovered from hologram observable #5
matches measured ET reorganisation energy.

**Why this is the headline, not Paper 5:** the monograph's promise was to first
*construct* the protein (Papers 1–3), then *observe* the actual electron
transfer through it (this paper). Compound I formation (Paper 5) is downstream
chemistry that follows the electrons' arrival.

**Data:** BRENDA (CPR/P450 kinetics), published transient absorption
spectroscopy (Murataliev 2004), EPR characterisation of flavin semiquinones
(Narayanasami 1997), PDB structures for CPR (1AMO, 3ES9) and CYP3A4 (1TQN).

#### **Paper 5: Compound I Formation via O-O Heterolysis** *(downstream chemistry of the headline)*

**Path:** `publications/catalytic-cycle/compound-i-formation/`

**Role in the monograph:** What happens *after* the electrons delivered by
Paper 4 arrive at the heme iron. Paper 4 is the headline (the observation
event); Paper 5 is the chemistry that the observed electrons subsequently
drive. The Compound I result is not the monograph's centre of gravity — it
is a demonstration that the framework, having observed the transfer, also
predicts the downstream catalytic chemistry without additional machinery.

**Thesis:** Compound I (Fe⁴⁺=O porphyrin•⁺) is the most controversial intermediate
in all of biocatalysis. Compute its formation as a partition transition involving
simultaneous (a) O-O bond order trajectory from 1 (peroxo) to 0 (cleaved),
(b) proton arrival from I-helix Asp251/Thr252 water network, (c) Fe(III) → Fe(IV)
redox transition, (d) porphyrin radical localisation.

**Method:** Bond-order partition coordinate (new), PCET trajectory (new),
multi-modal hologram for the transition state, **anharmonic Poincaré
non-recurrence** (replaces "energy barrier" picture), R-C-L circuit model for
Fe(IV)=O electronic state.

**Validates against:** Green & Rittle 2010 MCD spectroscopy on CYP119 Compound I,
ENDOR experiments, DFT (multireference difficulties), QM/MM.

**Falsifiable claims:** Specific Compound I lifetime, oxidation potential,
porphyrin radical localisation pattern. Bond-breaking is structurally guaranteed
by anharmonicity, not a rare event.

#### **Paper 6: C-H Activation: Hydroxylation, Epoxidation, and Oxygen Rebound**

**Path:** `publications/catalytic-cycle/ch-activation-rebound/` (planned)

**Thesis:** C-H bond cleavage by Compound I is a three-body trajectory
(substrate C, H, Fe=O). Track the H-atom transfer + oxygen rebound (Groves 1986)
as an S-expression with intermediate substrate-radical state. Cover aliphatic,
allylic, benzylic, aromatic hydroxylations and epoxidation with one mechanism.

**Validation:** Reproduce kinetic isotope effect ranges (KIE ≈ 4–11 typical),
regioselectivity for representative substrates from BRENDA.

---

### Part IV — Reaction Repertoire (2 papers)

#### **Paper 7: Heteroatom Oxidation and Dealkylation Reactions**

**Path:** `publications/reactions/heteroatom-dealkylation/` (planned)

**Thesis:** N-, O-, S-dealkylation, sulfoxidation, N-oxidation, deamination —
all share a common trajectory pattern: Compound I → heteroatom radical cation
→ product. Distinguish single-electron-transfer (SET) vs. hydrogen-atom-transfer
(HAT) pathways as receiver-dependent evaluations.

**Data:** ChEMBL substrate metabolism records, mechanistic studies (CYP2D6, 3A4).

#### **Paper 8: Atypical Reactions and the Reaction Repertoire Atlas**

**Path:** `publications/reactions/atypical-reactions-atlas/` (planned)

**Thesis:** Catalogue all ~30 distinct reactions documented for the P450
superfamily — including rearrangements (e.g., CYP19A1 androgen-to-estrogen
aromatisation, three sequential oxidations), ring contractions, C-C bond
cleavages, dehalogenation, isomerisation. Show each as a sub-expression of
$\xi_{\mathrm{P450}}$ with its specific substrate, receiver, and termination
condition.

**Data:** ENZYME (EC 1.14.13.x, 1.14.14.x), KEGG, MetaCyc, Reactome.

**Output:** Complete reaction atlas — all P450 reactions characterised in one
framework.

---

### Part V — Diversity, Pharmacology, and Disease (2 papers)

#### **Paper 9: The 57 Human Isoforms as Variants of One S-Expression**

**Path:** `publications/diversity/57-human-isoforms/` (planned)

**Thesis:** CYP1A2, CYP2D6, CYP3A4, CYP19A1 (aromatase), CYP51A1
(sterol demethylase) are not separate proteins — they're receiver instantiations
of one $\xi_{\mathrm{P450}}$ with different substrate-channel sub-expressions.
Substrate selectivity emerges from differences in $\xi_{\mathrm{substrate-channel}}$.

**Data:** PDB structures for all human CYPs with substrates, ChEMBL (>100,000
activity records), Human Protein Atlas, GTEx.

**Validation:** Predict substrate preferences for held-out drug-CYP pairs;
benchmark against AlphaMissense / ESM scores.

#### **Paper 10: Polymorphisms, Drug-Drug Interactions, and Inhibitor Design**

**Path:** `publications/diversity/polymorphisms-ddi-inhibitors/` (planned)

**Thesis:** Pharmacogenomic outcomes (CYP2D6 ultra-rapid metaboliser codeine
toxicity; CYP2C9$^*$3 warfarin sensitivity; CYP21A2 congenital adrenal
hyperplasia) are address mutations in $\xi_{\mathrm{P450}}$, predictable from
sequence alone. Drug-drug interactions are competing completion conditions on
a shared receiver. Inhibitor design becomes an inverse-trajectory problem:
specify the desired completion (no Compound I formation), invert to substrate.

**Coherence-severity exponential** $\tau \propto \exp[10 \cdot (r - 0.5)]$
gives quantitative pharmacogenomic phenotype severity from H-bond network
coherence loss; alleles below the BMD on/off threshold (42.1×) are
non-functional.

**Data:** PharmGKB, ClinVar, gnomAD, COSMIC, DrugBank.

---

### Part VI — Construction and Membrane Context (1 paper)

#### **Paper 11: Membrane Anchoring, Cofactor Insertion, and Partner Coupling**

**Path:** `publications/construction/membrane-cofactor-cpr/` (planned)

**Thesis:** Eukaryotic P450s are tethered to the ER bilayer by an N-terminal
helix; heme is inserted post-translationally; CPR docks transiently. Each is
a sub-expression $\xi_{\mathrm{membrane}}$, $\xi_{\mathrm{heme-insertion}}$,
$\xi_{\mathrm{CPR-coupling}}$ with its own receiver. The Miracle Principle
licenses heme insertion as a locally-infeasible step (heme thermodynamically
prefers solution but globally completes the protein).

**Data:** OPM database (membrane orientations), CPR structures (PDB), cryo-EM
density maps (EMDB).

**Validation:** Membrane orientation matches OPM; CPR docking interface
matches published cryo-EM (e.g., Hamdane 2009).

---

### Part VII — Spectroscopic Atlas, Synthesis, and Validation (3 papers)

#### **Paper 12: The Seven States as a Closed Orbit in S-Entropy Space**

**Path:** `publications/synthesis/seven-state-closed-orbit/` (planned)

**Thesis:** States 1–7 of the catalytic cycle form a closed trajectory under
$\mathrm{eval}_{\mathcal{R}}(\xi_{\mathrm{P450},t \in [0, T_{\mathrm{cycle}}]})$.
Each state has a distinct hologram (Paper 13), distinct partition fingerprint,
distinct HMR cycle rank for the heme. **Catalysis is quasi-periodic completion**:
generative-novelty corollary says exact recurrence has Lebesgue measure zero,
so each turnover ends in a slightly-different conformation.

**Imports:** Newton's-cradle proton non-identity, partition-lag conductance,
five-fold operational regimes, anharmonic Poincaré non-recurrence.

**Falsifiable predictions:** Specific Compound I lifetime, peroxo S-entropy
address, Fe spin states at each step, regime classification per state.

#### **Paper 13: The Complete Spectroscopic Atlas of the Catalytic Cycle**

**Path:** `publications/atlas/spectroscopic-atlas/` (planned)

**Thesis:** For each of the seven catalytic states, generate a multi-modal
hologram superposing UV-Vis, Resonance Raman, EPR, ENDOR, Mössbauer, and 2D-IR
signatures. Use ensemble strobes to predict transient lifetimes; use the
categorical spectrometer to predict hyperfine couplings (Fe-57 A-tensor). The
seven holograms together constitute the spectroscopic phenotype of the cycle.

**Data:** Published spectra for resting/substrate-bound states (UV-Vis Soret,
EPR, RR ν4/ν2/ν3, Mössbauer); cryogenic crystallography for Compound 0 and I
(Schlichting 2000, Rittle/Green 2010).

**Validation:** Compute spectra ab initio from $\mathrm{eval}_{\mathcal{R}}(\xi_{\mathrm{state}})$;
compare against published data.

**Output:** Complete spectroscopic atlas as a Zenodo dataset.

#### **Paper 14: Database-Wide Validation: Recovering the P450 Literature**

**Path:** `publications/atlas/database-wide-recovery/` (planned)

**Thesis:** Take every PDB entry, every BRENDA kinetic parameter, every ChEMBL
activity record, every PharmGKB outcome for cytochrome P450, and recover them as
$\mathrm{eval}_{\mathcal{R}}$ of sub-expressions of $\xi_{\mathrm{P450}}$.
Quantify recovery rate (target: >90% within receiver floor $\mathfrak{S}_{\mathrm{floor}}$).
The non-recovered cases become falsifiable predictions: either the framework
is wrong, or the experimental record is.

**Output:** Reproducible benchmark suite for any future framework extension.

---

## Foundation / Source Papers

The cytochrome monograph rests on a stack of framework-level theoretical papers
that establish the receiver, conversion functors, partition-lag formalism, and
the four physical instruments. These are NOT part of the monograph proper but
are cited throughout. Located at `cytochrome/sources/`:

| Source Paper | Provides |
|---|---|
| `spectroscopic-derivation-of-elements.tex` | Bounded phase space axiom, partition coordinates (n, ℓ, m, s), capacity C(n)=2n², Triple Equivalence, categorical-physical commutation |
| `unconstrained-substructures.tex` | Floor Theorem, Triple Equivalence at category level, Unconstrained Subtask Theorem (Miracle Principle), recursive triple decomposition |
| `tripple-isomorphism-architecture.tex` | Closed-form conversion functors $F_{OC}, F_{CB}, F_{BO}$, R-C-L ↔ S-axes, BMD on/off ratio (42.1×), variance–free-energy identity, anharmonic Poincaré non-recurrence, generative novelty |
| `biological-current-flux.tex` | Partition-lag conductance formula, Newton's-cradle proton non-identity, pump as partition-gradient generator, isotope non-transfer prediction |
| `biological-partition-landscape.tex` | Partition depth M as scalar field, $k_{\mathrm{cat}}/K_M \propto 1/(d_C \tau_p)$, $\log_{10}(k_{\mathrm{cat}}/K_M) \approx 10 - d_C$, Arrhenius derivation, SOD1 worked example |
| `cellular-charge-trajectory.tex` | Four-state partition operators (DNA bases), capacitor model ($C \approx 300$ pF for genome), polymerase as $d_C=1$ aperture, electrostatic chamber theorem, three-layer cellular charge architecture |
| `hyperfine-transitions.tex` | Categorical Spectrometer (23 modalities, 70 lines), four hardware oscillators (CPU/bus/LED/refresh), atomic spectral atlas |
| `harmonic-molecular-resonator.tex` | Number-theoretic harmonic proximity, cycle rank, tree entropy, holonomy phase, vibrational graph topology |
| `ensemble-strobes.tex` | Three temporal gates (W_Sk fs / W_St ns / W_Se long), zeptosecond LED timing, three algebraic self-validation channels |
| `superimposed-multi-modal-holograms.tex` | Coherent superposition of state spectra, six derived observables (coupling matrix, Franck-Condon, Stokes, Huang-Rhys, Marcus λ, point group), 5-pass GPU pipeline |

Older publications referenced throughout (located in `models/publications/`):

- `observation-computation-framework.tex` — capstone Triple Equivalence paper, GPU as physical observation apparatus
- `categorical-protein-database.tex` — Empty Dictionary Principle, ternary encoding for amino acids
- `purpose-based-protein-model.tex` — compiled probe / LoRA architecture
- `folding-partition-calculus.tex` — Kuramoto folding, six-pass shader pipeline

Earlier electron-trajectory papers (located in `publication/`):

- `azurin-copper-redox-mechanism.tex` — single-hop electron trajectory tracing
- `superoxide-dismutase.tex` — Cu/Zn SOD redox cycle as categorical mechanics
- `biological-partition-landscape/` — published version of the partition landscape
- `atomic-spectrometers/atomic-ternary-spectrometers.tex` — atoms-as-spectrometers

---

## Directory Structure

```
cytochrome/
├── README.md                                 # This file (master plan)
├── sources/                                  # Foundation papers
│   ├── spectroscopic-derivation-of-elements.tex
│   ├── unconstrained-substructures.tex
│   ├── tripple-isomorphism-architecture.tex
│   ├── biological-current-flux.tex
│   ├── biological-partition-landscape.tex    (also in publication/)
│   ├── cellular-charge-trajectory.tex
│   ├── hyperfine-transitions.tex
│   ├── harmonic-molecular-resonator.tex
│   ├── ensemble-strobes.tex
│   └── superimposed-multi-modal-holograms.tex
├── glb/                                      # Test GLBs + the levinthal_glb package
│   ├── README.md
│   ├── levinthal_glb/                        # Python package (Paper 2.5)
│   │   ├── __init__.py
│   │   ├── cpk.py                            # CPK colour decoder + vdW + S-coords
│   │   ├── parser.py                         # GLB scene-graph traversal
│   │   ├── structure.py                      # contact maps, Fe finder
│   │   ├── s_entropy.py                      # element -> S-coord, trit address
│   │   └── rbio.py                           # R_bio applied to GLB structures
│   ├── test_glb_pipeline.py                  # end-to-end pipeline smoke
│   └── *.glb                                 # three CYP test GLBs
└── publications/                             # Monograph papers
    ├── foundations/
    │   ├── expression-algebra-for-biomolecules/   ✓ Paper 1
    │   │   ├── expression-algebra-proteins.tex
    │   │   ├── references.bib
    │   │   └── validation/
    │   │       ├── README.md
    │   │       ├── run_all.py
    │   │       ├── scripts/                  (8 .py files)
    │   │       ├── results/                  (8 .json files + summary)
    │   │       └── figures/
    │   │           ├── generate_panels.py
    │   │           ├── algebra-captions.tex
    │   │           └── panel_NN_*.png        (8 panels)
    │   └── glb-structural-input/             ✓ Paper 2.5 (methods)
    │       ├── glb-structural-input.tex
    │       ├── references.bib
    │       ├── validation/                   (same structure as Paper 1)
    │       └── figures/                      (8 panels + captions)
    ├── manifold/
    │   └── p450-address-manifold-cyp3a4-fold/   ✓ Paper 2
    │       ├── p450-manifold-cyp3a4-fold.tex
    │       ├── references.bib
    │       └── validation/                    (same structure as Paper 1)
    ├── equilibrium-states/
    │   └── cyp3a4-resting-substrate-bound/   ✓ Paper 3
    │       ├── cyp3a4-resting-substrate-bound.tex
    │       ├── references.bib
    │       └── validation/                    (same structure)
    ├── catalytic-cycle/                      (planned: Papers 4, 5, 6)
    │   ├── multi-hop-et-chain/               (Paper 4)
    │   ├── compound-i-formation/             (Paper 5 - HEADLINE)
    │   └── ch-activation-rebound/            (Paper 6)
    ├── reactions/                            (planned: Papers 7, 8)
    │   ├── heteroatom-dealkylation/
    │   └── atypical-reactions-atlas/
    ├── diversity/                            (planned: Papers 9, 10)
    │   ├── 57-human-isoforms/
    │   └── polymorphisms-ddi-inhibitors/
    ├── construction/                         (planned: Paper 11)
    │   └── membrane-cofactor-cpr/
    ├── synthesis/                            (planned: Paper 12)
    │   └── seven-state-closed-orbit/
    └── atlas/                                (planned: Papers 13, 14)
        ├── spectroscopic-atlas/
        └── database-wide-recovery/
```

---

## Validation Pattern (uniform across all papers)

Each paper has a `validation/` subdirectory with the same layout:

```
validation/
├── README.md                  # documents what is/isn't validated
├── run_all.py                 # driver
├── scripts/
│   ├── _common.py             # shared utilities, constants, S-coords
│   ├── 01_*.py                # 8 numbered validation scripts
│   ├── 02_*.py
│   ├── ...
│   └── 08_*.py
├── results/
│   ├── _summary.json          # aggregate pass/fail report
│   ├── 01_*.json              # per-script JSON outputs
│   └── ...
└── figures/
    ├── generate_panels.py     # produces 8 PNG panels
    ├── *-captions.tex         # LaTeX figure captions
    └── panel_NN_*.png         # 8 panels, 4 charts each, ≥1 3D chart per panel
```

**Run convention:**
- `python run_all.py` → runs all 8 scripts, writes `_summary.json`
- `python scripts/0N_*.py` → runs single script standalone
- `python figures/generate_panels.py` → regenerates all panels from results JSON

**Verdict:** each script returns `PASS` or `FAIL` based on its `checks` dict;
aggregate verdict is `PASS` only if all 8 pass.

**Honesty:** every paper's `validation/README.md` documents what is validated,
what is calibrated (relaxed thresholds), and what is deferred to the production
implementation.

---

## Database Integration Plan

Primary sources (priority order):

1. **PDB** — all P450 structures (~1500 entries), all heme-protein structures
   for cofactor reference (~10,000), cryo-EM density maps via EMDB for
   membrane-bound complexes.
2. **UniProt** — all 400,000+ P450 sequences across kingdoms; SwissProt
   curated entries for the 57 human isoforms.
3. **BRENDA** — kinetic parameters (Km, kcat, kcat/Km) for every characterised
   P450 reaction; substrate/product structures.
4. **ChEMBL** — activity data for P450 inhibitors and substrates (~100,000
   records for human CYPs).
5. **KEGG + MetaCyc + Reactome** — pathway context (steroid biosynthesis,
   drug metabolism, vitamin D, bile acid).
6. **ENZYME + Pfam + InterPro** — taxonomic and functional classification.
7. **PharmGKB + ClinVar + gnomAD + COSMIC** — pharmacogenomic and disease
   variation.
8. **DrugBank + STITCH** — drug-target and drug-drug interactions.
9. **Human Protein Atlas + GTEx** — tissue and cell-type expression.
10. **CYPED** — engineering / mutagenesis data for substrate specificity
    prediction.

Spectroscopic / physical:

11. **NIST ASD** — atomic data (Fe, C, N, O, S, H).
12. **HITRAN** — molecular spectroscopy.
13. **CCDC** — small-molecule crystal structures (substrates, products,
    inhibitors).
14. **CSD + ICSD** — heme and metalloporphyrin structures.

Methodology / literature:

15. **PubMed** — for citation tracing and validation against published
    mechanisms.
16. **Open Targets** — drug-target evidence.

---

## Workflow

1. **Validation and fast prototyping in Python** during paper-writing phase.
   Each paper has a self-contained Python validation suite that emits
   structured JSON.
2. **All papers written and submitted** before production implementation.
3. **Production development in Rust** after the monograph completes.

This ordering prevents premature optimisation and ensures the monograph's
methodological claims are settled before lower-level implementation choices
are baked in.

---

## Cross-cutting Deliverables (post-monograph)

### Software (Rust workspace)

One crate per major method:

- `levinthal-sexpr-bio` — the receiver $\mathcal{R}_{\mathrm{bio}}$ and instrument bindings
- `levinthal-fold` — partition-calculus Kuramoto folding (Paper 2)
- `levinthal-et-trajectory` — multi-hop electron transfer (Papers 4, 5)
- `levinthal-hologram` — multi-modal holograms (Paper 13)
- `levinthal-strobes` — ensemble-strobe measurement (Paper 13)
- `levinthal-hmr` — harmonic molecular resonator (Paper 13)
- `levinthal-cat-spec` — categorical spectrometer (Paper 13)
- `levinthal-p450-monograph` — orchestrator wiring all crates together

Plus Python bindings via PyO3 for accessibility, and a Docker/Apptainer
reproducibility container.

### Data deposition

- Zenodo: complete $\xi_{\mathrm{P450}}$ S-expression in canonical JSON form
- All hologram outputs for the 7 catalytic states (Paper 13)
- All sub-expression decompositions as JSON-serialised recursion trees
- Cross-validation benchmark suite against PDB/BRENDA/ChEMBL (Paper 14)

### Open infrastructure

- Web frontend (Vercel deployment of existing dismutase/shakespear apps)
  where users submit a CYP sequence and receive: full S-expression, predicted
  catalytic cycle holograms, substrate selectivity, falsifiable predictions
- Live integration with PDB and UniProt APIs

### Documentation

- "Levinthal Framework Handbook" companion volume — extracts the framework
  methodology from the cytochrome instance and presents it as a generic
  template for applying the same approach to any enzyme/protein/biomolecule
  (kinases, ribozymes, photosystems, etc.)

---

## Headline Numbers Already Validated

| Result | Computed | Reference / Target | Paper |
|---|---|---|---|
| Floor $\mathfrak{S}_{\mathrm{floor}}(\mathcal{R}_{\mathrm{bio}})$ | 3.43×10⁻⁴ | ~3.7×10⁻⁴ | 1 |
| C(n) = 2n² for n=1..7 | exact | electron shells | 1 |
| 20 amino acids unique at depth 9 | 20/20 | required | 1 |
| Cycle closure n preservation | 100% | required | 1 |
| Manifold density (P450 subregion) | 99% in bounds | predicted | 2 |
| 18 family clusters at k=5 | recall 0.94 | Nelson nomenclature | 2 |
| 57 human isoform separation at k=8 | 0.97 distinctness | required | 2 |
| 13 CYP2D6 alleles separated | 4 phenotypes recovered | PharmVar | 2 |
| CYP3A4 fold $\log_3 N \approx 6$ | 5.69 | predicted | 2 |
| ΔM (Fe LS → HS) | 0.918 | 0.92 | 3 |
| $E_a$ spin-crossover | 14.3 kcal/mol | ~14 | 3 |
| Heme capacitance | 56.7 aF | ~57 | 3 |
| Heme stored energy | 1.39 eV | ~1.4 | 3 |
| RC discharge | 56.7 ps | ~60 | 3 |
| **Redox shift +120 mV** | **122 mV** | **120 mV (Daff 1997)** | **3** |
| Resting state ⟨r⟩ | 0.999 | coherent regime | 3 |
| GLB Fe-N$_\text{porphyrin}$ | 2.01–2.04 Å | 2.0 Å (crystallographic) | 2.5 |
| GLB Fe-S$_\text{thiolate}$ | 2.228 Å | 2.2 Å (crystallographic) | 2.5 |
| GLB Fe-O$_\text{axial}$ (state 4) | 1.814 Å | 1.80–1.85 Å (oxy/peroxo) | 2.5 |

---

## Headline Numbers Pending

| Quantity | Predicted | Paper |
|---|---|---|
| $k_{\mathrm{cat}}/K_M$ for ET chain | ~10⁶ M⁻¹s⁻¹ | 4 |
| Compound I lifetime | sub-ms | 5 |
| Compound I oxidation potential | within experimental error of Rittle/Green | 5 |
| KIE for hydroxylation | 4–11 | 6 |
| Total reaction repertoire | 30 distinct mechanisms | 8 |
| Pharmacogenomic phenotype recovery | >90% | 10 |
| 7-state spectroscopic atlas | matches all RR/EPR/Mössbauer | 13 |
| Database recovery rate | >90% within floor | 14 |

---

## License and Citation

License: same as Levinthal repository root (see `LICENSE`).

Citation (preliminary): Sachikonye, K. F. *Categorical Mechanics of Cytochrome
P450: A Complete Description of Substrate Oxidation in Bounded Phase Space*.
Monograph in preparation, Technical University of Munich. 2026.

Contact: kundai.sachikonye@bitspark.com / kundai.sachikonye@wzw.tum.de
