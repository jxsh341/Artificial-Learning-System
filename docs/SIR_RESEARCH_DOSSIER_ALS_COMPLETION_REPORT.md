# SiR Research Dossier — ALS Completion Report

**Work package:** WP1 — ALS Technical Research Dossier · **Date:** 2026-11-13
**Repository:** `artificial_cns` at HEAD `f901870` (main, clean tree)
**Discipline:** read-only research and documentation. No source code, config, protocol, or
validated artefact (including D800) was modified; no new experiments were run; the pending Stage 3
v2 execution was not inspected and nothing in the dossier assumes its outcome.

## 1. Documents produced

| Document | Content |
|---|---|
| `docs/SIR_RESEARCH_DOSSIER_ALS.md` | the dossier: executive summary, architecture (with Mermaid diagram, SSM equations, module map), capability evidence, the strongest experiments (D800 reference, state-content/substitution, acquisition routes, retention timescales and failure mechanisms, maintenance chain incl. the H8r rejection, M1 continuity and inspection), limitations, and ALS's contribution to SiR in four tiers |
| `docs/SIR_RESEARCH_DOSSIER_ALS_EVIDENCE_REGISTER.md` | claim-by-claim register: source file → artefact → commit → test/validation → classification (1–6), plus test-suite evolution table |
| `docs/SIR_RESEARCH_DOSSIER_ALS_COMPLETION_REPORT.md` | this report |

## 2. Evidence consulted (all read-only)

- `TECHNICAL_REPORT.md` (§1 system summary; §4 memory table; §5 curriculum study; §6/§8h D800;
  §7 state-content; §8b–§8v registered chain — in depth §8h, §8r, §8s; §9 reproduction; §10 M1
  pre-registration and results)
- `README.md` (findings table, position statement, status); `ALS_COGNITIVE_READINESS.md`
  (28-item capability classification, evidence register, executive assessment)
- `docs/ALS_M1_COMPLETION_REPORT.md`, `ALS_M1_LIVE_INSPECTION_DESIGN.md`,
  `ALS_M1_LIVE_INSPECTION_IMPLEMENTATION.md`, `ALS_M1_LIVE_INSPECTION_TEST_REPORT.md`,
  `ALS_M1_HYBRID_INTERFACE_CONTRACT.md` (section D)
- Source: `src/artificial_system/{models/{ssm,gnn_ssm},training/trainer,utils/checkpoints}` (skim
  for exact equations, trainer semantics, execution-state contents)
- Git history: full 84-commit log mapped to experiments; reflog not needed (no incident)
- Result directories verified present (not modified): `results/d800_interleaved/`,
  `results/d800_population/` (`reliability_summary.json`, `unseen_size_corrected.json`),
  `results/m1_resume/` (E1/E1b arms incl. the superseded unpinned-environment arm),
  `results/m1_progress/` (P_seed000..007), plus the §8-series directories
- Live verification performed: full unit suite **180 passed** (47.1 s) — the existing suite only;
  no experiment was executed

## 3. Strongest findings

1. **Causal working memory** (class 1): state substitution flips behaviour (`follows` 1.000,
   zero → chance) while instance evidence is absent from the state; information survives delay in
   23/23 population seeds.
2. **A reliable reference baseline** (class 1): frozen D800 succeeds 20/20 fresh seeds, 23/23
   combined, exact 95% CI [0.852, 1.000], state-reset control at chance in every seed.
3. **Separable stages with predictable applicability** (class 1): representation / utilisation /
   retention dissociate; capture-and-hold works exactly where the state probe predicts it (3/3 vs
   0/6); encoder-side acquisition is a distinct replicated route (3/3).
4. **Deterministic continuity** (class 1): fresh-process resume is bit-exact (E1 4/4; E1b PASS
   after a recorded infrastructure re-run with superseded artifacts retained); live inspection is
   read-only by construction and verified bit-identical to never-inspected twins.
5. **A measured negative-result chain** (class 6, deliberately preserved): first-order direction
   control fails pre- and post-Adam (H8r 0/12; §8s 12/12 representational collapse with a measured
   10.6× curvature discrepancy); tolerance is dynamical, not geometric (§8t); only direct realised
   state-displacement control maintains the rescue (§8q), and online prevention works but at
   adaptation cost (§8v).

## 4. Limitations of the dossier and its subject

- Population-scale evidence exists only for the D800 reference and the 23-checkpoint maintenance
  population; most §8 studies are n = 2–12 per condition on one task family, CPU scale (≤ 1,000
  nodes, ≤ 10k instances).
- The M1 models used for continuity/progress probing are maintenance-regime (delayed accuracy
  0.34–0.66 after the 400-episode registered budget), not ceiling models — the phase-decoding
  result is about representation structure, not peak competence.
- One documented scale-generalisation outlier (population seed 19, edge-density calibration).
- Encoder-side acquisition beyond its one checkpoint is unreplicated territory (flagged in-record).
- The dossier inherits the report's own scope caveats: state-reset necessity ≠ a general
  working-memory faculty; no claim is made beyond registered bounds.

## 5. Unresolved questions

1. Per-instance/accepted-candidate guard fidelity (§8v E4's measured 5.55-unit prediction gap) —
   can prevention be obtained without the adaptation freeze?
2. How far does encoder-side acquisition generalise beyond its single checkpoint?
3. Which side of a joint stage supplies the destabilising gradient at delay 30 (§8m, open by
   design)?
4. Long-delay maintenance beyond the measured band (delay ≥ 30); delay-100 attractor contraction
   — can dispersion be preserved under any update rule that still adapts?
5. A causally-used progress/goal representation (M1-c shows the current phase signal is
   correlate-only).
6. Reward-driven action selection (no RL pathway is active; `policy_gradient_loss` unused).
7. Population-scale studies for the intervention chain (H7b/B2/§8q are 3–12 checkpoint studies).

## 6. Statement of scope

The dossier documents ALS as it is evidenced at `f901870`. It makes no claims about SiR's overall
capabilities, and does not represent ALS as an intelligent system. It treats the architecture's
*potential* as unproven unless a registered experiment supports it. No further action was taken
beyond documentation.
