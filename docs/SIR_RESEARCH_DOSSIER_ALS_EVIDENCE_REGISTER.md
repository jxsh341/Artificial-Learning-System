# SiR Research Dossier — ALS Evidence Register

Companion to `docs/SIR_RESEARCH_DOSSIER_ALS.md`. Every major claim is linked to source file,
experiment artefact, commit hash, test/validation, and evidence classification.

**Classification legend:**
1 = Experimentally demonstrated · 2 = Implemented but not experimentally demonstrated ·
3 = Partially demonstrated · 4 = Proposed · 5 = Not demonstrated · 6 = Negative or rejected result

All results live under `results/` (gitignored by repo convention — regenerable from committed
configs + seeds; numbers are recorded in `TECHNICAL_REPORT.md` and `README.md`). Commits listed are
`artificial_cns` history at HEAD `f901870` (main). Test counts are cumulative suite sizes at the
stated commits (150 → 165 → 180; current count 180 passed, 2026-11-13).

---

## A. Architecture and internal-state machinery

| Claim | Class | Source file | Artefact / experiment | Commit | Test / validation |
|---|---|---|---|---|---|
| `GraphData` is the single stable interface; adapters integrate new sources without model rewrites | 1 | `data/schema.py`, `data/adapters.py` | all experiments consume it | `04c919e` | `tests/test_data_schema.py`; full pipeline tests |
| GNN message passing is *necessary* for relational answers (M1/M6 ablations at chance) | 1 | `models/gnn.py`, `models/gnn_ssm.py` (M6) | ablation matrix; degree-preserving relabelled control 1.000; unseen size 1.000 | `04c919e` | §3.3 report; rewired/degree controls in `data/perturbations.py` |
| SSM state is *necessary* for bridging (state reset → chance) | 1 | `models/ssm.py` (`reset_state`, mask) | `delayed_reset_cleared` 0.480–0.515 in every seed | `04c919e` | `tests/test_ssm.py` (12), `test_training_smoke.py` reset test |
| Spectral-radius-bounded `A` (diagonal/dense parameterisations) | 1 | `models/ssm.py` | enforced by construction | `04c919e` | `tests/test_ssm.py` bounded-decay test |
| Mappings live in weights; durable across delay 500+ | 1 | `training/trainer.py` | memory curves; retention@500 = 1.0 | `04c919e` | `evaluation/memory_curve.py`, `tests/test_perturbations.py` |
| Episodic BPTT, one Adam step per episode; supervised CE only (reward recorded, unused) | 1 | `training/trainer.py`, `training/losses.py` | all registered runs | `04c919e` | `tests/test_training_smoke.py` (4) |
| Two trainers share one optimizer; interleaved condition-D schedule | 1 | `experiments/runner.py` | curriculum study | `fa616d3` | `tests/test_curriculum_study.py` (5) |
| Ablation matrix M1–M6 (composable builders) | 1 | `models/` (builders) | §1/§3 ablation results | `04c919e` | model unit tests |
| Checkpoints embed environment metadata incl. git commit | 1 | `utils/checkpoints.py` (`collect_env_info`) | every experiment dir | `04c919e` | `tests/test_checkpoints.py` |

## B. Working memory / state content (the core capability)

| Claim | Class | Source file | Artefact / experiment | Commit | Test / validation |
|---|---|---|---|---|---|
| Answer decodable from state at every trial point (held-out ridge ≈ 1.000 vs majority 0.55) | 1 | `evaluation/state_content.py` (`decode_at_points`, `ridge_decode_cv`) | `results/*/state_content/` | `623fdc7` | `tests/test_state_content.py` (perfect-vs-shuffled baselines) |
| State carries the *answer*, not the evidence (features absent; sparse code, 3/8 dims) | 1 | `evaluation/state_content.py` (`avf_*`) | §7B | `623fdc7` | same |
| State substitution is causal: diff-answer donation **followed** 1.000; zero → chance | 1 | `evaluation/state_content.py` (`state_substitution`) | §7D; §8l/§8g rubrics | `623fdc7` | substitution-accounting unit tests |
| Retained state is flexible (update on new/flipped evidence 1.000; ignore irrelevant 1.000) | 1 | `evaluation/` counterfactual probe | §5 Phase D, all successful seeds | `fa616d3` | registered protocol |
| Decodability ≠ use: failure phenotypes (not-computed / decays / present-but-unused) | 1 | §7C analysis | failed seeds carry decodable-but-unused codes | `623fdc7` | §7 per-seed CSVs |
| Retention timescales: perfect ≤ delay 10 (23/23); delay 30 info 23/23, use 11/23 (bimodal, confidently wrong); delay 100 info gone | 1 | `scripts/run_delay_scaling.py`, `evaluation/` | `results/delay_scaling/`, `delay_transition/` | `a07f76f`, `74671c8` | registered eval-only protocol; delay-30 rows reproduced to 0.000 in §8o |
| Delay-30 failures = displaced decision boundary (rotation/decay rejected) | 1 | `evaluation/readout_geometry.py` | `results/readout_geometry/` | `d2e1a24` | geometry unit tests (`be2ca98`); cos 0.982 vs 0.980 |
| Delay-100 mechanism = attractor contraction (dispersion >2000× at pinned norm; amplitude ~730× down; ordering survives) | 1 | `scripts/run_delay_transition.py` | `results/delay_transition/` | `74671c8` | per-dimension variance-floor correction `23a86dc` |
| Behaviour and function are the same event (R1 confirmed, R2 empty; information to 50/65) | 1 | `scripts/` transition-functional runner | `results/transition_functional/` | `ed7bb56` | full substitution battery, delay-3 anchor exact |

## C. Acquire/utilise/retain — interventions

| Claim | Class | Source file | Artefact / experiment | Commit | Test / validation |
|---|---|---|---|---|---|
| Interference measured: naive delayed-only updates → collateral damage 0.735–1.000 | 1 | `experiments/coupling_rescue.py` | `results/delay30_isolation/` etc. | `5cc13e6` | registered doses; per-run CSVs |
| H5: readout-only training restores utilisation (frozen generation, unchanged geometry); scrambled control clean | 1 | `scripts/run_coupling_rescue.py` | `results/coupling_rescue/` | `769236d` | pre-registered rubric (§8b) |
| Coupled readout is transient (~20-episode oscillation; anti-phase gradient cosine) | 1 | §8c–8d diagnostics | `results/coupling_rescue_h6/` | `1ed225d` | registered; correction preserved (§8d) |
| H7b alternating restore: full rubric PASS on both phenotype-iii sources (immediate 1.000, delayed 1.000, follows 1.000, geometry preserved) | 1 | `experiments/coupling_rescue.py` (alternating mode) | `results/coupling_rescue_h7b/` | `6a4da64` | full pre-registered rubric; §8f corrections preserved |
| Applicability predicted in advance: present-but-unused 3/3 vs answer-absent 0/6 | 1 | state probe → capture gate | `results/coupling_rescue_h7b_generalisation/` | `b939367` | exact CI reported (0.292–1.000) |
| B2 encoder-side acquisition: readout exactly frozen, decode 0.53→1.000, 3/3 data orders | 1 | `coupling_rescue.py` (`data_seed`) | `results/coupling_rescue_b2_replication/` | `0c6c532` | head/C/D deltas exactly 0.000; full rubric |
| One answer-absent checkpoint passed via encoder-side route — **flagged, single observation** | 3 | same | §8i | `b939367` | recorded as needing replication |

## D. Reliability reference (D800)

| Claim | Class | Source file | Artefact / experiment | Commit | Test / validation |
|---|---|---|---|---|---|
| D800 original: interleaved 800-episode 3/3 ≥ 0.9; GNN-frozen staged 3/3 (immediate rule sacrificed) | 1 | `configs/experiments/d800_interleaved.yaml` | `results/d800_interleaved/` | `7b363f3` | pre-registered *observed* criterion (≥90% in sample) |
| D800 population: 20/20 fresh seeds; 23/23 combined; exact CP 95% CI [0.852, 1.000]; state reset at chance in every seed | 1 | `configs/experiments/d800_population.yaml`, `scripts/d800_population_analysis.py` | `results/d800_population/reliability_summary.json` | `36f4b11` | CP bisection verified against known values |
| Unseen size: 22/23 ≥ 0.985; seed 19 = edge-density calibration gap (degree-matched 1.000) | 1 (with documented exception) | `scripts/recompute_population_unseen_size.py` | `results/d800_population/unseen_size_corrected.json` | `36f4b11` | corrected protocol recomputed for all 23 |
| Earlier unseen-size anomaly = evaluation-protocol defect; controlled size sweep flat | 1 | `scripts/run_size_sweep.py` | `results/*/size_sweep/` | `fb12872` | degree-matched sweep |

## E. Maintenance under adaptation (incl. the H8r rejection)

| Claim | Class | Source file | Artefact / experiment | Commit | Test / validation |
|---|---|---|---|---|---|
| State-budget line search: ε=0.2 → 7/12 full rubric with real adaptation; ε=0.05 → 11/12 near-frozen; decode 1.000 × 24 runs | 1 | `scripts/run_state_budget.py` | `results/delay30_maintenance/` | `0384db7` | registered; displacement metric recorded pre-implementation (`ea72729`) |
| ε=0.2 failures are access oscillation, not collapse | 1 | §8q trajectories | same | `0384db7` | transition classification |
| **H8r: state-Jacobian-projected delayed gradients (pre-Adam, full magnitude) — REJECTED 0/12** (outcome E 11/12 + seed-17 decode-intact access collapse) | 6 | `scripts/run_jacobian_projected.py` | `results/delay30_jacobian/` | prereg `10624bc`; verification `f396b64`; results `cf6cc35` | projection nulling verified (2.7e-05) before results; per-step logs |
| H8r mechanism: Adam re-magnifies the state-null residual (GNN delta 1.17–1.72 > unconstrained 0.85–1.15); displacement 0.70 after one step | 1 (mechanism of a rejected hypothesis) | same | same | `cf6cc35` | per-step ratio/displacement records |
| §8s: post-Adam step projection also rejected 12/12 (10.6× curvature at step 1; residue concentrated along J's top singular direction) | 6 | `scripts/run_step_projected.py` | `results/delay30_step_proj/` | prereg `4958139`; results `ba1447c` | micro-replication exact (0.1239 = 0.1239) |
| §8t: maintenance tolerance is dynamical, not geometric (all registered geometric tests fail; seed-14 held-out refutation 25–50×) | 6 (H8t rejected) | `scripts/run_tolerance_profile.py` | `results/delay30_tolerance/` | prereg `a3c9d38`; results `8cdca7a` | Spearman/LOO per registration; named held-out case |
| §8u: access oscillation = state-cloud translation across frozen readout boundary; \|bvm\|<0.5 rule 409/409; un-instrumented control bit-identical | 1 | `scripts/run_oscillation_mechanism.py` | §8u records | prereg `fa01d4d` (+amend `7f01c0d`); results `0768917` | M1/M3 letter-vs-spirit verdicts recorded; runner fixes logged |
| §8v: V1 detection PASS (0/1,212 episodes); V2 prediction FAIL (sens 0.410); V3 prevention PASS at adaptation cost (10/12 zero crossings; seed 14 protected; freeze bottleneck) | 1 + one registered FAIL | `scripts/run_online_fragility.py`, `scripts/analyze_online_fragility.py` | §8v records | prereg `c2406d3`; amendments `c537528`/`ba4edc4`/`ec31f14`; results `388e452` | thresholds fixed pre-run from §8u gap; amendments recorded before affected results used |

## F. Execution continuity and inspection (M1)

| Claim | Class | Source file | Artefact / experiment | Commit | Test / validation |
|---|---|---|---|---|---|
| Execution-state checkpoint stores params + optimizer + 3 RNG streams + env stream states + step counters | 1 | `utils/checkpoints.py` (`save_execution_state`), `training/trainer.py`, `environment/*.state_dict` | `results/m1_resume/E1_seed00X__*/` | `dd4907f` | 12 dedicated unit tests (incl. bit-exactness) |
| **E1: episode-boundary resume in a fresh process reproduces the trajectory bit-for-bit — 4/4 PASS** | 1 | `scripts/run_resume_exactness.py` | `results/m1_resume/` (control+resume per seed) | prereg `dd4907f`; results `1726fb3` | serialized-float equality on 4-value battery; scored exactly as registered |
| E1b: mid-episode interrupt → episode restart — PASS after recorded infra re-run (unpinned BLAS first attempt superseded, artifacts retained) | 1 | same | `E1b_seed000__resume_midep(+__unpinned_env_superseded)` | infra note `5f48d3b`; result `1726fb3` | bit-identical snapshot under pin; registered semantics |
| SystemHandle (checkpoint mode): probe/inspect_state/summary; legacy checkpoints need explicit dims | 1 | `inspection.py` | M1-b | `dd4907f` | unit tests; suite 165 green |
| **LiveSessionHandle: detached frozen deep-copy snapshot; read-only by construction; frozen-while-live-advances** | 1 | `inspection.py` (`LiveSessionHandle.capture`) | 15/15 live-inspection tests | `55d7086` | bit/hash-exact tests incl. live-vs-checkpoint agreement; suite 180 green |
| `parameter_digest` = sha256 over sorted state dict; `captured_digest` identifies the execution version | 1 | `inspection.py` | same | `55d7086` | digest tests |
| Supported boundaries: between trainer calls; registered mid-episode pause (`pause_after_steps`) | 1 | `trainer.py` | same | `55d7086` | boundary tests |
| Live inspection is observability only — no persistent-memory claim (explicit disclaimer in `summary()`) | 1 | `inspection.py` | `ALS_M1_HYBRID_INTERFACE_CONTRACT.md` §6 | `f901870` | contract doc |
| **M1-c PC1: trial phase decodable from state — 8/8 (Δ +0.55…+0.63 over 0.15 threshold)** | 1 | `scripts/run_progress_probe.py`, `scripts/analyze_m1_progress.py` | `results/m1_progress/P_seed000..007/` | prereg `dd4907f` (machinery `4975ec7`); results `64e3978` | registered ridge-CV protocol; r-decode sanity check |
| **M1-c PC2: phase decodable-but-not-causally-used (2/8) — correlate, not cause** | 1 (measured non-use) | substitution at pre_response | same | `64e3978` | registered paired treatment/control rule |

## G. Explicit non-capabilities

| Claim | Class | Basis |
|---|---|---|
| General reasoning / compositional inference chains | 5 | no compositional multi-hop task or reasoning probe exists; readiness audit item 22 ABSENT |
| Multi-step autonomous planning / agent loop | 5 | action = argmax; no planner/goal store/arbitration (code search, readiness §B item 28) |
| Persistent memory across executions | 5 | state resets per episode by design; no cross-run store (readiness item 9/7) |
| Autonomous goal management; causally-used progress representation | 5 | M1-c PC2 shows phase is correlate-only (2/8); no goal variables exist |
| Reward-driven action selection | 5 | reward recorded, never in loss; `policy_gradient_loss` unused |
| Complete deterministic mid-episode continuation | 5 | registered contract is episode restart; live autograd graph not serializable (recorded rationale) |
| `verify_reproducibility.py` regenerates published metrics 12/12 | 1 | `scripts/verify_reproducibility.py` (commit `7b363f3`) |
| H8r premise (useful admissible directions at full step size) | 6 | §8r/§8s chain (see E) |
| General intelligence / broader inference from architecture | 5 | one task family, CPU scale, registered scope caveats |

## H. Test suite evolution

| Commit / stage | Suite size |
|---|---|
| `04c919e` (initial platform) | 117 |
| WP1 audit (`56fbaad`, 2026-11-10) | 150 |
| M1-b (checkpoint-mode query API) | 165 |
| Live inspection (`55d7086`/`f901870`) | 180 |
| Current (2026-11-13) | **180 passed** (47.1 s) |
