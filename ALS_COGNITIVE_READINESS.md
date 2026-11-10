# WP1 — ALS Cognitive Capability Readiness Audit

**Deliverable:** `ALS_COGNITIVE_READINESS.md` · **Owner:** ALS development agent · **Date:** 2026-11-10
**Repo audited:** `artificial_cns` (HEAD `388e452`, main, pushed) · **Audit type:** read-only — no source, test, config, or result files were modified.
**Verification baseline:** 150 pytest cases passing at audit time; `TECHNICAL_REPORT.md` (29 sections, §1–9 incl. 21 registered experiment sections §8b–§8v); `README.md` findings table cross-references every headline claim.

---

## A. Executive assessment

ALS (`artificial_cns`) is a **validated research substrate for one specific cognitive function: task-dependent internal state maintenance**. It is a GNN→SSM system (`GraphTemporalModel`) that computes a compressed relational answer at stimulus presentation and holds that answer in a fixed-size recurrent state `[batch, state_dim]` through up to ~100 blind (distractor-blank) steps, with the architecture, data interface, and readout all frozen under a registered reference configuration (D800).

What makes ALS unusually trustworthy as a *contributor* to a unified system is its measurement discipline: ~28 pre-registered experiments (§8b–§8v) with amendment protocols, oracle baselines, control arms that eliminate shortcuts, bit-exactness checks, and a 12–23-seed population record. Headline verified capabilities:

- **Causal working memory:** the state does not merely *correlate* with the answer — state-substitution experiments causally flip behaviour (`subst_diff_follows` = 1.000 in the strongest checkpoints; information survives delay in 23/23 population seeds). The state is *queried* (state capture/probe/decode machinery everywhere), *modified* (substitution, masked reset), and *restored* (capture-and-reinstate across delays).
- **Fast, precise, safe adaptation:** supervised episodic learning with a measured tolerance profile (per-checkpoint τ), a measured interference problem (collateral damage 0.735–1.000 for naive multi-task updates), and measured mitigation mechanisms (freeze partitions, rehearsal schedules, balanced staging, per-step state-budget line search).
- **An online access/fragility instrument:** the |bvm| boundary-position statistic classifies the functional state on 0 violations / 1,212 probed episodes across a 12-checkpoint population, and a bvm-guarded update rule prevented functional access loss in 10/12 population checkpoints (incl. one checkpoint previously classified as unrescuable).

What ALS is **not**, today: it has no persistent (cross-run) memory store, no goal/progress representation, no reward-driven action selection (rewards exist in the environment and are recorded but unused for learning), no planner, and no demonstrated multi-step *reasoning* (as distinct from multi-step *computation*, which is demonstrated). Also, its checkpoint format omits RNG state, so "resume mid-episode exactly" is not currently guaranteed.

**Issuer readiness verdict:** ALS can contribute today as the **internal-state / adaptive-maintenance subsystem** of a minimal cognitive loop — the working-memory and gated-adaptation roles — with high experimental credibility. It cannot yet contribute to persistent memory, goal-directed action, or reasoning-linked operations without the minimal additions listed in §E.

---

## B. Capability inventory

| # | Capability (WP1 wording) | Classification | One-line justification |
|---|---|---|---|
| 1 | Internal state representation | **DEMONSTRATED** | Fixed-size SSM state; captures, decodes, and *causally* drives behaviour (state substitution) |
| 2 | Structured vs distributed state | **DEMONSTRATED** (with a caveat) | Distributed within the recurrent pool; decode machinery + geometry analyses expose its content structure |
| 3 | Preserving intermediate computational states | **DEMONSTRATED** | Answer computed at presentation is maintained through 90–100 distractor steps (delay-30 family; §8k–§8q) |
| 4 | State queried, modified, restored | **DEMONSTRATED** | `capture_trial_states`, ridge decode, masked reset, donor-state substitution + reinstatement |
| 5 | State reusable across tasks | **PARTIALLY DEMONSTRATED** | One state serves two tasks (immediate + delayed rule) in the same population; but only within the registered task family |
| 6 | State represents goals / dependencies / task progress | **ABSENT** | No goal/progress variables exist anywhere in the codebase; state encodes answer & task identity only |
| 7 | Memory mechanism (persistent store) | **ABSENT** | Confirmed by search: no replay buffer / memory store in `src/`; the only recurrent store is the SSM state |
| 8 | Memory-like behaviour (within-episode retention) | **DEMONSTRATED** | Memory curves, retention ratios, delay sweeps 3→5,000 steps, causal substitution |
| 9 | Demonstrated useful *persistent* memory (survives execution/runs) | **ABSENT** | State is reset per episode by design (`initial_state` per episode) |
| 10 | Selective retrieval (learned cue→content addressing) | **PARTIALLY DEMONSTRATED** | Retrieval = the state itself carrying the cued answer across delay; no selective archive retrieval exists |
| 11 | Learn from feedback | **PARTIALLY DEMONSTRATED** | Supervised CE loss only; environmental reward recorded but unused; `policy_gradient_loss` exists, unused |
| 12 | Learning persists across episodes | **DEMONSTRATED** | Weights update across the whole 800-episode curriculum; checkpoint-captured states resume learning |
| 13 | Transfer to subsequent tasks | **PARTIALLY DEMONSTRATED** | Generalization transfer-gap measured (rule-level held-out cues); cross-task interference is the known pain-point |
| 14 | Correct previous failures | **PARTIALLY DEMONSTRATED** | Reversal-learning measured (trials-to-criterion post-reversal); no explicit error-driven meta-learning |
| 15 | Catastrophic interference | **DEMONSTRATED (as a measured phenomenon + measured mitigations)** | Collateral damage 0.735–1.000 quantified (§8h/§8i/§8n); freeze/rehearsal/balanced/budget mitigations measured |
| 16 | Learning ≠ memorisation | **DEMONSTRATED (within registered scope)** | Ridge-decode CV on held-out splits, held-out rule-conforming cues, unseen-size probes (1 known anomaly, documented) |
| 17 | Maintain intermediates across successive operations | **DEMONSTRATED** | Delay-30 maintenance incl. under adaptation (state-budget ε=0.2: 7/12 with real adaptation, decode 1.000 ×24) |
| 18 | Execute dependent computational steps | **DEMONSTRATED (computational, not 'reasoning')** | Delay+1 trial structure; response only valid at final step given presentation-time computation |
| 19 | Represent relations between multiple pieces of information | **DEMONSTRATED** | Connectivity/adjacency graph tasks with rewired + degree-matched controls eliminating level statistics |
| 20 | Modify behaviour from intermediate results | **PARTIALLY DEMONSTRATED** | Per-step bvm-guard in §8v conditions future updates on measured state (an engineering control, not autonomous behaviour) |
| 21 | Tasks beyond immediate input→output mapping | **DEMONSTRATED (classification-with-delay)** | Delay-30 tasks; explicitly NOT evidence of open-ended reasoning |
| 22 | Multi-step *reasoning* (compositional inference chains) | **ABSENT** | No compositional multi-hop tasks, no reasoning probes registered; §4 of the report makes no such claim |
| 23 | Save/restore internal state | **DEMONSTRATED** | Checkpoints (model+optimizer+step+config+env) + capture/reinstate of neural state within trials |
| 24 | Task progress survives interruption | **PARTIALLY DEMONSTRATED** | Mid-training resume is standard practice (captured checkpoints → 100+ further episodes); but no RNG/env-stream state saved |
| 25 | Recover objectives/intermediates; resume without restart | **PARTIALLY DEMONSTRATED** | Resumed-run results (§8b–§8v) start *from* saved checkpoints without re-learning from scratch; intra-episode resume ABSENT |
| 26 | Recovery preserves behavioural equivalence | **DEMONSTRATED** (for the covered case) | `verify_reproducibility.py`: 12/12 seeds reproduce recorded metrics within TOL = 1e-6 from checkpoint+config alone |
| 27 | Adaptive *self-regulation* of learning rate/behaviour | **PARTIALLY DEMONSTRATED (as engineered mechanisms)** | Tolerance-matched ε, budget line search, bvm-guard — external, registered, measured; not autonomous meta-cognition |
| 28 | Action selection / planning / agent loop | **ABSENT** | Only `argmax` over logits; no planner, goal store, or arbitration (confirmed by code search) |

Classification legend: DEMONSTRATED = implemented + evidence-backed by a registered experiment or test; PARTIALLY DEMONSTRATED = implemented with evidence but bounded/scope-limited (bounds stated in §C); IMPLEMENTED BUT UNTESTED = nothing currently qualifies (no such entry needed); ABSENT = not present or not evidenced.

---

## C. Evidence register

Reproduction root: `cd artificial_cns`; package code lives in `src/artificial_system/`; registered experiment runners + scorers in `scripts/`; all results on disk under `results/` (gitignored by convention — the report and README carry the numbers); documentation: `TECHNICAL_REPORT.md` §1–9 (29 sections), `README.md` findings table + status. **150 pytest cases passing; no experimental claim in this section is unverified against its record.**

### C.1 State representation (Investigation 1)

| Claim | Source files / functions | Tests | Experimental results | Limitations |
|---|---|---|---|---|
| Fixed-size recurrent state, carries across time | `models/ssm.py` (`StateSpaceModel`, `step`, `forward`, `initial_state`, `reset_state(state, mask)`), `models/gnn_ssm.py` (`GraphTemporalModel.step` returns `(logits, new_state, aux)` with `aux["state"]`) | `tests/test_ssm.py` (12: state carryover makes output history-dependent, step≡forward unroll, distant-input influence, bounded decay, partial reset), `tests/test_gnn_ssm.py` (6) | §1/§3 of TECHNICAL_REPORT (base curriculum results) | state_dim fixed at config (64); no added dimensions permitted (frozen reference D800) |
| State externally capturable at fixed trial points | `evaluation/state_content.py::capture_trial_states` (pre_info / post_present / delay_1..D / pre_response); `evaluation/retention.py::probe_mapping` (returns hidden states [T,B,H]) | `tests/test_state_content.py::TestCaptureTrialStates` (shape + determinism) | §7 state-content analysis; every §8l+ experiment's per-episode CSV columns | Capture is a diagnostic overlay; it does not (and is registered not to) alter the run — §8u pure-control bit-identity |
| State content decodeable (queried) | `state_content.decode_at_points` / `ridge_decode_cv` (5-fold held-out) | `tests/test_state_content.py` (perfect-vs-shuffled-vs-majority baselines) | §7: answer decodable above majority at delay points; size-sweep norm/attention stats | Decodability ≠ use — hence the causal test below |
| State causally drives behaviour (modifiable) | `state_content.state_substitution` (donor states: same_answer / diff_answer / shuffled / zero) | `tests/test_state_content.py` (substitution accounting, donor-label pairing) | **subst_diff_follows = 1.000** in strongest checkpoints (§8g/§8l record); §8u: substitution flips access ON/OFF throughout, decode 1.000 × 477 dense probes | Substitution studied at presentation and delay points, not arbitrary code paths |
| State reset semantics (full + masked) | `ssm.reset_state(state, mask)`; trainer resets state per episode | `test_ssm.py::test_state_reset/test_partial_state_reset`; `test_training_smoke.py::test_state_resets_between_episodes` | delayed-reset control arms in every maintenance experiment (intact-vs-reset gap is the memory signature) | Reset masks are positional, not semantic ("forget this fact") |
| State task-specific vs reusable | single state serves immediate + delayed rules; `size_sweep._probe_size` intacts vs resets; `coupling_rescue.set_trainables` (gnn / pool / ssm A_raw,B / C / D / head partitions) | `test_coupling_rescue.py` (13), `test_size_sweep.py` (6) | §8c/§8e separation-of-loci results; §8n isolation arms | Reuse demonstrated only across the registered two-task family |
| Goals / dependencies / progress | **no implementation exists** (code search: no goal/plan/agent modules) | — | — | ABSENT — see §D |

### C.2 Memory and retrieval (Investigation 2)

| Claim | Source | Tests | Results | Limitations |
|---|---|---|---|---|
| No persistent store beyond SSM state | searched `replay\|buffer\|memory\|persistent` across `src/`: only static `readout.py` register_buffer (node-index), `TransformerTemporal._cache` (transient ablation), H6 rehearsal *schedule* (`experiments/coupling_rescue.py`) — no data store | n/a | — | ABSENT by design (architecture freeze) |
| Memory-like behaviour | `evaluation/memory_curve.py::run_memory_curve` (delay sweep 0…5,000), `evaluation/retention.py::run_retention` (retention_ratio vs delay-0) | `tests/test_perturbations.py::TestMemoryCurve` (12 incl. curve rows + repeat shrinking); `test_metrics.py` (11: retention_ratio, trials_to_criterion, CIs) | memory_half_life + retention curves recorded per seed (README §4 consolidated memory table) | Curve is within-episode only |
| Demonstrated useful memory (within-episode) | state substitution + delayed-intact vs delayed-reset (`evaluation/size_sweep.py`) | `test_state_content.py`, `test_size_sweep.py` | delayed intact ≫ delayed reset (population bimodal, §8k); **information survives delay in 23/23 seeds** (README findings) | — |
| Characteristics of retention | `run_memory_curve` memory_half_life; `run_delay_scaling.py` (delay 3/10/30/100) | `test_metrics.py` | §3: bimodal population at delay 30 — 11/23 behavioural, 23/23 information-retaining | Bimodality = the fragility finding; §8t tolerance profile explains it (dynamical, not static tolerance) |
| Retrieval is learned, fixed, or external | retrieval = the cued answer held in state; cue→mapping learned (`environment/tasks.py` onehot/hierarchical families); readout is a frozen selected-node projection (`models/readout.py`) | `test_environment.py` (13), `test_gnn.py` (9) | cue-set generalization (`evaluation/generalization.py::probe_cue_set`, transfer_gap recorded) | No archive/associative recall; no selective external retrieval |
| Prior knowledge influencing later episodes | episodic weight updates; H7b capture-and-hold alternating-restore (`experiments/coupling_rescue.py` alternating events mode) | `test_coupling_rescue.py` | §8l: stage-A→stage-B carry into delay-30 regime; balanced staging (§8m) | Carry is *weight-level*, not state-level |
| Update without corrupting unrelated info | freeze partitions + collateral-damage measurement (`experiments/coupling_rescue.py`; `scripts/run_isolation_arms.py`) | `test_coupling_rescue.py::set_trainables` | **collateral damage 0.735–1.000** for naive delayed-only updates (§8h/§8i/§8n) — corruption is real and quantified; mitigations (freeze readout, rehearsal, balanced stage-4, state-budget ε) measured | No mechanism fully eliminates interference; §8v shows safety is bought with adaptation freezing |

### C.3 Adaptive learning (Investigation 3)

| Claim | Source | Tests | Results | Limitations |
|---|---|---|---|---|
| Learning mechanism | `training/trainer.py::Trainer.train` — episodic BPTT, Adam, CE loss, per-episode reset; curricula in `experiments/runner.py::run_curriculum_study` (A immediate / B delayed / C staged / D interleaved, budget-matched) | `test_training_smoke.py` (4: loss decreases, pipeline e2e) | §5 curriculum study (4 conditions); population families §8k+ | Supervised only: reward is recorded, never in the loss (`policy_gradient_loss` in `training/losses.py` exists, unused) |
| Feedback learning | label-derived loss only | as above | acquisition metrics (`evaluation/acquisition.py::run_acquisition`: first/last-10% delta, trials-to-criterion) | No reward/RL pathway active |
| Persists across episodes | weights persist across 800-episode chains; checkpoints resume training (`utils/checkpoints.py`) | `verify_reproducibility.py` | §8b–§8v runs all start from **captured checkpoints** and train further | See C.5 for resume gaps |
| Transfer | `evaluation/generalization.py` (held-out cues obeying the same rule) | `test_curriculum_study.py` (5) | transfer_gap recorded per condition | One documented open gap: a single population seed with an unseen-size failure (README open items) |
| Correct previous failures | `evaluation/reversal.py::run_reversal` (flip mapping; measure `residual_preference_old_mapping`, trials-to-criterion post-reversal) | `test_metrics.py::residual_preference` | reversal.csv recorded under cue-learning phase (§2, toy_small/base dirs) | Standard relearning; no meta-level error correction |
| Catastrophic interference | measured §8h/§8i/§8n; mitigations: H6 freeze-vs-rehearsal (`results/coupling_rescue_h6`), H7b alternating-restore (`results/coupling_rescue_h7b*`), balanced stage-4 (`results/delay30_balanced`), gradient isolation (`results/delay30_isolation`), state-budget (`results/delay30_maintenance`) | `test_coupling_rescue.py`, `test_curriculum_study.py` | interference quantified (0.735–1.000); H7b + state budget mitigate with measured residual cost | Schedule trade-offs documented (ε=0.05 near-freeze vs ε=0.2 oscillation, §8q) |
| Learning independent of memorisation | held-out ridge decode (CV), rule-conforming held-out cues, unseen-size probes, oracle baselines in every registered comparison | `test_state_content.py::TestRidgeDecode` | §7 + §8 series (every claimed effect has an oracle/shortcut-control) | Memorisation vs generalization separated *at cue/instance level*; concept-level abstraction not probed |

### C.4 Multi-step computation and reasoning readiness (Investigation 4)

| Claim | Source | Tests | Results | Limitations |
|---|---|---|---|---|
| Maintain intermediates across operations | delay trials in `environment/graph_environment.py` (markers visible only at step 0; generic blank thereafter; targets active only at the final step) | `test_graph_tasks.py::test_delay_masks_targets_then_labels`, `test_environment.py::TestDelay` | delay-30 maintained in 11/23 seeds behaviourally, 23/23 information (§8k); §8q: ε=0.2 budget → 7/12 with real adaptation, decode 1.000 ×24; ε=0.05 → 11/12 | Population heterogeneity is the open engineering problem, now instrumented (§8t/§8u/§8v) |
| Dependent computational steps | presentation→(30 blind steps)→response: output at step t requires computation done at t=0 | as above | subst follows 1.000 (behaviour depends on presentation state, not re-encoding) | This is computation-with-state, not symbolic dependency graphs |
| Represent relationships | `data/graph_tasks.py::make_connectivity_dataset` (adjacency / bounded path-length), rewired controls, `evaluation/readout_geometry.py` boundary statistics | `test_graph_tasks.py` (20: label correctness, edge-count matching, rewired controls, shortcut elimination) | §3 graph benchmark; §8u mechanism record | Relations limited to graph-structural predicates |
| Use previously acquired information in subsequent operations | stage-B regimes initialized from stage-A checkpoints (§8l family) | `test_coupling_rescue.py` | §8l: stage carry measured (delay30_h7b) | — |
| Modify behaviour from intermediate results | `scripts/run_state_budget.py::BudgetSearch.step` (per-step displacement line search); `scripts/run_boundary_guard.py::GuardedBudgetSearch` (per-candidate bvm guard, amend-3 predicted-test-cloud instrument) | `tests/` do not cover these scripts (they are covered by pure-control bit-identity in §8u and canary runs) | **§8v: V1 detection 0 violations/1,212 episodes; V3 E1 zero crossings 10/12** (E3 cost FAIL: GNN δ < 50% control in 9/10) | Guard is an engineered, registered control — not autonomous self-regulation |
| Distinction: computation vs reasoning | delay-classification is the demonstrated competence; **no compositional/reasoning battery exists** | — | TECHNICAL_REPORT §4 explicitly does not claim reasoning | Reasoning readiness = the state machinery could host intermediate *symbolic* variables, but no such variables or tasks exist |

### C.5 Continuity and recovery (Investigation 5)

| Claim | Source | Tests | Results | Limitations |
|---|---|---|---|---|
| Save/restore internal state | `utils/checkpoints.py::save_checkpoint/load_checkpoint` — model + **optimizer + scheduler** + step + config + seed + `env_info` (incl. git commit); verified contents by direct read at audit time | none directly (utility exercised by every experiment) | all §8b–§8v runs resume from `checkpoint.pt` / `captured_checkpoint.pt` | **RNG state is NOT saved** (no `torch.get_rng_state()` in payload) |
| Resume without restarting learning | `scripts/run_state_budget.py` loads captured checkpoint → 100+ further episodes; `scripts/verify_reproducibility.py` rebuilds model+dataset from embedded config and reproduces recorded metrics | `verify_reproducibility.py` (TOL = 1e-6) | **12/12 exact matches** (README status) | Episode-level, not intra-episode: a half-elapsed episode cannot be resumed exactly (env stream + RNG not serialized) |
| Behavioural equivalence after restore | same verifier + §8u pure-control bit-identity (un-instrumented ≡ instrumented run) | — | bit-identical control recorded | — |
| Recovery of objectives/intermediates | task objectives are in config (restored); neural intermediate results recoverable within trials via capture/reinstate | `test_state_content.py` | §8 series capture-and-hold (H7b) | No execution/progress journal ("what had I been doing") |

---

## D. Capability gaps (what is missing or insufficiently demonstrated)

1. **Persistent (cross-run / cross-episode) memory store — ABSENT.** The only retention is within-episode recurrent state (reset each episode) and episodic weights. No archive, no selective recall, no consolidation. *(Largest scientific gap; also the largest design question — must respect the frozen-architecture constraint or open a new registered amendment series.)*
2. **Goal / task-progress representation — ABSENT.** State encodes answer + task identity; nothing encodes "what am I trying to achieve / how far along am I". Verification would follow the §7 decode + substitution blueprint (decode progress variables, then substitute them causally).
3. **Reward-driven learning & action selection — ABSENT (partially implemented).** Environment emits rewards; `Trainer.train` ignores them; `policy_gradient_loss` exists but is unused; action = argmax(logits). No non-supervised learning pathway has ever been registered.
4. **True mid-execution resume — PARTIAL.** Checkpoints omit RNG and environment-stream state; resume is episode-aligned only. Behavioural-equivalence evidence therefore covers episode boundaries, not arbitrary interruption points.
5. **Reasoning tasks — ABSENT.** No multi-hop/compositional battery has been registered; the distinction between computation-with-state and reasoning is currently unfalsifiable because no reasoning probe exists.
6. **Interference not solved (measured).** §8h/§8i/§8n quantify collateral damage; mitigations trade adaptation for safety (freeze/rehearsal/balanced/budget/guard). The §8v guard prevents access loss but at <50% adaptation in 9/10 adapted checkpoints — safe maintenance currently means slowed learning.
7. **Instrument fidelity bottleneck (measured, bounded).** The §8v E4 measurement: the 16-instance replay-batch prediction error is 5.55 bvm units against a 0.5-unit threshold — the current guard is over-conservative. A per-instance / accepted-candidate test-cloud guard is the registered follow-up.
8. **Reasoning-relevant generalization depth** — one documented open anomaly (a single population seed with an unseen-size gap; encoder edge-density calibration, README open items).

---

## E. Minimum requirements for M1 (smallest contributions to a minimal unified cognitive loop)

Interpreting M1 as: *an ALS instance that a coordinator (like a SiR loop) can checkpoint, query, task, and trust to learn and maintain a state faithfully.* No new cognitive modules are designed here; each item is the smallest verified-construction or protocol change.

1. **Complete the checkpoint contract (continuity).** Add RNG (python/numpy/torch) and environment-step state to `save_checkpoint` — a `resume()` utility that re-enters mid-run deterministically. Validation: a registered interrupted-resume experiment with bit-exactness against uninterrupted control (the §8u pure-control pattern). *Smallest possible change; directly closes Gap 4; no architecture impact.*
2. **Expose a state/query/status API using what already exists.** Everything = `load_checkpoint` + `capture_trial_states` + `boundary_statistics` — wrap as a queryable interface: `inspect_state(checkpoint)`.
3. **Register a goal/progress-field probe before any architecture change.** Before claiming or denying goal-readiness: extend the state-content battery to (a) decode a defined progress variable (e.g., phase of the curriculum, delay-phase index) via ridge decode; (b) test causal use by substitution. If decoding fails, the gap stays ABSENT **with evidence**. This is a measurement, not a module.
4. **Activate an explicit alternative pathway for a registered experiment** (reward-weighted loss or policy-gradient arm) on the existing cue task where reversal/reward data already exist — scoped to one experiment section, keeping D800 frozen as reference.
5. **Define the integration contract for memory.** ALS does not need a new store to be useful for M1; it needs a declared boundary: what enters via `GraphData`/adapters, what serializes via `save_checkpoint`/`GraphData` payloads.

---

## F. Recommended development sequence

Ordered by (1) cognitive-loop importance, (2) complexity, (3) dependencies, (4) scientific uncertainty, (5) independent validatability:

| Step | Item | Why here | Effort | Risk |
|---|---|---|---|---|
| 1 | **Checkpoint completeness + deterministic resume (E.1, E.2)** | highest importance-to-effort ratio; zero science risk; required by any external coordinator | small (utils + registered resume experiment) | none (pure addition, D800 untouched) |
| 2 | **Query/status API (E.2)** — load_checkpoint + capture_trial_states + boundary_statistics wrapped as inspect/checkpoint fanout | enables all later items; already validated machinery | small | none |
| 3 | **Goal/progress probe registration (E.3)** | settles the biggest uncertain claim with the established §7 decode/substitution protocol before any design commitment | small-medium (new registered experiment section, no model change) | low (a measurement) |
| 4 | **Persistent-memory design study → registered experiments (Gap 1, 6)** | the decisive unknown items; must be pre-registered like §8b–§8v: candidate = an archival store with selective read into state, tested under the already-measured interference instruments (§8n collateral, §8v guard) | large (pre-registration + multi-arm study on the established 12-seed population) | high science risk; mitigated by existing instrumentation |
| 5 | **Reward pathway (E.4)** | needed for any action-selection role; existing env/reversal data make it cheap | small-medium | medium (behavioral regime shift; needs interference re-measure) |
| 6 | **Reasoning battery (Gap 5)** | only after the state/goal/memory picture is settled; define compositional probes with oracle baselines per the house standard | medium | high (unknown outcomes — register everything) |

Rationale for the order: steps 1–2 are enabling and riskless resources; step 3 converts the most important uncertainty into measured evidence cheaply; steps 4–6 are genuine scientific workload and should each be pre-registered with acceptance criteria, which is the established house method.

---

## Summary — the short version

**Three most important capabilities ALS already demonstrates:**
1. **Causally verified working memory in a fixed-size internal state** — computed answers maintained through 30+ blind steps across a population; behaviour follows substituted state content (23/23 information survival; subst 1.000) with a captured/queried/modified/restored state lifecycle.
2. **Measured and safe adaptation** — episodic supervised learning persists across episodes; the interference costs are precisely quantified (collateral damage 0.735–1.000), and maintenance is now *guardable at run time*: the §8q state-budget and §8v bvm-guard mechanisms prevent functional access loss in 10/12 checkpoints (incl. a previously-unrescuable case), at a measured adaptation cost.
3. **Audit-grade reproducibility discipline** — ~28 pre-registered experiments with amendments/controls, 12/12 checkpoint-reproduction at 1e-6 from config alone, bit-identical instrumented-vs-pure controls, and a 150-case test suite — exactly the trust property a unified system needs from a subsystem.

**Three most important gaps preventing fuller contribution to SiR:**
1. **No persistent memory** — nothing is retained across runs beyond episodic weights; the only store is in-episode state (reset per episode) — would be the backbone of any cross-task continuity role.
2. **No goal/progress representation or reward-driven action selection** — the state encodes answers and identities, not intentions or progress; the environment's reward signal is never used to learn — which also keeps ALS out of the action-selection role.
3. **Incomplete execution-continuity contract and evidenced-but-open reasoning readiness** — checkpoints lack RNG/env-stream state, so arbitrary mid-execution resume is not guaranteed; reasoning/compositional tasks have never been registered, so "can host reasoning" remains unverifiable as written.

---

*Compliance note: WP1 restrictions were observed — no ALS architecture, experiment, test, or result files were modified during this audit; the sole artefact produced is this document. All classifications above cite on-disk results and the registered TECHNICAL_REPORT.md record; no claim rests on code appearance alone (Investigation-question restriction 6).*
