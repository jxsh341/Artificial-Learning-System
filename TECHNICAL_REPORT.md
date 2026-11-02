# Technical Report — GNN + SSM Artificial Learning System

**Status:** Working experimental platform, Milestones 1–5 complete (spec §27), graph-dependent
benchmark complete, curriculum-scheduling study complete (best condition 2/3 seeds; reliability
is the open problem).
**Test suite:** 117 tests passing. **Results:** all numbers below are read directly from
`results/<experiment>/summary.json` and per-seed CSVs (regenerable from
`configs/` via the documented commands).

---

## 1. System summary

Pure-PyTorch implementation (no PyG dependency), spec §4 structure:

- **Data**: stable `GraphData` interface with validation + disjoint-union batching;
  5 synthetic graph families; adapter registry (integration seam); connectivity/adjacency
  task generator with shortcut controls (`data/graph_tasks.py`).
- **Models**: message-passing GNN (mean/sum/max aggregation, residual, LayerNorm, node-mask
  silencing), readouts (mean/max/attention/selected pooling), linear state-space SSM with
  spectral-radius-bounded `A` (`initial_state`/`step`/`forward`/`reset_state`), composable
  `GraphTemporalModel` (`forward(sequence)` + online `step(graph, state)`), ablation builders M1–M6.
- **Environment**: configurable cue-learning environment (delays, stochastic rewards, reversal)
  and a graph-classification environment (immediate/delayed variants, three blank modes).
- **Training**: episodic BPTT trainer; staged curriculum support.
- **Evaluation**: ARRG protocols, memory curves, temporal perturbations + storage-locus,
  graph perturbations (incl. degree-preserving swaps), M7 controls, counterfactual probe,
  multi-seed statistics (mean/std/median/CI95).
- **Experiments**: reproducible multi-seed runner, ablation matrix, resumable seed merging.

---

## 2. Cue-learning phase (spec §14–§19), results verified

Base config: small-world graph, 100 nodes, GNN 2×64, diagonal SSM 64, delay=2, 2 seeds
(plus 3-seed scale runs). All values from `results/base/summary.json` etc.

| Experiment | Result (mean) |
|---|---|
| Acquisition | 0.67 → 1.00, criterion at ~trial 91 |
| Retention ratio @ delay 500 | 1.00 |
| Generalization (held-out cues) | 1.00 |
| Reversal | relearns to 1.00; ttc ≈ 281; residual preference at onset = 1.00 |

**Scaling (everything else fixed, 3 seeds each):** acquisition 0.63→1.00 at 100 nodes,
0.62→1.00 at 500, 0.63→1.00 at 1000; criterion ~98/99/100; retention@500 = 1.00 everywhere;
reversal ttc 294/355/356. Learning speed essentially flat across 10× structural growth.

**Memory curve** (`deep_analysis`, 2 seeds; accuracy vs temporal distance with frozen weights):
ceiling 1.00 at delays 0–2; chance (0.43–0.58) from delay 10 through 5000; contiguous
half-life ≈ 2 steps. Dissociation: **the cue trace in the SSM state is short-lived; durable
memory of the mapping lives in the weights** (which is why ARRG retention @ 500 = 1.0 —
the cue is re-presented at test).

**Temporal attacks** (same run): noise σ ≤ 1.0 per step: no degradation (0.997–1.0); partial
state resets up to 75% of dimensions: no degradation; full state reset: 0.500 (chance);
truncation every 2/10 steps: intact; every 5 steps: 0.901 (timing interaction, unexplained).

**Storage locus** (state × graph × response regime, 2 seeds):

| condition | trained delay (2) | immediate (0) |
|---|---:|---:|
| full system | 1.000 | 1.000 |
| state reset each step | **0.515** | 1.000 |
| graph features shuffled | 1.000 | 1.000 |
| weights only | **0.486** | 1.000 |

Interpretation (cue task): the learned mapping resides in the weights (+cue-in-input);
cue-bridging across the delay requires the SSM state; graph structure is not measurably used.

**Graph attacks** (15 conditions: edge removal/rewiring/weight corruption/feature shuffling/
node silencing/community silencing, up to 75% levels): **max behavioural delta = 0.000**.
The cue task never exercises relational computation — an important negative result that
motivated the graph-dependent benchmark.

---

## 3. Graph-dependent benchmark (the decisive phase)

### 3.1 Task design (`data/graph_tasks.py`, `label_mode`)

- **adjacency**: both classes drawn from the *same connected-graph family*; label = whether
  the two marked nodes are adjacent (positives: a random edge) vs distance ≥ `neg_min_distance`
  (negatives, same component). No graph-level statistic (component count, degree, density)
  can reveal the label — verified by `test_adjacency_mode_label_depends_on_marked_pair_only`.
- **path**: positives distance ≤ max_path_len, negatives distance > max_path_len, same family.
- Markers A/B are the only class-asymmetric feature; node features are iid per instance.

### 3.2 Failures found and fixed (each verified by results)

1. **Degenerate first task**: in the "path" task, label ⟺ global component count (positives
   were connected graphs, negatives 2-component graphs). Detection: on
   `graph_rel_delayed_sum`, `delayed_reset_cleared` equalled full performance (0.76/0.81) —
   the model answered correctly with **no state and no markers**, so the marker channel was
   decorative. The locus probe caught the design flaw. Fixed by the adjacency redesign.
2. **Environment bug**: `GraphClassificationEnvironment.step()` returned the full instance
   (markers visible) at delay/response steps, contradicting its docstring. The "delayed"
   training was never actually delayed. Smoking gun: **M2 (GNN, no SSM — no state at all)
   reached train-acc 1.000** on the delayed task
   (`graph_adjacency_delayed_bare`: train 1.000, locus `immediate` 1.0 / `delayed_retain` 0.48).
   Fixed; regression test `test_step_observation_blanked_during_delay` added.
3. **Mean-aggregation oversquashing**: on the (degenerate) path task, mean aggregation failed
   (task acc 0.60/0.56, class-biased) while **sum aggregation learned** (0.81/0.77, both
   classes 0.75–0.85, unseen-size 0.77/0.80). Longer training (200 vs 80 episodes) changed
   nothing → converged, not data-limited.
4. **Memorization beats the rule** (immediate adjacency, rich 16-dim instance features):

   | intervention | train acc | test acc |
   |---|---:|---:|
   | 400 instances | 0.91 / 0.96 | 0.57 / 0.46 |
   | 2000 instances | 0.59 / 0.59 | 0.58 / 0.52 |
   | + weight decay 0.01 | 0.73 / 0.64 | 0.52 / 0.53 |
   | 400 instances, 600 episodes | **0.999** | **0.51 / 0.45** |

   Data scaling suppressed memorization (train ≈ test) but did not produce the rule; more
   training only perfected memorization. **Fix: strip identity features** (`node_feature_dim:
   2` — bare marker channel), making topological memorization infeasible.

### 3.3 Positive result — immediate relational task (bare features, sum aggregation, 2 seeds)

`graph_adjacency_bare`: **test accuracy 1.000/1.000** (neg and pos both 1.000), converged by
~episode 150 of 600; **unseen size (n=200): 1.000/1.000**.

M7 controls (`results/graph_adjacency_bare/summary.json`):

| condition | accuracy |
|---|---:|
| intact | 1.000 |
| **degree-preserving swaps + recomputed labels** (all levels) | **1.000** |
| randomized rewiring + recomputed labels | 0.918 / 0.860 / 0.970 |
| swaps + original labels | 0.882 → 0.798 → 0.675 (→ chance with level) |
| rewiring + original labels | 0.887 → 0.748 → 0.522 |

Interpretation: predictions follow the *presented* connectivity (original-labels probes
degrade toward chance as structure is destroyed), are **not** explained by degree statistics
(degree-preserving relabeled = 1.000), and are not instance memorization (unseen instances
and sizes = 1.000). Message passing is necessary: M1/M6 (no message passing) and M3 are at
chance on every variant tested (`graph_classification` ablations: 0.50–0.60).

### 3.4 Positive result — delayed relational task (the SSM's distinct role)

Frozen configuration: adjacency task, bare markers, sum aggregation, 2-layer GNN,
**state_dim 8** (continuous 8-dim bottleneck — linear decoding cannot memorize thousands of
instances), **generic blank observations** (one fixed instance-independent graph at delay and
response steps), 2000 instances. With this configuration, memorization pathways are closed by
construction (`graph_adj_delayed_generic`: train-acc flat at 0.50 — nothing spurious to learn).

**Curriculum (immediate-first, then delayed; `graph_adj_delayed_curr`, 4 seeds):**

| seed | phase 1 (immediate) | delayed test | unseen n=200 | `delayed_retain` | `delayed_reset_cleared` |
|---|---:|---:|---:|---:|---:|
| 0 | 1.000 | 0.52 | 0.505 | 0.52 | 0.52 |
| 1 | 1.000 | **1.000** | 0.505* | **1.000** | **0.480** |
| 2 | 1.000 | **1.000** | **1.000** | **1.000** | **0.480** |
| 3 | 1.000 | **1.000** | **1.000** | **1.000** | **0.480** |

\* seed 1's unseen-size probe returned 0.505 — since traced to the legacy probe's blank
protocol mismatch (delay steps without the trained generic blank); corrected reading is
**1.000** (`scripts/recompute_unseen_size.py`, §6). The 0.505/0.52 legacy values for seed 0
are chance readings **of a failed seed under the mismatched protocol** and remain
historical.

M2 (GNN, no SSM) on the delayed task: 0.48–0.52 across all seeds — structurally cannot bridge.
Without the curriculum (joint discovery, `graph_adj_delayed_generic`), zero progress in 300
episodes (train-acc flat 0.50, both seeds).

**The full dissociation on successful seeds**: the SSM state carries the computed relational
answer across the marker-free delay (`delayed_retain` 1.000); resetting the state at the
response destroys it (`delayed_reset_cleared` 0.480 ≈ chance); recomputation from persistent
input works (`delayed_markers_kept_reset` 1.000). **Success rate: 3/4 seeds** — the
reliability question is open and is the next experiment.

---

## 4. Consolidated memory table (what lives where — all measured)

| Information | Storage | Evidence |
|---|---|---|
| Learned cue→action / adjacency→response mappings | Weights | durable across delay 500+; survives state reset (immediate regime 1.000) |
| Currently-present cue bridging a gap | SSM state | memory curve half-life ≈ 2 steps; full state reset → chance (0.500–0.515) |
| Computed relational answer (working memory) | SSM state | successful seeds: `delayed_retain` 1.000, `delayed_reset_cleared` 0.480 |
| Relational structure itself | GNN message passing | M1/M6 chance; degree-preserving relabeled control 1.000; bare features required |

**Scope caveats (honest):** (a) the state-reset result establishes *necessity of the state*
for the observed behaviour, not that the state implements a general working-memory system;
(b) the working-memory result holds for 3/4 seeds under one curriculum and one task family;
(c) the cue-task graph attacks showed the *original* task was graph-independent — the
graph-dependent claims rest on the adjacency benchmark only; (d) all results are CPU-scale
(≤ 1000 nodes, ≤ 10k instances) and 2–4 seeds; (e) the truncate@5 dip (0.901) is unexplained.

---

## 5. Curriculum-scheduling study (Phase B–D) — run and complete

`results/curriculum_study/` — 4 budget-matched conditions × 3 seeds × 400 episodes
(optimizer steps matched; presentation counts differ by construction and are recorded).
Frozen baseline: adjacency task, bare markers, sum aggregation, 2-layer GNN, `state_dim` 8,
generic blanks, 2000 instances. Success = delayed test accuracy ≥ 0.9.

| condition | success rate | delayed accs | immediate accs |
|---|---:|---|---|
| A immediate-only | 0/3 | 0.48–0.52 (chance) | 1.00 (sanity check) |
| B delayed-only | **0/3** | 0.48–0.52 | 0.49–0.52 (rule never learned) |
| C staged (200 imm + 200 del) | 1/3 | 0.52 / 0.48 / **1.000** | 0.52 / 1.00 / 1.00 |
| D interleaved | **2/3** | 0.48 / **1.000** / **1.000** | 1.00 / 1.00 / 1.00 |

(Reference: the earlier staged run with a *double* phase-2 budget — 200 + 400 episodes —
succeeded 3/4 seeds, `graph_adj_delayed_curr`; bridge success depends on delayed-phase budget.)

**Failure-mode taxonomy** (from the intact-vs-reset trajectories recorded every 100 episodes —
Phase C):

1. **Bridge never emerges** (seed 0 under every schedule): immediate rule learned (imm 1.00
   throughout), delayed accuracy flat at chance — initialization sensitivity.
2. **Catastrophic forgetting of the rule** (C seed 0): immediate 1.00 after phase 1 → 0.52
   after 200 delay-only episodes — the rule must be rehearsed during the delayed phase.
3. **Bridge instability**: C seed 1 reached delayed 1.00 at episode 300 with the rule intact,
   then *lost* the bridge by episode 400; D seed 1 emerged (0.95 @ ep200), lost (ep300),
   and *recovered* (1.00 @ ep400). Interleaving appears to both protect the rule and give
   the bridge repeated opportunities to re-emerge.

**Joint discovery is impossible in this setup**: B_delayed (400 episodes) never learned
either capability (both flat at chance in every probe).

### Counterfactual probe (Phase D) — all successful seeds, all conditions = 1.000

| condition | meaning | result (3 successful seeds) |
|---|---|---:|
| `no_change` | same instance re-presented → retain | 1.000 |
| `update` / `update_flipped` | new instance, answer must change | **1.000** |
| `irrelevant` | blank revision cue → must NOT overwrite | 1.000 |

The retained state is not a rigid imprint: the system updates its answer on genuinely new
evidence (including answer-flipping evidence) and ignores non-informative input.

### Computed-answer retention (successful seeds, per-seed)

- C seed 2: 1.000 at delays 0–30, chance at 100
- D seed 2: 1.000 at delays 0–30, chance at 100
- D seed 1: 1.000 at delays 0–3, chance at 10

Working memory of the computed answer spans **3–30 steps — 1–10× the trained delay (3)** —
seed-dependent, without ever being trained at those longer delays.

### Interpretation

The curriculum effect is real but its mechanism is now precise: (i) joint discovery of rule
+ bridge fails outright; (ii) the rule must be learned first *and rehearsed* during bridge
training (interleaving dominates staged-with-equal-budget under this budget); (iii) the
bridge emerges within 50–200 delayed episodes once the rule exists, but is initially
unstable; (iv) one seed (0) never discovered the bridge under any schedule — an
initialization/optimization sensitivity that remains the main open reliability problem.

Pre-registered criterion (observed success proportion ≥ 90%) is **not yet met** by any
condition of the base study (best: 2/3). Note: observed proportions are weak evidence about
the population success rate — the reliability interventions in §6 address this with
additional seeds and a tighter criterion interpretation.

---

## 6. Bridge-reliability experiments (single-variable interventions) — run and complete

Pre-launch reproducibility milestone: `scripts/verify_reproducibility.py` reloads every
saved checkpoint + stored config, rebuilds the dataset and model from config alone, and
re-runs the probes — **12/12 condition-seeds reproduce the published metrics exactly**
(tolerance 1e-6), and re-aggregated per-seed values match the committed `summary.json`.

Two interventions, tested separately (never combined), 3 seeds each:

### Intervention 1 — D800: interleaved training extended to 800 episodes

`results/d800_interleaved/`: **3/3 seeds (100% observed)** — the pre-registered *observed*
criterion (≥ 90% success proportion in the sample) is met. Statistical caveat: 3/3 does not
establish that the population success probability exceeds 90% (if p = 0.9, P(3/3) = 0.73;
if p = 0.7, P(3/3) = 0.34 — weak evidence against lower rates). Additional seeds are
required for a tighter bound; see §7.

| seed | trajectory (delayed intact/reset) | outcome |
|---|---|---|
| 0 | 0.48 → 0.48 → 0.48 → **0.90** @400 → 1.00 @500–800 | bridge emerged *after* the original budget — failure mode 1 was partly under-training |
| 1 | 0.48 → 0.48 → 0.52 → 0.52 → **1.00** @500–800 | slower emergence |
| 2 | **1.00** from episode 100 | early emergence, stable |

State-reset dissociation holds in all seeds (intact 1.000, reset 0.48–0.52); immediate rule
preserved everywhere (1.000); legacy unseen-size readings were 0.505 on seeds 0–1 — since
traced to the probe blank-protocol defect and corrected to 1.000 (see below).

### Intervention 2 — C_frozen: GNN frozen during the delayed phase (400-episode budget)

`results/c_frozen_gnn/`: **3/3 seeds (100%)** — freezing the relational representation
during bridge training rescued both seeds that failed under unfrozen staged training.

| seed | bridge | immediate rule after frozen phase |
|---|---|---|
| 0 | emerged at ep300, stable | **0.52 — destroyed** despite the frozen GNN |
| 1 | emerged at ep300, stable | **0.52 — destroyed** |
| 2 | emerged at ep300, stable | 1.000 |

**Key mechanistic finding**: freezing the GNN prevents representation forgetting but *not*
readout forgetting — the immediate rule is read through the head/SSM pathway, and phase-2
training reorganizes it for state-reading. Rule forgetting was therefore never (only) a GNN
phenomenon. The trade-off is explicit: C_frozen buys 3/3 bridge success at the cost of the
immediate skill (2/3 seeds); **D800 dominates** — 3/3 on both capabilities with no rule loss.

### Updated reliability table

| condition | budget | success | immediate preserved |
|---|---|---:|---:|
| B delayed-only | 400 | 0/3 | n/a (never learned) |
| C staged | 400 | 1/3 | 1/3 |
| C staged | 600 (200+400) | 3/4 | 3/4 |
| D interleaved | 400 | 2/3 | 3/3 |
| **D interleaved** | **800** | **3/3** | **3/3** |
| **C staged, GNN frozen in phase 2** | 400 | **3/3** | 1/3 |

**Resolved defect, not a model property**: the unseen-size (n=200) probe in
`run_graph_generalization` fed delay steps in the *markers-cleared-instance* blank format,
while training used the **generic blank graph** — a protocol mismatch that read as "chance"
in successful runs (`scripts/recompute_unseen_size.py`; JSON at
`results/curriculum_study/unseen_size_corrected.json`). Under the training-consistent
protocol all successful seeds score at ceiling:

| run | legacy probe | corrected (generic blank) |
|---|---:|---:|
| C_staged seed 2 | 1.000 | 1.000 |
| D_interleaved seed 1 | 0.505 | **1.000** |
| D_interleaved seed 2 | 1.000 | 1.000 |
| D800 seeds 0–2 | 0.505 / 0.505 / 1.000 | **1.000 / 1.000 / 1.000** |
| C_frozen seeds 0–2 | 0.895 / 1.000 / 0.495 | **1.000 / 1.000 / 0.995** |

The matching defect is fixed at the source (the probe now threads `generic_blank` through,
as training does), with a regression test.

### Controlled size sweep (`scripts/run_size_sweep.py`, no retraining)

Sizes 100→130→160→200 with node count isolated from mean degree (density fixed vs
density ∝ 100/N), 2 independent instance sets × 200 instances per (size, mode), evaluation
under the corrected protocol. **No degradation anywhere**:

| | immediate | delayed intact | delayed reset |
|---|---|---|---|
| D800, all sizes/modes | 1.000 | 1.000 (min 0.995) | ≈ 0.50 |
| C_frozen, all sizes/modes | ≈ 0.65–0.69 (flat) | 0.995–1.000 | ≈ 0.50 |

D800 generalizes to unseen graph sizes perfectly under both density regimes; activation
statistics (pooled-graph ‖z‖ 13.9→11.9 only at the degree-doubled n=200 fixed-density
setting, attention marker share 0.94→0.88) show mild degree dependence of the sum-aggregated
representation that the attention readout absorbs — behaviour is flat in the tested range.
Scope: this establishes that **no behavioural failure was observed** at sizes 100–200 under
either density regime in the tested checkpoints; it does not exclude scale-sensitive
internal representations outside this range (activation statistics still vary with graph
size). C_frozen's immediate impairment is size-independent (readout forgetting, not scale
sensitivity). Size generalization is safely claimable for successful runs within the tested
range.

### Interpretation

The bridge is reliably learnable given either (a) sufficient interleaved budget — emergence
times ranged from episode 100 to 500 across seeds — or (b) a frozen relational
representation. The three original failure modes reduce to: under-budget emergence (fixed by
D800), readout forgetting (caused C's losses; *not* fixed but made irrelevant by D800's
rehearsal), and genuine bridge-instability (absent in both successful interventions' final
phases).

---

## 7. State-content analysis — run and complete

`scripts/run_state_analysis.py` (`results/*/state_content/`), on saved checkpoints of
D800, C_frozen, and the base study — 18 condition-seeds, all held-out test instances,
linear (ridge, 5-fold CV) probes. Answering: **what does the 8-dim state carry, and is
that content the cause of the prediction?**

### A. Answer decodability (held-out linear probe, vs majority baseline = 0.55)

| seeds | pre-info | post-presentation | mid/final delay | response point |
|---|---:|---:|---:|---:|
| all successful (6) | 0.45 (= baseline) | **1.000** | **1.000** | **1.000** |
| A_immediate ×3 (rule learned, no bridge training) | 0.45 | 1.000 | 0.45–1.00 (seed-dependent decay) | 0.45–1.00 |
| D seed 0, C_frozen-style failure C seed 1 | 0.45 | 1.000 | **1.000** | **1.000** |
| B_delayed, C_staged seed 0 (rule never learned) | 0.45 | 0.45 (= baseline) | 0.45 | 0.45 |

### B. Answer vs feature retention (`avf_*`)

The state separates perfectly by answer class (cosine separation −2.0 for successful
seeds) but carries **no detectable instance features**: same-answer states are
indistinguishable from each other (own-class-centroid distance ≈ 1e-4 vs cross-class
0.38), and nearest-neighbour same-label rate (1.000) is explained entirely by class
clustering (chance 0.498). Direct inspection (200 instances/seed): the successful code
is a **sparse signed pattern** — exactly 3 of 8 dims flip sign between classes (means
±0.50, std 0), the other 5 dims saturate identically. The state stores the computed
answer, not the evidence; decomposing features from it is not observed.

### C. Successful vs failed — the signature is not decodability, it is *use*

The most informative result: **two failed seeds (D seed 0, C_staged seed 1) have a
perfectly linearly decodable answer in their state at every capture point, yet produce
chance behaviour and — decisively — fail every substitution control.** Comparison table
(state-level):

| group | decode (all points) | subst same-answer | subst diff-answer follows donated | subst zero |
|---|---:|---:|---:|---:|
| D800, C_frozen, base successes (9) | 1.000 | **1.000** | **1.000** | chance |
| bridge-failed, answer retained (2: D seed 0, C seed 1) | 1.000 | 0.48–0.52 | 0.48–0.52 (chance) | chance |
| immediate-only (A seeds) | post 1.000, delay 0.45–1.00 | chance | chance | chance |
| rule never learned (B, C seed 0) | chance | chance | chance | chance |

Also: failed retained-answer states saturate toward ±1 in *all* 8 dims with a residual
answer margin of < 0.03 (own vs other-class centroid distances 0.0008 vs 0.0212), while
successful states keep a large signed margin (0.0001 vs 0.38). "Answer present but
saturated" reads as a success signature: the route the head reads must include the state
answer code; decodability alone does not imply use — matching the caution that a linear
probe establishes accessibility, not symbolic storage or causal use.

`drift_*` metrics: successful models move the state little under irrelevant blank input
(blank-step drift 0.42–0.96) and far more at presentation (revision gain 2.5–6.0).

### D. Causal state substitution (`subst_*`)

After instance i's presentation, the carried state is replaced (donors from independent
trials at the *same* computation point — in-distribution, norm ratio ≈ 1.0) and the
delay finishes on generic blank observations, so the substituted state is the **only**
instance information:

- **same-answer substituted** (different instance, same label): prediction correct at
  **1.000** in all 9 successful seeds — the identity of the presenting evidence is not
  read; only the answer is.
- **different-answer substituted**: prediction follows the **donated** answer at
  **1.000** — direct causal proof that the (donated) state content controls the
  prediction, not the presentation history (which pointed the other way).
- **zero state**: chance (0.48–0.52) in every seed; **shuffled own states**: ≈ chance.
- ALL failed seeds: every substitution condition ≈ chance-equivalent — their state
  content, however decodable, does not reach the prediction.

### Interpretation

The delayed-answer mechanism in this model class is: presentation computes a **sparse
signed answer code in 3 state dims**; blank inputs leave it intact (small drift); the
response head reads that code. Bridge failures decompose into exactly the three internal
phenotypes (i) the answer never computed (B, C seed 0 — no immediate rule), (ii) the code
decays during delay (A seeds), and (iii) the code present-but-saturated and causally
disconnected from the readout (D seed 0, C_staged seed 1) — the third phenotype is a new,
sharper target than "bridge never emerged".

---

## 8. Next steps (in order)

1. ~~Bridge reliability~~ — **achieved in observed samples**: D800 (3/3, both capabilities
   preserved) and C_frozen (3/3, immediate rule lost in 2/3) meet the pre-registered
   *observed* criterion (success proportion ≥ 90% in the sample). Statistical caveat:
   3/3 does not bound the population rate tightly (P(3/3 | p=0.7) = 0.34); item 4
   adds seeds to tighten the estimate. D800 (interleaved, 800 episodes) is the
   recommended schedule.
2. ~~Scale-generalization anomaly~~ — **resolved as an evaluation defect**: the legacy
   unseen-size probe used the wrong blank protocol during the delay (markers-cleared
   instance instead of the trained generic blank). Corrected, every successful seed scores
   0.995–1.000 at n=200, and the controlled size sweep (100→200, degree-matched and
   density-fixed) is flat at 1.000 for D800. `run_graph_generalization` now threads the
   generic blank through (regression-tested); recompute with `scripts/recompute_unseen_size.py`.
3. ~~State-content analysis~~ — **run and complete** (§7): the state carries a sparse
   signed 3-dim answer code, causally driving the prediction in successful seeds
   (substitution evidence); failed seeds split into answer-not-computed, code-decay,
   and code-present-but-unused phenotypes. Follow-ups: targeted interventions on the
   phenotype-(iii) seeds (e.g., head/state coupling during bridge training).
4. **Statistical power**: add D800 seeds — 3/3 meets the pre-registered *observed*
   proportion, but does not tightly bound the population success rate.
5. **Delay scaling**: successful seeds already generalize to 10-30 steps untrained; test
   curricula with increasing delays to push the working-memory horizon deliberately.
6. Only after reliability holds at scale: richer relations (paths, multi-hop), multiple
   retained items, then the Level-3 agenda (evidence integration, rule revision).

### Following the H5 result (§8b): next mechanistic steps

1. **Protect the immediate rule through coupling + joint training**: Treatment A on
   C_staged s1 met 5/6 criteria; the failure was immediate erosion in stage 4. Interleave
   a low-rate immediate rehearsal during/after coupling, or freeze the immediate readout
   route while the delayed route trains — pre-register the same rubric.
2. **Why did the D-int s0 readout path fail to converge** (with identical pre-state
   geometry)? Longer/reshaped readout-only training; inspect whether the head's input
   route (C·s vs the D(z) direct path from the generic blank) is degenerate for that
   seed.
3. **Reliability track (unchanged)**: additional D800 seeds for the population bound;
   D800 reference configuration stays frozen.

---

## 8b. H5 — state-to-readout coupling hypothesis (pre-registered design)

**Formal statement (user-specified):** *A major failure mode in delayed-task acquisition
occurs when the SSM develops an answer-bearing internal state without establishing an
effective functional connection between that state and the output readout. Selectively
improving this coupling during bridge training should rescue delayed-task performance
without requiring the underlying state representation to be relearned.*

Claim language for the motivating finding (per §7): in certain failed runs the correct
answer remained perfectly decodable from the persistent state, yet state substitution
failed to influence the output — a dissociation between information *accessibility* and
*functional use* in the tested architecture. FAILED seeds are not yet said to have a
"readout defect" (H5 is the hypothesis tested here); the state is not yet called an
independent memory system.

**Starting points (phenotype iii):** base-study checkpoints
`curriculum/D_interleaved/seed_000` and `curriculum/C_staged/seed_001` (delayed 0.48/0.48,
immediate rule intact, answer decode = 1.000 at all points, substitution ≈ chance).
Reference configuration: D800 (`configs/experiments/d800_interleaved.yaml`), unchanged —
no new modules, no state dimensionality change.

**Stages per arm (400 total episodes each):**

| Arm | Stage 3 (coupling, 200 ep) | Stage 4 (restore, 200 ep) |
|---|---|---|
| `Control` | nothing frozen — normal interleaved (persistence check) | normal interleaved |
| `Treatment A` | GNN + pooling + SSM state-generation (A_raw, B) frozen; SSM output path (C, D) + head train on **correctly labelled** delayed trials | fully unfrozen, normal interleaved |
| `Treatment B` | identical to A, but state–target association broken (labels shuffled, same instances/inputs) | fully unfrozen, normal interleaved |

**Pre-registered success criteria (all required):**

| Metric | Required outcome |
|---|---|
| immediate accuracy | ≥ 0.90 |
| delayed accuracy (intact) | ≥ 0.90 |
| substitution, correct-state (same answer) | ≥ 0.90 |
| substitution, different answer | prediction follows donated answer |
| substitution, zero state | returns to chance |
| substitution, shuffled state | returns to chance |
| state representation | decode stays 1.000; answer margin/state geometry consistent with the starting sparse-signed phenotype |

Substitution conditions are decisive: improved delayed accuracy without causal dependence
on the donated state has not demonstrated the mechanism. Measured before and after the
intervention: decode accuracies, substitution battery, answer-class margin (own vs
other-class centroid distance), state norms, logit margins.

**Pre-registered outcome taxonomy:** (A) successful rescue → supports H5; (B) decodability
persists, rescue fails → failure involves more than insufficient readout training →
investigate state geometry and readout functional capacity; (C) delayed improves but
immediate deteriorates → coupling competes with rule retention even with GNN/SSM-input
frozen.

### Results (`results/coupling_rescue/`, `scripts/run_coupling_rescue.py`)

End-of-intervention battery (200 test instances; stage-3 boundary additionally probed
with the quick immediate/delayed probes, `training_probes.csv`):

| run | imm | delayed | subst same | subst diff→follows | zero | shuffled | decode (pre/post) |
|---|---:|---:|---:|---:|---:|---:|---|
| D-int s0 **control** | 0.520 | 0.480 | 0.48 | 0.52 | chance | chance | 1.0 / 0.45 |
| D-int s0 **treat A** | 0.520 | **0.620** | **0.695** | 0.635 | chance | chance | 1.0 / 1.0 |
| D-int s0 **treat B** | 0.525 | 0.480 | 0.48 | 0.52 | chance | chance | 1.0 / 0.45 |
| C-st s1 **control** | 0.520 | 0.480 | 0.48 | 0.52 | chance | chance | 1.0 / 1.0 |
| C-st s1 **treat A** | 0.520 | **1.000** | **1.000** | **1.000** | chance | 0.59 | 1.0 / 1.0 |
| C-st s1 **treat B** | 0.570 | 0.480 | 0.48 | 0.52 | chance | chance | 1.0 / 0.46 |

Stage-3 boundary (`training_probes.csv`): C-st s1 treat A reached **immediate 0.94,
delayed 1.00** after readout-only training alone — both core criteria met at that point;
D-int s0 treat A reached 1.00 / 0.52 (no delayed rescue). Substitution battery was then
evaluated at the end (§8b stage 5).

### Verdict per arm

- **Treatment B (negative control) is clean**: identical training on scrambled
  associations changed nothing (delayed 0.48, substitution chance) on both sources. The
  treatment-A effects are therefore coupling-specific, not extra-training artifacts.
- **Control arms**: phenotype persists in both sources (delayed 0.48, substitution dead);
  vanilla continued training additionally eroded the immediate rule (0.84→0.52 on D s0).
- **Treatment A on C_staged s1 — pre-registered outcome C, with H5's mechanism confirmed
  at stage 3**: readout-only training on correctly labelled state-target pairs *directly*
  established the missing functional connection (delayed 0.48→1.000; correct-state
  substitution 1.000; different-answer donations followed at 1.000; zero/shuffled chance;
  decode preserved; state norms within 3% of pre — the representation was NOT relearned).
  After stage-4 joint training the delayed skill **retained** its causal reading, but
  the immediate rule dropped 0.94→0.52 — the broader interference problem reproduces in
  the most targeted setting only after coupling is established. 5 of 6 rubric criteria
  met; the failure is the immediate-accuracy criterion only.
- **Treatment A on D_interleaved s0 — partially informative (outcome B flavour)**:
  delayed 0.62 and substitution 0.695/0.635 — directional improvements over control but
  far below criteria. Decisive constraint: pre-intervention state geometry is
  **indistinguishable** between the two sources (identical saturated code, margins,
  drift: own-class-centroid 0.0008 vs 0.0212 for D s0; revision-blank drift 0.82 vs
  0.48; decode 1.0 both) — yet one readout path converged and the other did not. The
  differentiator lies outside the measured linear state geometry: in the readout path
  parameters/optimisation. Logit margins post-intervention: C s1 A = 2.17 (confident,
  coupled) vs D s0 A = 0.11 (weak).

### Interpretation (bounded)

In the tested phenotype-(iii) checkpoints, the existing readout path **can** learn to use
already-encoded answer information (one of two sources), without any change to the state
representation — supporting H5's rescuability claim for one source; its failure on the
other source localises the residual cause outside the linear state geometry, and the
immediate-rule degradation under joint training reproduces outcome C. The targeted
language stands: *information accessibility and functional use of the persistent state
are dissociable, and functional use is improvable by selective readout training in at
least one of the tested failing runs.* No claim is made that the readout "defect" is
exhaustively characterised; the D-int s0 readout path's convergence failure is an open
mechanistic question (candidates: deeper readout-path dynamics through C/D interplay
with the blank-graph shortcut, or an optimisation basin requiring longer/other-shaped
readout training).

---

## 8c. H6 — capability retention under coupling + restoration (pre-registered design)

**Question:** can ALS acquire persistent-state utilisation while preserving an existing
capability through subsequent joint training? Acquisition and retention under joint
optimisation are now treated as separate challenges (the C s1 rescue held 0.94 immediate
*until* joint training began).

**Starting points:** the same two phenotype-(iii) checkpoints (H5 starting state, NOT the
rescued weights); C_staged s1 is the primary source (H5-positive), D-int s0 retained as
the difficult source. Reference configuration unchanged; no new modules, no added state
dimensions.

**Arms (400 episodes: 200 coupling + 200 restore, as in H5):**

| Arm | Stage 3 (coupling, 200 ep) | Stage 4 (restore, 200 ep) |
|---|---|---|
| `h6_control` | readout-only delayed (identical to H5 treatment A — interference recurrence expected) | unfrozen, normal interleaved |
| `h6_rehearsal` | readout-only delayed, but **1 immediate episode interleaved every 4** delayed episodes (co-adaptation from the first coupling step) | unfrozen, normal interleaved |
| `h6_freeze_readout` | readout-only delayed (as H5-A) | **head + SSM output (C, D) frozen**; GNN/pooling/state-generation trainable; immediate+delayed joint episodes — the immediate skill is restored/kept by representation adaptation under a static readout |

The H5 result already provides the arm-level baseline (original coupling+restoration);
`h6_control` re-runs it under the extended instrumentation.

**Measured at every stage (pre / post-coupling / post-restore):** immediate and delayed
accuracy, full substitution battery (correct-state ≥ 0.90 required, different-answer
following, zero/shuffled chance), state geometry (decode, margins, norms), readout logit
margins, and **per-group parameter displacement** (head / SSM-C / SSM-D /
state-generation / pooling / GNN) — to distinguish route-overwrite from shared-parameter
conflict from representation drift.

**Pre-registered success criteria:** immediate ≥ 0.90 AND delayed ≥ 0.90 AND the
substitution battery intact (same criteria table as §8b) at end of restoration. Improved
delayed accuracy *without* intact causal substitution does not pass.

## 8d. D-int s0 readout-path convergence diagnostic (pre-registered, separate track)

For D_interleaved seed 0 only, readout-only delayed training repeated from the *untouched*
original checkpoint under logging that H6's arms do not record:

- initial readout loss and loss trajectory;
- per-episode gradient norms per readout group (head / SSM-C / SSM-D);
- **gradient cosine similarity between the immediate-loss gradient and the delayed-loss
  gradient** over the trainable readout path (positive = locally aligned, ≈0 = weak
  interaction, negative = conflicting optimisation directions);
- logit margins during training;
- learning-rate sensitivity (3e-4 / 1e-3 / 3e-3);
- variability across repeated restarts from the identical checkpoint (different data-order
  seeds, copies everywhere — the original checkpoint is never written).

This distinguishes competing-objective failure from poorly-conditioned readout training;
by itself it does not establish the cause.

### §8d results — and a correction to H5's interpretation (both sources DO acquire)

With gradient clipping and varied data-order seeds (restarts), the diagnostic shows:

| LR | delayed at final probe (4 restarts) |
|---:|---|
| 3e-4 | 0.52 / 0.48 / 0.48 / 0.52 — never reaches the solution in 100 ep |
| 1e-3 | **1.000** / 0.48 / 0.52 / 0.52 |
| 3e-3 | 0.48 / 0.52 / **1.000** / 0.52 |

Direct Trainer-based probing of the *same* protocol (both phenotype-iii sources) reveals
why: **delayed accuracy oscillates** — 1.000 at episodes 100 and 150, chance at 125, 175,
200; the response-step logit margin cycles with a period of ≈ 20 episodes (0.95 → 0.02 →
0.25 → 0.02 → …), and the **immediate-vs-delayed gradient cosine flips in anti-phase with
the cycle** (≈ +0.45 when the delayed margin recovers, ≈ −0.37 as it decays; mean ≈ +0.1,
strongly bimodal). The coupled readout passes through the solution repeatedly but is **not
an attractor** of delayed-only readout training at any tested LR; final-probe "outcomes"
are phase samples.

**Correction (replaces the H5 outcome-B reading of D-int s0):** both phenotype-iii
sources *acquire* the coupled readout transiently. H5's stage-3 boundary measurements
(C s1 "rescued"; D s0 "not rescued") sampled different phases of the same oscillation.
What stage-4 joint training did to the C s1 run — hold delayed 1.000 with confident
margins through the full battery — remains valid as observed behaviour; whether joint
training *stabilised* an up-phase (immediate gradients counteracting the delayed-only
drift) or merely sampled another phase is exactly what H7 tests. No claim that the readout
path "cannot converge" survives; the open question is now **stability, not acquisition**.

## 8e. H7 — capture-and-hold (pre-registered design)

**Hypothesis:** the coupled readout solution is a transient, not an attractor, under
delayed-only readout training; capturing it at an up-phase and then training only the
representation under joint episodes (frozen readout) will hold both capabilities —
because the joint objective counteracts the delayed-only drift.

**Arms** (both phenotype-iii sources; 400-episode budget as before):

| Arm | Stage 3 | Stage 4 |
|---|---|---|
| `h7_capture_hold` | readout-only delayed; probe every 5 episodes; **freeze everything at the first probe with delayed ≥ 0.9** (capture event recorded) | readout **stays frozen**; GNN/pooling/state-generation trainable; joint immediate+delayed episodes |
| `h6_control` (existing) | no capture — run to 200 then restore | normal interleaved |

**Success rubric: unchanged from §8c** (immediate ≥ 0.90, delayed ≥ 0.90, substitution
battery intact at end of restoration). Recorded additionally: capture episode, immediate
accuracy at capture, margins at capture vs end.

### §8e results (`results/coupling_rescue_h7/`)

| source | capture | end immediate | end delayed | subst battery | rubric |
|---|---|---:|---:|---|---|
| C_staged s1 | **episode 5** (imm 0.805, del 1.000, reset 0.52) | **1.000** | **1.000** | same 1.000 / follows 1.000 / zero+shuffled chance / decode 1.000 | **ALL PASS** |
| D_int s0 | episode 60 (imm 1.000, del 1.000, reset 0.48) | 0.525 | 0.520 | all chance | fail |

C s1 end-state parameter displacement in stage 4: head/C/D **exactly 0** (frozen),
pooling 1.24, GNN 2.61, state-generation 0.48 — the representation adapted under joint
episodes while the captured readout held both capabilities; state-norm ratio 1.065,
decode preserved.

**Milestone (§8c question) met on one source:** ALS acquired persistent-state utilisation
*and* preserved the pre-existing immediate rule through subsequent joint training — via
capturing the coupled readout at an up-phase and holding it static while the
representation adapted. The intervention uses only schedule changes (probe-and-freeze +
encoder-only restore); no architecture change, no added state dimensions.

**D_int s0 — resolved by §8f/§8g (see below).** Its captured readout *can* be acquired
(episode 60, full up-phase behaviour). The H7 destruction under joint training was an
artifact of the **blocked** restore (100 consecutive immediate episodes, then 100
delayed); with exact-RNG reconstruction the per-episode-alternating regime holds both
capabilities, and the pre-registered single-variable intervention H7b (§8g) confirms it
on the full rubric.

---

## 8f. Per-group encoder gradient conflict (pre-registered design)

**Question:** where, in the shared representation pathway, do the immediate and delayed
objectives conflict during the regime that destroyed D s0's captured solution?

**Setup:** starts from the *captured* checkpoint of the H7 D-s0 run (preserved at
`results/coupling_rescue_h7/D_interleaved_seed000__h7_capture_hold/captured_checkpoint.pt`;
original never overwritten — the v1 run without checkpoint saving is kept as
`*_v1_no_ckpt`). Regime = H7 stage 4 exactly: readout frozen, joint immediate+delayed
episodes, representation trainable. The applied update alternates delayed/immediate as in
interleaved training; both gradients are measured every episode.

**Measured per parameter group (GNN / pooling / SSM state-generation), every episode:**
‖g_del‖, ‖g_imm‖, cos(g_del, g_imm), ‖g_del + g_imm‖; delayed logit margins; behavioural
probes every 5 episodes (the cadence that revealed the ~20-episode oscillation). Gradient
alignment is compared against actual accuracy changes — conflict alone does not
establish behavioural interference.

**Pre-registered possibility taxonomy:** (A) consistent negative alignment → persistent
representation-level conflict; (B) periodic sign changes → the representation cycles
between compatibility with one objective and the other; (C) conflict concentrated in one
group → targeted intervention candidate. No gradient surgery, separate optimisers, or
architecture changes in this experiment.

### §8f results — and a second protocol correction

With exact RNG continuation from the captured state, the reconstructed stage-4 regime
(per-episode alternating delayed/immediate updates, readout frozen) **does not destroy**
D s0's captured solution: after an initial decay (0.48–0.52 for ~25 episodes) the delayed
capability **re-locks and holds 1.000 from episode ~60 to 200** with a stable margin
(≈ 0.30), immediate stays 1.000 throughout, and once the immediate objective is satisfied
its gradient norms collapse to ≈ 0.003 (nothing left to push). Mean per-group gradient
cosines are weakly positive (GNN +0.13, pooling +0.08, ssm-gen +0.04; ~30% of episodes
negative) — **possibility A (consistent conflict) is rejected**; the cosine swings are
episode-scale, not locked to the 20-episode behavioural period (**B** not supported at
encoder level); gradient magnitude is concentrated in the GNN (‖g_del‖ ≈ 0.27 vs pooling
0.013) but its alignment is not systematically negative (**C** partially — magnitude, not
conflict, concentrates in the GNN).

The reconstruction differs from the actual H7 stage 4 in one respect that turns out to be
decisive: **the H7/H6/H5 restore stage alternated in blocked halves** (100 consecutive
immediate episodes, then 100 delayed), while the reconstruction alternates per episode.
Blocked phases let each objective's representation drift 100 steps away from the other's
solution; alternating updates never let it drift far. **Second correction of the day:**
the earlier "joint training destroys the captured solution" conclusion (H6 freeze arm, H7
D s0) is a property of the *blocked* restore, not of joint adaptation per se.

## 8g. H7b — alternating restore (pre-registered, single variable)

**Hypothesis:** restoring with strict per-episode alternation (instead of blocked halves)
from the captured state preserves both capabilities on both phenotype-iii sources.

**Manipulation:** identical to `h7_capture_hold` except stage 4 applies immediate and
delayed episodes in strict 1:1 alternation. Everything else — capture rule, frozen
readout, budget, rubric — unchanged. Predictions: D s0 rescued (per §8f reconstruction);
C s1 remains passing.

### §8g results (`results/coupling_rescue_h7b/`)

| source | capture | end imm | end delayed | subst same | follows | zero / shuffled | norm ratio | rubric |
|---|---:|---:|---:|---:|---:|---|---:|---|
| D_int s0 | ep 60 | **1.000** | **1.000** | 1.000 | 1.000 | chance / 0.54 | 0.87 | **ALL PASS** |
| C_staged s1 | ep 5 | **1.000** | **1.000** | 1.000 | 1.000 | chance / 0.59 | 1.07 | **ALL PASS** |

**The §8c milestone is met on BOTH phenotype-(iii) checkpoints**: ALS acquires
persistent-state utilisation and preserves the pre-existing immediate rule through
subsequent joint training, using only schedule-level changes (probe-and-freeze capture +
per-episode alternating restore with a static readout). No architecture change, no added
state dimensions, no new modules. The full intervention recipe: readout-only delayed
training from the failing checkpoint, freeze at the first delayed ≥ 0.9 probe, then
1:1 alternating joint episodes with the readout frozen while the representation adapts.

### Research-history note (corrections preserved)

Two corrections were made during this phase and are kept in the record: (1) the H5
"rescued vs not-rescued" difference between sources was phase sampling of an oscillating
coupled readout (§8d) — both sources acquire; (2) the "joint training destroys the
captured solution" conclusion (H6 freeze arm, H7 D s0) was an artifact of the blocked
restore schedule (§8f) — alternating joint adaptation preserves it. Both corrections were
driven by diagnostics run *after* the attractive interpretations were recorded.

---

## 8h. D800 population reliability (complete)

**Workstream separation (user-directed):** population reliability runs independently of
the H7b generalisation study; D800's configuration is frozen
(`configs/experiments/d800_population.yaml` differs from
`configs/experiments/d800_interleaved.yaml` only in experiment name and seed list).

**Design:** 20 additional independent seeds (3-22); original seeds 0-2 kept **visible
separately** (no silent pooling). Original per-seed success criterion: delayed accuracy
>= 0.9. Immediate accuracy, state-reset, and unseen-size reported per seed. Observed
success proportions reported with exact two-sided Clopper-Pearson 95% intervals
(`scripts/d800_population_analysis.py`; bisection on the binomial CDF, verified against
known values). Identical evaluation and checkpoint-reconstruction procedures; the
resume-guarded runner is re-invoked until all seeds complete.

### §8h results (`results/d800_population/`)

| sample | seeds | delayed >= 0.9 | 95% CI (exact) | immediate | state reset |
|---|---|---:|---|---:|---:|
| original (kept visible) | 0-2 | 3/3 = 1.000 | 0.292-1.000 | 1.000 (min 1.000) | 0.493 |
| population (new) | 3-22 | **20/20 = 1.000** | **0.832-1.000** | 1.000 (min 1.000) | 0.500 |
| combined (power only) | 0-22 | 23/23 = 1.000 | 0.852-1.000 | 1.000 (min 1.000) | 0.500 |

Every seed reaches delayed 1.000 and immediate 1.000, and in every seed the delayed
state-reset probe returns to chance (0.48-0.52) — the capability is state-carried in all
23, not just in the original three. Per-seed values: `reliability_summary.json`,
`curriculum/D800_interleaved/seed_*/seed_summary.json`.

**Unseen size (corrected protocol; all 23 recomputed with one identical procedure —
`scripts/recompute_population_unseen_size.py`).** The original seeds' recorded values
(0.505, 0.505, 1.000) include the known pre-fix probe artifact (missing generic blank);
recomputed they are 1.000, 1.000, 1.000, consistent with §3.4. Of the 20 population seeds,
19 are >= 0.985 and one (seed 19) is **0.560**. That seed is a genuine, reproducible
outlier, and it is not a state-utilisation failure: its in-distribution delayed and
immediate accuracies are both 1.000, and at n=200 its *immediate* accuracy degrades to the
same level as delayed (0.555 vs 0.560) while state reset stays at chance (0.495). The
failure tracks **mean degree, not node count**: at n=200 with density lowered to keep mean
degree (0.02 instead of 0.04) the same checkpoint returns immediate 1.000 and delayed
1.000; at n=300 (density 0.04) both fall to 0.535. So this checkpoint's encoder is
calibrated to the trained edge-density regime — a scale-generalisation limitation of that
seed, not a loss of persistent-state use.

**Bounded interpretation.** Within the tested regime (n=100, density 0.04, delay 3, 2000
training instances, 800 interleaved episodes), the frozen D800 configuration succeeds in
20/20 fresh seeds and 23/23 overall; exact 95% lower bounds are 0.83 (n=20) and 0.85
(n=23). This establishes D800 as a *reliable baseline in this regime*; it does not
establish a schedule law beyond it (other sizes, densities, delays, and instance counts
remain untested at population scale).

**Operational note.** Long runs are now launched as N parallel single-process workers with
disjoint seed lists (`--seeds`), each capped at 3 torch threads, logging to
`results/d800_population/logs/w*.log`; the runner prints a flushed per-seed `DONE` line.
This removed both the silent-pipe problem (`| Select-Object -Last N` buffers until exit)
and the ~2-seeds-per-45-min throughput limit.

## 8i. H7b generalisation to new failing checkpoints (complete)

**Question:** does capture-and-hold with per-episode alternating restore work beyond the two
checkpoints that motivated it?

**Source set (fixed before running, no phenotype selection):** every condition-seed in the
frozen curriculum study (`results/curriculum_study/`) whose delayed accuracy is < 0.9,
excluding the two already-used sources (`C_staged:1`, `D_interleaved:0`). That is
`A_immediate` seeds 0/1/2, `B_delayed` seeds 0/1/2, and `C_staged` seed 0 — seven
checkpoints spanning two distinct failure modes (immediate-only training with delayed at
chance; delayed training that never acquired the capability at all).

**Procedure:** identical to §8g — readout-only delayed training, freeze at the first
delayed >= 0.9 probe, then 1:1 alternating joint episodes with the readout static. Full
rubric, unchanged: immediate >= 0.9, delayed >= 0.9, same-answer substitution >= 0.9,
different-answer substitution follows >= 0.9, zero/shuffled at chance, state representation
preserved.

**Reporting:** pass/fail per source plus the observed pass proportion with an exact
Clopper-Pearson 95% interval. A source where the capture threshold is never reached is
reported as *not capturable* (informative about state content, not about the schedule), not
as a rubric failure.

**Not changed:** architecture, state dimensionality, optimiser, learning rate, budget,
capture rule, or rubric. No gradient surgery, no separate optimisers.

### §8i results (`results/coupling_rescue_h7b_generalisation/`)

State-content classification of the same checkpoints was measured **independently of the
intervention** (`scripts/run_state_analysis.py --results results/curriculum_study`; held-out
linear probe of the answer at the response step, plus the causal substitution battery):

| source | failure class | decode answer at response | capture | end imm | end delayed | subst follows | rubric |
|---|---|---|---:|---|---:|---:|---:|---|
| A_immediate s0 | computed then decayed | 0.465 | none | 1.000 | 0.480 | 0.520 | fail |
| A_immediate s1 | computed then decayed | 0.450 | none | 1.000 | 0.520 | 0.480 | fail |
| A_immediate s2 | **present but unused** | 1.000 | **ep 80** | 1.000 | **1.000** | **1.000** | **PASS** |
| B_delayed s0 | never computed | 0.450 | none | 1.000 | 0.480 | 0.520 | fail |
| B_delayed s1 | never computed | 0.450 | none | 1.000 | 0.480 | 0.520 | fail |
| B_delayed s2 | never computed | 0.450 | none | 1.000 | **1.000** | **1.000** | **PASS\*** |
| C_staged s0 | never computed | 0.450 | none | 1.000 | 0.480 | 0.520 | fail |

Across all **nine** failing checkpoints in the frozen study (the seven above plus the two
from §8g), the failure classes and outcomes are:

| class | n | answer in state at response | captured | full rubric pass |
|---|---:|---|---:|---:|
| never computed | 4 | no (0.450 at every point) | 0/4 | 1/4 (\*) |
| computed then decayed | 2 | no (1.000 at delay-1, 0.450 by delay-3) | 0/2 | 0/2 |
| **present but unused** | **3** | **yes (1.000)** | **3/3** | **3/3** |

**The intervention's applicability is set by state content, not by the schedule.** Where the
answer survives to the response step in the recurrent state, capture succeeds and the full
rubric passes **3/3** (exact 95% CI 0.292-1.000). Where it does not, readout-only training
has nothing to couple: 6/6 such checkpoints failed to reach the capture threshold, and 5/6
remained at chance after the full recipe. Overall pass rate on the nine failing checkpoints
is 4/9 (0.137-0.788).

**(\*) An unplanned second route, from `B_delayed` s2.** This checkpoint never captured
(stage-3 delayed stuck at 0.52, state decode 0.53) yet passed the full rubric after the
stage-4 phase: head and SSM C/D deltas were exactly 0.0, while GNN (3.60), pooling (3.26)
and state-generation (0.82) moved, and the answer's decodability went **0.53 -> 1.000**
(substitution follows 1.000). So encoder adaptation to a frozen, never-captured readout can
*create* the state representation the readout needs. This is a distinct mechanism from
capture-and-hold, observed in 1 of 6 answer-absent checkpoints; it needs replication across
data-order seeds before any claim is attached to it, and it does not weaken the
state-content rule above (the representation had to be created before the rubric could
pass).

**Bounded claims.** (1) For checkpoints whose state still carries the answer at response
time, capture-and-hold with per-episode alternating restore passes the full pre-registered
rubric 3/3 — this is the generalisation §8i asked for, and it is now demonstrated on three
independent failure histories rather than two. (2) For checkpoints whose state does not
carry the answer, the schedule is not the limiting factor: readout-level training cannot
manufacture a representation that is not there, and 5/6 stay at chance. (3) Schedule-only
interventions therefore do not generalise to all failures; they generalise exactly to the
class they target, and that class is identifiable in advance by a state probe.

## 8j. B2 replication — encoder-side acquisition (pre-registered, running)

**Question:** is the `B_delayed` s2 result — a full-rubric pass with **no capture event**
under an exactly frozen readout, in which the answer's decodability rose 0.53 -> 1.000 —
a repeatable second acquisition route, or an isolated trajectory?

**Design:** the same source checkpoint and the same arm (`h7b_capture_alternate`), with
**only the training data order varied**. `run_coupling_arm` now takes an optional
`data_seed` that re-keys the environment RNG streams (delayed and immediate) while the
model initialisation, the checkpoint, the evaluation datasets, the generic blank
observation and the probe CV splits all stay keyed to the source seed. Three independent
data seeds (3, 4, 5); the original run (data order 2) is kept visible separately, not
pooled.

**Measured per replicate:** initial state decodability; head and SSM C/D parameter deltas
(must remain exactly 0.0 for the "frozen readout" claim to hold); GNN / pooling /
state-generation deltas (where the adaptation occurred); post-stage answer decodability;
same-answer and different-answer substitution; zero and shuffled controls; immediate and
delayed accuracy; whether a capture event occurred.

**Interpretation (fixed before running):** 3/3 full-rubric passes → strong evidence for a
second acquisition route in this setup; 2/3 → promising but data-order sensitive; 0-1/3 →
the original result remains an isolated observation and the distinguishing feature of B2
must be investigated. Functional outcomes decide; parameter displacement *magnitudes* are
not required to match across seeds.

### §8j results (`results/coupling_rescue_b2_replication/`)

| data order | capture | decode pre → post | head/C/D delta (stage 4) | GNN / pooling / gen | imm | delayed | same | follows | zero / shuffled | rubric |
|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|
| 2 (original) | none | 0.530 → 1.000 | 0.000 / 0.000 / 0.000 | 3.60 / 3.26 / 0.82 | 1.000 | 1.000 | 1.000 | 1.000 | 0.520 / 0.460 | PASS |
| 3 | none | 0.530 → 1.000 | 0.000 / 0.000 / 0.000 | 3.64 / 3.25 / 0.74 | 1.000 | 1.000 | 1.000 | 1.000 | 0.520 / 0.460 | PASS |
| 4 | none | 0.530 → 1.000 | 0.000 / 0.000 / 0.000 | 3.64 / 3.41 / 0.80 | 1.000 | 1.000 | 1.000 | 1.000 | 0.480 / 0.460 | PASS |
| 5 | none | 0.530 → 1.000 | 0.000 / 0.000 / 0.000 | 3.57 / 3.19 / 0.82 | 1.000 | 1.000 | 1.000 | 1.000 | 0.520 / 0.460 | PASS |

**Verdict against the pre-registered rule: 3/3 → strong evidence for a second acquisition
route in this setup.** Under three independent training data orders, the same checkpoint
reaches the full rubric with **no capture event**, with head and SSM C/D exactly frozen
(deltas 0.000), by moving the representation pathway, and the answer's decodability rises
from 0.530 to 1.000. The causal substitution battery confirms the result is state-read, not
a reroute (same 1.000, different-answer follows 1.000, zero/shuffled at chance).

**Mechanistic reading (bounded).** In stage 3 the readout is trained on a state that carries
no answer: it settles on an *uninformative but fixed* decision direction. In stage 4 the
readout is frozen, so the encoder must instead shape the state to that direction — and it
does, in every replication. Acquisition and utilisation therefore occur in **different
stages**: the utilisation mapping exists (as a fixed direction) before there is anything to
utilise, and acquisition is what makes it functional. This is a distinct route from
capture-and-hold, where utilisation is learned after acquisition.

**Scope.** The route is data-order robust *for this checkpoint* (3/3), but rare across
checkpoints (1 of 6 answer-absent checkpoints). It should not be merged into H7b: the
mechanisms differ in which side of the interface adapts. Whether the route generalises to
other answer-absent checkpoints, or depends on the particular frozen direction that stage 3
happened to settle on, is open.

## 8k. Delay scaling — evaluation only (complete)

**Question:** after the capability is acquired, which of the three kinds of retention
survive longer delays? These are reported separately and never collapsed into one number.

- **information retention** — is the answer still decodable from the state at the response
  step (held-out linear probe; criterion >= 0.95)?
- **functional retention** — does that state still *control* the response (causal
  substitution: different-answer follows >= 0.9, zero and shuffled at chance <= 0.65)?
- **behavioural retention** — is the response correct (delayed accuracy >= 0.9)?

**Design:** all 23 successful D800 checkpoints (original 3 + population 20), evaluated at
delays 3, 10, 30 and 100. **No training, no parameter change, no state-dimension change** —
each checkpoint is loaded and probed. Per checkpoint and delay: delayed accuracy, state
reset control, probe decodability at the response step, mean state norm, mean logit margin,
and the full substitution battery. `scripts/run_delay_scaling.py`, sharded across workers;
outputs `results/delay_scaling/` (one CSV row per checkpoint x delay + per-seed
trajectories in `summary.json`). Individual trajectories are reported, not pooled means.

**Working hypothesis (to be tested, not assumed):** behavioural retention will fail before
information retention does, because §7 already established that a decodable answer can be
present in the state while being functionally unused. If that pattern appears under delay,
it is the same dissociation as a function of time rather than of training.

### §8k results (`results/delay_scaling/`)

All 23 successful checkpoints, no training, no parameter change, 92 evaluations (23 x 4):

| delay | information (decode >= 0.95) | functional (follows >= 0.9, controls at chance) | behavioural (>= 0.9) | mean margin | mean state norm |
|---:|---|---|---|---:|---:|
| 3 (trained) | 23/23 | 23/23 | 23/23 | 3.63 | 1.94 |
| 10 (3.3x) | **23/23** | **23/23** | **23/23** | 1.77 | 1.62 |
| 30 (10x) | **23/23** | **11/23** | **11/23** | 0.57 | 1.52 |
| 100 (33x) | 0/23 | 0/23 | 0/23 | 0.59 | 1.52 |

**The hypothesis is confirmed, and the dissociation is bimodal.** At delay 30 every single
checkpoint still carries a linearly decodable answer at the response step (decode 1.000 in
23/23, and the causal substitution battery agrees: 11 follow the donated answer, 12 are at
chance), yet only 11/23 still *use* it. There are no intermediate cases: 11 checkpoints are
at 1.00/1.00/1.00 and 12 are at ~0.50 behaviour with decode 1.000. So behavioural retention
fails before information retention, and it fails as a discrete switch rather than a gradual
decay. At delay 100 the information itself is gone (decode ~0.53), so that is a different
failure — representation loss rather than disconnection.

**The delay-30 failures are confidently wrong, not weakly signalled.** The 12 checkpoints
that fail at delay 30 have *larger* response margins than the 11 that pass (mean 0.74, range
0.35-1.18 vs mean 0.38, range 0.20-0.42) and larger mean state norms (1.65 vs 1.39). A
decayed or shrunken signal would show the opposite. The pattern instead indicates that the
carried state has moved (rotated/shifted) relative to the *fixed* readout direction while
remaining linearly separable — i.e. the same "answer present but no longer aligned with the
readout" phenotype that §7 and §8i found after training, here produced by time alone. This
is a hypothesis about geometry; it is supported by decode-plus-margin, not yet by a direct
measurement of the angle between the probe direction and the readout weights.

**Per-checkpoint trajectories** (behaviour / decode / follows at 3, 10, 30, 100) are in
`results/delay_scaling/summary.json`. Representative: `original seed 1` and `population
seeds 3, 5, 7-10, 15, 19, 20` hold all three to delay 30; `original seed 0`, `population
seeds 4, 6, 11-14, 16-18, 21, 22` hold information but lose use at delay 30; all 23 lose
information by delay 100.

**Bounded claims.** (1) Within this architecture and task, persistent-state utilisation is
perfectly retained to at least 10 steps (3.3x the trained delay) on all three measures.
(2) Between 10 and 30 steps the capability breaks in about half the population, and it
breaks as a *disconnection* (information present, use gone), not as forgetting.
(3) Information survives longer than use but not indefinitely: by 100 steps (33x) the state
no longer carries the answer. (4) These are evaluation-only results on the trained
checkpoints; no intervention was attempted at longer delays.

## 8l. Geometry predicts rescue at delay 30 (two-stage, pre-registered)

**Question.** §8k produced 12 checkpoints that fail behaviourally at delay 30 while their
state still decodes the answer at 1.000. Two geometric explanations are possible and they
make different predictions:

- **rotation** — the carried state has rotated away from the fixed readout direction, so
  the probe direction and the effective readout direction diverge;
- **translation** — the directions still agree, but the state distribution has shifted
  along the readout direction relative to the fixed decision boundary (equivalently, the
  readout's threshold is displaced for the shrunken delay-30 state), so one class falls
  entirely on the wrong side.

These are separable in one coordinate system. The probe boundary is
`w_probe . (s - mean) = 0`. The SSM applies `tanh` to the state before the readout
(`s_t = act(A s_{t-1} + B z_t)`, `y_t = C s_t + D z_t`), so the state→logit map is **not
globally linear** and the readout boundary is not a single hyperplane in the carried-state
space. We therefore use the **effective local readout direction**
`w_readout = d(logit1 - logit0)/ds` at the mean state — the gradient includes C and the
state nonlinearity — for alignment (cosine and projections), and compute the boundary
position, readout accuracy and prediction agreement from the **actual response-step
logits** returned by the model. A unit test checks the gradient against a finite
difference of the actual margin.

> **Correction (recorded before stage-A results).** An earlier version of this
> pre-registration stated that `logit1 - logit0 = w_readout . s + c` exactly. That is
> false: the SSM applies `tanh` to the state, so the map is only locally linear. The
> statistic definition was corrected to use actual logits for the boundary and accuracy,
> and the local gradient only for direction. The pilot was re-run under the corrected
> definition and gave the same numbers (cosine, boundary positions and agreement unchanged
> to two decimals).

**Pilot (disclosed, not counted as stage-A evidence).** Five checkpoints at delay 30
(failing: original 0, population 6, 22; passing: original 1, population 5). Cosine was
0.982–0.999 in both groups. The failing checkpoints classified one class at 1.000 and the
other at 0.000, with the boundary 1.1–1.8 class-separations from the midpoint
(`boundary_vs_mid` -1.67, +1.75) and probe–readout agreement 0.48–0.52; the passing
checkpoints had `boundary_vs_mid` -0.45, -0.10 and agreement 1.000. The pilot therefore
favours translation. Stage A tests this on all 23 checkpoints; the pilot is reported so the
hypotheses are not presented as post-hoc.

### Stage A — geometry before any intervention

All 23 successful checkpoints at delays 3, 10 and 30, no parameter change
(`scripts/run_readout_geometry.py`, `results/readout_geometry/`). Per checkpoint x delay:
held-out probe decodability; cosine between probe and effective readout direction;
per-class and overall readout accuracy; the normalised boundary position
`boundary_vs_mid = -(t1+t0)/2 / (t1-t0)` where `t0,t1` are the class-mean *actual*
margins (0 = boundary midway between the class means; |value| > 0.5 = one class entirely
on the wrong side); signed and absolute margin; state norm; projections onto each
direction; and probe–readout prediction agreement. Groups at delay 30 are defined by the already-published
behavioural outcome (`delayed_intact >= 0.9`), and the recomputed behaviour is checked
against `delay_scaling.csv`.

**Pre-registered hypotheses (fixed before stage A runs):**

- **A1 rotation:** failing delay-30 checkpoints show lower cosine than passing ones.
  Rejected if failing-group cosine >= 0.90.
- **A2 translation:** failing delay-30 checkpoints show |`boundary_vs_mid`| > 0.5 and
  prediction agreement near chance, while passing checkpoints show |`boundary_vs_mid`| <=
  0.5 and agreement near 1.0, **with cosine preserved**.
- **A3 decay:** failing checkpoints show smaller state norm and smaller absolute margin
  than passing ones. Rejected if the failing group is larger on either.

**Stage-A prediction handed to stage B.** A failing checkpoint is predicted rescuable by
readout-only retraining iff its information is present (`decode >= 0.95`) and its effective
readout direction is aligned (`cos >= 0.90`). If A2 holds, all 12 failing checkpoints are
predicted rescuable: a threshold displacement is exactly what readout-only training can
correct, because the frozen encoder already writes a decodable state in the direction the
readout uses.

### Stage A results (`results/readout_geometry/`)

All 23 checkpoints x delays 3/10/30; behaviour reproduced exactly (`max |diff| = 0.000000`
against `delay_scaling.csv`). Groups are split by the behavioural outcome at the same
delay.

| delay | group | n | cos(probe, readout) | agreement | mean \|boundary_vs_mid\| (max) | mean \|margin\| | state norm | class 0 / class 1 acc |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | pass | 23 | 0.975 | 1.000 | 0.014 (0.047) | 3.63 | 1.94 | 1.00 / 1.00 |
| 10 | pass | 23 | 0.978 | 1.000 | 0.066 (0.161) | 1.77 | 1.62 | 1.00 / 1.00 |
| 30 | pass | 11 | 0.980 | 1.000 | 0.279 (0.452) | 0.38 | 1.39 | 1.00 / 1.00 |
| 30 | **fail** | **12** | **0.982** | **0.493** | **1.110 (1.749)** | **0.74** | **1.65** | **0.33 / 0.67** |

**A1 (rotation): rejected.** The failing group's cosine is 0.982, slightly *higher* than
the passing group's 0.980 (individual range 0.940–0.999). The probe direction and the
effective readout direction are the same direction in every checkpoint, including every
failure.

**A2 (translation): confirmed.** Every one of the 12 failing checkpoints has
|`boundary_vs_mid`| > 0.5 (range 0.523–1.749; 8 fail with class 0 entirely wrong, 4 with
class 1 entirely wrong), and probe–readout agreement is at chance (0.480 or 0.520). The 11
passing checkpoints have agreement 1.000 and max |`boundary_vs_mid`| 0.452. At delays 3 and
10 every checkpoint is aligned (max |`boundary_vs_mid`| 0.047 and 0.161), so the
displacement is specific to delay 30.

**A3 (decay): rejected.** The failing group has *larger* absolute margins (0.74 vs 0.38)
and larger state norms (1.65 vs 1.39), not smaller. A decayed signal would show the
opposite; this is the confidently-wrong signature.

**Reading.** At delay 30 the answer is still present, still linearly decodable, and still
written in essentially the same direction the readout uses; what has moved is the decision
boundary along that direction. Because the state distribution has contracted and shifted,
both class means end up on one side of the fixed boundary, so one class is confidently
called the other. This is a **translation of the decision boundary**, not a rotation of the
representation and not a loss of information. It also explains why the failures are
confidently wrong rather than weakly signalled.

**Stage-A prediction (recorded per checkpoint in `summary.json`): 12/12 failing checkpoints
predicted rescuable**, because a boundary displacement is precisely the defect that
readout-only retraining (head bias + C/D) can correct while the encoder already writes an
aligned, decodable state. Stage B tests that prediction; it is a prediction, not a result,
and decodability alone is never treated as use.

### Stage B — H7b at delay 30

The 12 delay-30 failing checkpoints, same arm (`h7b_capture_alternate`), same budget and
rubric as §8g/§8i, with `--delay 30` applied to the training environments, the capture
probes, the state summaries and the final battery. **Delay 30 is preserved throughout**;
the checkpoint is the original trained model, so this tests whether the intervention works
*at the long delay*, not whether it works after returning to the training delay. Procedure
unchanged: readout-only delayed training, capture at the first probe with delayed
accuracy >= 0.9, freeze the readout, then 1:1 alternating joint episodes, then the full
causal substitution battery.

**Budget note (recorded before running).** The budget is the standard 400 optimizer
episodes, unrolled for `sequence_length = 32` steps each. At delay 3 that is ~8 trials per
episode; at delay 30 a 31-step trial fills one episode, so the same episode budget contains
only ~400 response events instead of ~3200. The budget is kept identical to §8g/§8i so the
procedure is unchanged, and the response-event count is reported as a scope caveat. A
failure to capture under this budget is therefore evidence about the *standard* budget, not
a demonstration that the representation is inaccessible in principle.

**Pre-registered outcome rule (fixed before running):**

- **12/12 full rubric** → strong evidence that delay-induced disconnection is addressable
  by the same intervention, i.e. the geometry prediction holds across the cohort;
- **partial** → heterogeneous mechanisms or checkpoint-dependent susceptibility; compare
  stage-A geometry (cosine, agreement, `boundary_vs_mid`, margin, norm) between rescued and
  non-rescued checkpoints, with a statistic counted as predictive only if the two groups
  are rank-separated without overlap (n = 12; exact counts, no p-values);
- **0/12** → delay-induced disconnection is mechanistically different from the
  training-induced phenotype;
- **capture never fires** → readout-only coupling did not access the retained representation
  within the standard 400-episode budget (see the budget note above); a budget-matched
  follow-up would be required before claiming inaccessibility in principle.

A checkpoint is never classified as rescued from accuracy alone: the full rubric requires
immediate and delayed accuracy, same-answer substitution, different-answer following,
zero/shuffled controls at chance, and preserved decodability. The geometry is measured
before the intervention and in a separate results directory, so the intervention cannot
contaminate the diagnostic.

### Stage B results (`results/delay30_h7b/`)

All 12 runs show the **same pattern, with no exceptions**:

| measure | at capture / stage 3 (readout-only) | after stage 4 (joint, readout frozen) |
|---|---|---|
| capture event | **12/12**, at episodes 5–15, delayed = 1.000 at capture | — |
| delayed accuracy | **1.000** (12/12) | 0.48–0.52 (**0/12** >= 0.9) |
| probe decodability | 1.000 | 0.29–0.52 |
| same-answer substitution | 1.000 | 0.48–0.52 |
| different-answer follows | **1.000** | 0.48–0.52 |
| zero / shuffled controls | 0.48–0.52 / 0.48–0.58 (chance) | 0.48–0.52 (chance) |
| immediate accuracy | 1.000 | 1.000 |
| head / C / D delta | 0.011–0.029 / 0.022–0.080 / 0.023–0.103 | **0.000 exactly** (frozen) |
| GNN / gen delta | 0.000 (frozen) | 0.81–1.35 / 0.11–0.20 |

**Pre-registered full-rubric outcome: 0/12.** But the pre-registered interpretation
"0/12 → mechanistically different" does **not** apply as written, because the failure is
not at the access step:

- **Capture fired in 12/12.** Readout-only training at delay 30 reached delayed accuracy
  1.000 within 5–15 episodes in every checkpoint.
- **The readout-only stage alone reached the full rescue signature in 12/12**: delayed
  1.000, decode 1.000, same-answer 1.000, different-answer follows 1.000, zero/shuffled at
  chance, immediate 1.000. (The stage-3 analysis is a diagnostic instrumented in the
  procedure, not the success criterion; it is reported as such.)
- **The joint stage erased it in 12/12** while the readout stayed exactly frozen
  (head/C/D delta 0.000) and the encoder moved (GNN 0.81–1.35).

So the stage-A prediction is **confirmed at the level it was made**: the retained
representation is accessible to readout-only coupling at delay 30 in every failing
checkpoint, and repairing the boundary is sufficient to restore both behaviour and causal
state use. Delay-induced disconnection is not a different *accessibility* mechanism from
the training-induced one; the same intervention reaches it the same way.

**Why the joint stage fails at delay 30 but worked at delay 3.** The H7b joint stage is
"1:1 alternating" in *episodes*, and an episode is a fixed 32-step window. Measured
directly on the training environment, supervised response events per 32-step episode are:
immediate (delay 0) **32**, delay 3 **8**, delay 10 **2**, delay 30 **1**. Stage 4
alternates 100 delayed episodes with 100 immediate episodes, so it delivers 3200 immediate
events against **100** delayed events at delay 30 (32:1) but 3200 against 800 (4:1) at
delay 3 — an 8x smaller delayed share. Two further observations sharpen this:

- The successful delay-3 H7b runs moved the encoder **more** in stage 4 (GNN delta
  1.29–3.64) than these delay-30 failures (0.81–1.35). Raw encoder displacement is not what
  destroys the capability; the delayed task simply supplies almost no gradient to defend
  its representation under a 32:1 immediate majority.
- The delay-30 state is the accumulation of 31 decay steps, so a given encoder change has
  31 steps to compound before the response; the same drift is far more disruptive than at
  delay 3. (This is a hypothesis; the event-balanced test below separates the two.)

**Bounded claims.**
(1) Against the pre-registered rule the outcome is 0/12, and that is what is reported; the
stage-3 rescue is a diagnostic, not a substitute criterion.
(2) The erosion is caused by the *joint schedule*, not by readout inaccessibility: capture
and readout-only rescue succeeded in 12/12 with the causal battery intact.
(3) "1:1 alternating joint episodes" is not delay-invariant; it is 1:1 only in episodes,
and at delay 30 that is ~32:1 in supervised events. The recipe must be balanced in response
events to be tested fairly at long delays.
(4) Whether an event-balanced joint stage preserves the rescue is open; the captured
checkpoints (`captured_checkpoint.pt`, saved at the capture point, before stage 4) preserve
the rescued models for exactly that one-variable test.

## 8m. Event-balanced joint stage at delay 30 (pre-registered)

**Question.** §8l stage B left two explanations for why the joint stage erased a rescue
that readout-only training had achieved in 12/12:

- **S1 schedule imbalance** — 1:1 alternating *episodes* deliver 32:1 immediate:delayed
  response events at delay 30 (measured: 32/8/2/1 events per 32-step episode at delays
  0/3/10/30), so the delayed task supplies too little gradient to defend its
  representation;
- **S2 intrinsic fragility** — the long-delay state (31 decay steps of compounding) cannot
  be maintained by encoder-only joint updates even when the events are balanced; the
  bottleneck is readout compatibility of the long-delay state, not the schedule.

**Design.** Start from each **captured checkpoint** of §8l stage B
(`captured_checkpoint.pt`, saved at the capture point before stage 4 — the readout-only
rescue state, decode 1.000, substitution follows 1.000) and run **only** the joint stage,
with exactly one variable changed: the alternation is balanced in **response events**
instead of episodes. One delayed episode at delay 30 contains 1 response event per batch
element; it is paired with one immediate *step* (`sequence_length = 1` at delay 0, 1
response event), so every pair delivers equal events. The number of delayed episodes is
the same 100 as in §8l stage B; only the immediate side changes (100 steps instead of 100
32-step episodes). The readout stays frozen throughout (`encoder_only`; head/C/D deltas
must be exactly 0.000 for any conclusion to hold), the delayed data stream continues from
the RNG state saved at capture, and delay 30 is preserved in training, probes, state
summaries and the final battery (`scripts/run_balanced_stage4.py`,
`results/delay30_balanced/`).

**Measured per run:** delayed/immediate accuracy trajectory during the balanced stage
(probe every 10 pairs); pre (captured) and post state summaries with the full substitution
battery; the final battery and the **same pre-registered full rubric** as §8l/§8g;
parameter deltas; and a **secondary** post-stage geometry measurement (whether any erosion
is again a boundary translation or a loss of decodability).

**Interpretation (fixed before running):**

- **12/12 full rubric** → S1: the schedule imbalance was the cause; H7b generalises to
  delay 30 once the joint stage is event-balanced.
- **0/12** → S2: even with balanced events, encoder-only joint updates destroy the
  delay-30 rescue; the capability is not maintainable by the standard recipe at this
  delay, and readout compatibility of the long-delay state is the binding constraint.
- **partial** → mixed; compare trajectories and post-stage geometry between preserved and
  eroded runs, with a statistic counted as predictive only if rank-separated without
  overlap (n = 12; exact counts, no p-values).

If any captured checkpoint is missing, that run is reported as missing, never silently
excluded.

### §8m results (`results/delay30_balanced/`)

All 12 runs completed, all 12 with the same pattern:

| measure | value |
|---|---|
| starting point (captured rescue state) | decode **1.000**, follows **1.000** in 12/12 |
| delayed accuracy after the **first 10 pairs** | 0.48–0.52 (**12/12 already collapsed**) |
| final delayed accuracy | 0.48–0.52 (**0/12** >= 0.9) |
| final decodability | 0.29–0.52 |
| follows / zero / shuffled | 0.48–0.52 / chance / chance |
| immediate accuracy | 1.000 |
| head / C / D delta | **0.000 exactly** in 12/12 (readout frozen throughout) |
| GNN / gen / pooling delta | 0.89–1.55 / 0.10–0.19 / 0.29–0.72 |

**Pre-registered outcome: 0/12 → S2.** The schedule imbalance is *not* the cause. With
events balanced 1:1 (16:16 per pair) and the identical 100 delayed episodes, the rescue
still collapses — within the first 10 pairs, before most of the encoder displacement has
even accumulated. The trajectory is a fast destabilisation, not a gradual decay.

**The failure is representational, not another boundary translation.** Post-stage geometry
is degenerate in a qualitatively different way from §8l stage A: the class separation
collapses (decode 0.29–0.52, at chance), the probe direction becomes ill-defined, and the
boundary statistic loses meaning. After erosion the state no longer carries a linearly
decodable answer at all — unlike the original delay-30 failures, where decode was 1.000
and only the boundary was displaced.

**Comparison with §8l stage B pins the mechanism.** The balanced stage moved the encoder by
the same amount (GNN delta 0.89–1.55) as the episode-based stage (0.81–1.35) despite a 32x
smaller immediate share, and with the same delayed exposure. The drift is therefore not
primarily the immediate task's doing: **encoder-only updates at delay 30 are destabilising
in themselves.** Under a frozen readout, small encoder changes compound across the 31-step
delay, and the state leaves the region in which the answer is decodable at all. At delay 3
the same recipe holds because 4-step compounding is forgiving (H7b 3/3 with *larger*
encoder displacement, 1.29–3.64); at delay 30 it does not.

**Bounded claims.**
(1) Against the pre-registered rule, §8m is 0/12 → S2: the binding constraint at delay 30
is not the joint schedule's event balance but **readout compatibility of the long-delay
state** — the encoder cannot be updated at delay 30 under a frozen readout without
destroying the rescue within a few episodes.
(2) This sharpens the §8l reading into a precise trade-off: the rescue is *achievable*
(readout-only, encoder frozen: 12/12) and *unmaintainable* (any encoder-only joint update:
12/12). The readout can learn to use the state only while the encoder stands still; the
encoder cannot move without breaking what the readout learned.
(3) Which side of the balanced stage supplies the destabilising gradient (delayed or
immediate) is open — both trained here; isolating it would need a delayed-only or
immediate-only joint stage, which is outside the pre-registered design and is not claimed.
(4) The H7b recipe remains valid at its design delay and does not transfer to delay 30 by
schedule balancing alone; long-delay maintenance would require a mechanism that keeps the
encoder and the frozen readout jointly calibrated (e.g. constrained or readout-aware
encoder updates) — a candidate for future work, not a result.

## 8n. Gradient isolation at delay 30 (pre-registered)

**Question (left open by design in §8m):** which objective destabilises the delay-30
rescue — the delayed one, the immediate one, or only their combination?

**Design.** The same 12 captured checkpoints as §8m (the readout-only rescue state:
decode 1.000, substitution follows 1.000). Readout frozen in both arms
(`encoder_only`; head/C/D deltas must be exactly 0.000), and exactly one training stream
active per arm:

- **`delayed_only`** — 100 delayed episodes at delay 30: the identical delayed exposure
  of §8m and §8l stage B, continuing the same delay-30 data stream from the RNG state
  saved at capture;
- **`immediate_only`** — 100 immediate steps (`sequence_length = 1`, delay 0): the
  identical immediate exposure of §8m's balanced stage.

This completes the 2x2 decomposition of the joint stage: **no updates** (the captured
state itself), **delayed-only**, **immediate-only**, and **both** (§8m). Doses match §8m's
per-side exposure exactly, so any outcome difference is attributable to *which* objective
is active, not to dose (`scripts/run_isolation_arms.py`, `results/delay30_isolation/`).

**Measured per run:** delayed and immediate accuracy at a probe schedule (episodes 1–5,
then every 10); at the **first** probe with delayed accuracy < 0.9, an explicit
decodability measurement (held-out ridge decode, state norm, margin) to locate the
earliest representational collapse and distinguish a boundary translation (decode intact)
from representational loss (decode collapsed); pre and post full state summaries with the
substitution battery; the final battery and the **same pre-registered full rubric**;
parameter deltas; and post-stage geometry (secondary).

**Interpretation (fixed before running):**

- **delayed_only collapses (0/12), immediate_only does not (12/12)** → the delayed
  objective itself is destabilising under a frozen readout;
- **immediate_only collapses, delayed_only does not** → immediate-task adaptation destroys
  the long-delay representation;
- **both collapse** → either objective alone suffices; no interaction is needed;
- **neither collapses** → the interaction between the two objectives is necessary; §8m's
  collapse requires concurrent streams.

Partial outcomes are interpreted per arm with the rank-separation rule used throughout.
Two caveats, fixed in advance: (i) conclusions hold for the §8m-matched doses — a larger
immediate dose (e.g. the 100 full episodes of §8l stage B) is *not* tested here; (ii)
equal encoder displacement must not be read as equal functional consequence — displacement
and outcome are reported separately per arm.

### §8n results (`results/delay30_isolation/`)

The completed 2x2 decomposition of the joint stage:

| arm | updates | full rubric | collapse | failure mode |
|---|---|---|---|---|
| none (captured state) | — | 12/12 by construction | — | — |
| **delayed_only** | 100 delayed episodes | **0/12** | **episode 1, in 12/12** | representational |
| **immediate_only** | 100 immediate steps | **3/12** | episodes 1–50, heterogeneous | mostly compatibility |
| both (§8m) | 100 balanced pairs | 0/12 | within 10 pairs | representational |

**The delayed objective alone destroys the rescue — in one update.** All 12 collapse
between the pre-probe and the episode-1 probe (delayed 1.000 -> 0.48–0.52) and never
recover across 100 episodes. Decode at the collapse point is 0.45–1.00 (median 0.52) and
0.28–0.52 at the end: the representation itself is destroyed, not just the boundary.
GNN delta 0.85–1.15. The delayed updates also damage a task they never train on:
immediate accuracy falls to 0.735–1.000 (mean 0.93).

**The immediate objective is heterogeneous and mostly non-representational.** Three seeds
survive all 100 steps outright (seeds 4 and 13 hold 1.000 throughout; seed 11 dips at
episodes 10–60 and recovers by 70). Of the 9 failures, **7 have decode 1.000 at the
collapse point** — boundary translation with the information intact — and the encoder
drift is 10–50x smaller (GNN delta 0.02–0.13) than in delayed_only. Three outliers
(seeds 0, 16, 17; delta 1.00–1.46) resemble the delayed-arm collapse. Immediate accuracy
stays 1.000 (the arm trains on it).

**Why the asymmetry (measured at the captured point).** The two tasks are solved at very
different confidences. Delayed at delay 30: mean |margin| 0.26–0.42. Immediate at
delay 0: mean |margin| 4.4–7.1. The immediate objective therefore produces near-zero
gradients — the encoder barely moves, and where it does fail, the state drifts slowly past
the frozen boundary while the answer stays decodable. The delayed objective still has a
substantive loss to reduce at delay 30; its gradient becomes a full-size encoder step into
the state-generation pathway — the very pathway that writes the code the frozen readout
reads — and a single step is enough.

**Verdict against the pre-registered rule: the "partial" branch — heterogeneous
mechanisms.** The delayed objective is *sufficient* to destroy the rescue (12/12,
instantly, representationally); the immediate objective is not (3/12 survive, and 7 of the
9 remaining failures preserve decode). The §8m collapse is therefore attributable to the
delayed side of the joint stage.

**Bounded claims.**
(1) Under the tested doses, the delayed objective alone destroys the delay-30 rescue in
12/12 within one episode; the immediate objective alone preserves it in 3/12 and fails
non-representationally in 7 of the 9 remaining.
(2) Dose caveat as pre-registered: a larger immediate dose (100 full episodes, the §8l-B
side) is not tested here.
(3) The margin asymmetry is measured; the gradient-to-update account is an inference from
it, not a direct measurement of gradient magnitudes.
(4) Nothing here contradicts §8m: balancing events could not help because the
destabiliser was the delayed side all along. The binding constraint is readout
compatibility of the long-delay state **under delayed-gradient encoder updates**.

## 8o. Where the representation dies between delay 30 and 100 (pre-registered, evaluation only)

**Question.** §8k established the coarse picture: delay 10 retains everything (23/23),
delay 30 splits the population (information 23/23, behaviour 11/23), delay 100 has lost
the information itself (0/23). What happens between those points, and is it the same
event in every checkpoint? This is a *different* failure from §8l/§8m: at delay 30 the
information exists (and can even be repaired); by delay 100 there is nothing left to
reconnect.

**Design (evaluation only, no training, no parameter change).** All 23 successful
checkpoints x a fixed delay ladder
`[3, 10, 14, 18, 22, 26, 30, 40, 50, 65, 80, 100]` — ~4-step resolution across the 10–30
region where §8k found the behavioural switch, coarser beyond. Per checkpoint x delay:
behavioural retention and the state-reset control; held-out ridge decodability of the
answer at the response step; decodability at intermediate trial points (presentation,
first delay step, quarter/half/three-quarter points) to locate *where in the trial* the
information dies; per-dimension point-biserial correlation between each state dimension
and the label; mean state norm and mean absolute margin; across-instance dispersion of
the response-step state and the drift of the mean state from its delay-3 position
(attractor diagnostics). The causal substitution battery is **not** repeated here — §8k
already measured it at 3/10/30/100; a functional check at a newly located collapse point
would be a separate pre-registered follow-up (`scripts/run_delay_transition.py`,
`results/delay_transition/`).

**Pre-registered definitions.** Information threshold: decode >= 0.95 (as §8k); `D_info`
= first ladder delay with decode < 0.95. Behavioural threshold: delayed accuracy >= 0.9;
`D_behav` = first ladder delay below it. Abrupt vs gradual: abrupt if decode at `D_info`
<= 0.65, gradual if strictly between 0.65 and 0.95. Code carriers: dimensions with
|point-biserial r| >= 0.5 at delay 3 (§7 found a sparse ~3-of-8 code); a carrier is lost
at the first ladder delay with |r| < 0.3.

**Questions fixed in advance:** (1) the distributions of `D_info` and `D_behav` across
the 23 checkpoints, and whether behaviour still fails before information at this
resolution; (2) abrupt vs gradual, per checkpoint; (3) whether the same dimensions die
first in every checkpoint, and whether carriers die together or sequentially; (4) whether
the state converges toward a common attractor (dispersion and mean-state drift shrinking)
or becomes instance-dependent noise; (5) whether the delay-30 passing and failing groups
(§8k) differ in `D_info`.

**Interpretation.** Descriptive mapping of the failure boundary; no intervention is run
in this section, and claims are bounded to the measured quantities. The ladder spacing
bounds the temporal resolution of "abrupt vs gradual" (a drop could hide between ladder
points); the exact per-checkpoint transition is located only to the ladder grid.

> **Correction (recorded before reporting the affected columns).** The first pass of the
> per-dimension analysis computed point-biserial correlations with a variance guard of
> 1e-12. Five of the eight state dimensions are tanh-saturated (across-instance std
> <= ~5e-4 of floating-point noise), and correlations computed on that noise produced
> spurious |r| up to 1.0 — which would have made every dimension look like a code carrier,
> contradicting §7's 3-of-8 finding. Direct inspection confirmed the genuine carriers are
> the 3 sign-flipping dims (class means +-0.50, std ~0.5), with the other 5 saturated and
> constant. The correlation is now undefined (0.0) below a variance floor of 0.01 — three
> orders of magnitude below the code dims and two above the saturated noise — and carrier
> loss is tracked only for dims that were carriers at the reference delay. The decode,
> behaviour, dispersion and drift results are unaffected (the delay-30 rows reproduce §8k
> exactly, max |diff| = 0.000 on both measures). The affected columns were re-measured
> before any conclusion was drawn from them.

### §8o results (`results/delay_transition/`)

Consistency: the delay-30 rows reproduce the published §8k evaluation exactly
(max |diff| = 0.000 on behaviour and decode over all 23 checkpoints). The per-dimension
columns are from the corrected (post-variance-floor) re-run.

**1. The behavioural boundary is a band, not a point.** `D_behav` distribution: 22 (6
checkpoints), 26 (2), 30 (4), 40 (8), 50 (3). The 12 §8k delay-30 failures all fail by 30
— six already at 22, two at 26 — while the 11 passers hold to 30 and fail at 40 (8) or 50
(3). The bimodality §8k saw at delay 30 is now located in delay space: an early-failing
group (22–30) and a late-failing group (40–50).

**2. The information boundary is later, tighter, and independent of the behavioural
groups.** Every checkpoint holds decode >= 0.95 through delay 40; `D_info` is 50 (12
checkpoints) or 65 (11), and the cross-tab with delay-30 behaviour is mixed (D_info=50:
7 fail / 5 pass; D_info=65: 5 fail / 6 pass). When the information dies is not determined
by when utilisation dies. Behaviour fails before information in 22/23 (in one checkpoint
both fall at 50).

**3. Abruptness.** At the first sub-threshold ladder point, 11 checkpoints are already
<= 0.65 and 12 sit at 0.68–0.88. With 15-step spacing across 50–65, the transition
completes within one ladder interval for all 23 (from >= 0.95 to <= 0.88); no checkpoint
declines gradually across multiple intervals. The shape *inside* an interval is not
resolved by this ladder.

**4. Which dimensions die first: all of them, together.** The corrected per-dimension
analysis shows the code is written on a per-checkpoint subset of 3–8 dimensions (seed 0:
dims {3,5,7} — exactly §7's 3-of-8; seed 3: 6 dims; population range 3–8, mean ~5.4).
Carrier loss is **collective**: all 123 (checkpoint, carrier-dimension) pairs reach the
loss criterion by delay 65 (most at 50 or 65), within one ladder interval of their
checkpoint's `D_info`, with no consistent dimension ordering. The information does not
die dimension-by-dimension; the whole code loses amplitude together.

**5. The state converges to a common attractor.** Across-instance dispersion falls
0.331 (delay 3) -> 0.030 (30) -> 0.00015 (100) — a >2000x collapse — while the mean state
norm pins at ~2.22 from delay 30 onward (tanh saturation) and the mean state drifts only
~0.19. Measured in absolute units (original seed 1): the class-mean gap along the
strongest code dimension falls 1.014 (delay 3) -> 0.118 (30) -> 0.032 (50) -> 0.012 (65)
-> 0.0014 (100) — a ~730x amplitude collapse at a *fixed* state norm.

**6. Delay-100 failure is amplitude annealing, not code deletion.** The scale-free code
survives the whole ladder — for seed 1, the strongest dimension still orders the classes
in-sample (|r| = 1.0) at every delay including 100 — but its absolute amplitude drops
~700x below the trained operating point, so a fixed-resolution held-out decoder reads
chance (decode 0.42–0.68 at delay 100, chance-adjusted baseline ~0.52). Combined with (5):
the answer is neither moved away from the readout nor selectively erased; the entire state
contracts onto the SSM's saturated attractor and the answer-specific component anneals
within it. (The decay's functional form — exponential or otherwise — is *proposed*, not
fitted; no decay curve has been estimated against alternatives.)

**7. Three failure boundaries, separately scheduled in delay space.** Behavioural
(22–50, bimodal: 22–30 vs 40–50), functional (§8k: 11/23 still using the state at 30),
informational (functional carrier variance lost at 40–65; decode threshold crossed at
50/65). Behaviour precedes information in 22/23, and the informational boundary does not
align with the behavioural groups. Delay-100 is therefore not an earlier instance of the
delay-30 failure: it is a distinct mechanism (attractor contraction), reached after the
delay-30 disconnection (boundary displacement) has already occurred in 12/23.

**Bounded claims.** The ladder spacing bounds (3) and the exact `D` values to ±one
interval. The |r| = 1.0 statement is the per-checkpoint maximum over 8 dimensions
(selection) on 200 instances — evidence that an ordering survives, not a population
estimate. No functional (substitution) measurement was taken at the newly located
transition points, per the pre-registration. Evaluation-only throughout; no parameters
were changed.

**Three concepts, kept separate from here on** (all measured per checkpoint x delay):

- **state magnitude** — mean ||s|| at the response step. A stable magnitude is *not*
  evidence of retained information (§8o: norm pinned at 2.22 while information annealed);
- **state dispersion** — across-instance std of the response-step state. How much states
  differ between instances;
- **task-relevant separation** — the class-mean gap along code dimensions, and held-out
  decodability. Scale-free correlation on a dimension whose variance has vanished is *not*
  evidence of usable information (the §8o variance-floor correction exists for exactly
  this reason).

The delay-100 failure is the conjunction: magnitude stable, dispersion -> 0, task
separation -> 0.

## 8p. Transition-point causal confirmation at delays 40, 50, 65 (pre-registered)

**Question.** §8o located three boundaries in delay space — behavioural 22–50, functional
(<= 30 for 12/23, from §8k's delay-30 battery), informational 50/65 — but causal
utilisation has only ever been measured at 3/10/30/100 (§8k). This experiment runs the
full causal substitution battery at the newly located transition points, to establish
exactly where functional utilisation disappears relative to behavioural failure and to
informational loss.

**Design (evaluation only).** All 23 successful checkpoints x delays **[3, 40, 50, 65]**.
Delay 3 is an in-run anchor that must reproduce §8k's delay-3 values exactly (same
determinism check §8o used at delay 30). Delays 40/50/65 are **new causal measurements**,
not re-readings of §8o: §8o's rows at those delays carried no substitution battery, and
nothing here is inferred from them. Per checkpoint x delay: behavioural accuracy and the
state-reset control; held-out ridge decoding of the answer at the response step; and the
full substitution battery (own, same-answer, different-answer-follows, zero, shuffled).
No training, no parameter change (`scripts/run_transition_functional.py`,
`results/transition_functional/`).

**Groups (fixed from the published §8o table, not post-hoc):**

- behavioural: early-failing (`D_behav` <= 30; 12 checkpoints) vs late-failing
  (`D_behav` >= 40; 11);
- informational: `D_info` = 50 (12) vs 65 (11).

**Functional criterion (as everywhere):** different-answer follows >= 0.9, same-answer
>= 0.9, zero and shuffled <= 0.65. Never accuracy alone.

**Pre-registered interpretation rules:**

- **R1 (coincidence):** if every behaviourally-failed checkpoint also fails functionally
  at delay 40, behavioural and functional failure are the same event at this resolution,
  and the hierarchy collapses to behaviour+function before information.
- **R2 (dissociation):** if any behaviourally-failed checkpoint passes the full functional
  battery at 40, the state->answer causal map survives behavioural failure — a fourth
  phenotype (information present, causally readable from donated state, behaviour absent);
  the checkpoints are listed individually, and the follow-up question becomes why
  own-states fail while donated states read out correctly.
- **R3 (at the informational boundary):** for each `D_info` group, compare functional
  status at 50 and 65, interpreted **jointly with decode**: as the answer amplitude
  anneals (§8o), donor and own states converge, so the battery's same/diff contrast
  mechanically degrades. Where decode < 0.95, follows-at-chance is *contrast loss*, not
  evidence of disconnection. The clean functional test is therefore at delay 40, where
  decode >= 0.95 for all 23 and contrast is intact.
- **R4 (battery validity):** a functional claim at a given delay additionally requires
  zero and shuffled <= 0.65 at that delay.

**Bounded claims.** Checkpoint-level results are primary; group proportions secondary. No
parameters change and no maintenance intervention runs here — the constrained-update
experiment (§8q) follows only after these boundaries are causally confirmed.

### §8p results (`results/transition_functional/`)

Anchor: delay 3 reproduces §8k exactly — 23/23 behavioural, informational and functional
(follows/same 1.000, zero 0.496, shuffled 0.531).

**R1 confirmed (coincidence branch); R2 empty.** At delay 40: behaviour 3/23, functional
3/23, informational 23/23 (decode = 1.000 in every checkpoint). The functional-pass set
{seeds 1, 8, 15} equals the behavioural-pass set **exactly, checkpoint-for-checkpoint**
(agreement 23/23). All 12 early-failing (`D_behav` ≤ 30) checkpoints fail the full battery
at 40 (follows 0.48–0.52) with decode 1.000 and the zero/shuffled controls nominal
(max 0.52/0.59 ≤ 0.65, so the battery is valid — R4 holds). Functional utilisation and
behavioural retention are therefore not merely adjacent in delay space; they are the same
event per checkpoint. In the 20 functionally-dead checkpoints at delay 40, same-answer
substitution *also* fails (0.48–0.52): donating a state that carries the correct answer
does not preserve the response — the functional disconnection measured at delay 30 (§8k)
is the same disconnection at 40, now with the information still demonstrably present.

**R3 refinement (recorded before any dependent use).** The pre-registered contrast-loss
guard fired on decode < 0.95; at delay 50 both `D_info` groups have a contrast-free battery
for a different, sharper reason: donated states have annealed into near-coincidence
(donor dispersion 0.003–0.010), so same, diff, zero and shuffled substitutions all produce
the same chance response (~0.49–0.51) regardless of what the donor carries. In the
`D_info`=65 group this happens with decode **still 1.000** — the correct trigger for
"battery uninformative" is donor-state dispersion annealing, not the decode flag. The
clean causal statement is therefore at delay 40 (contrast intact, controls nominal); at
50/65 the battery measures the annihilation of donor contrast, not disconnection. The
per-checkpoint table (summary.json) preserves both decode and dispersion at 50/65.

**Updated boundary picture.** Functional utilisation ≤ 30 for 12/23 and dead by the first
ladder point past each checkpoint's `D_behav` for the rest — behaviour and function
co-fail across the whole population, at each checkpoint's own behavioural boundary
(22–50), while information survives to 50/65. The hierarchy is now: **behaviour =
function (one event, 22–50) < information (50/65)** — and §8o's mechanism explains the
third boundary (amplitude annealing), while the first (behaviour/function) is the
readout-detachment failure class of §8k–§8n. No checkpoint shows information present +
function absent + behaviour present at 40 — the R2 phenotype does not exist in this
population; donating the right answer never rescues a failed response past 40.

**Bounded claims.** Causal claims are bounded to the substitution battery as implemented
(§7 machinery); delay-50/65 functional status is reported as unmeasurable-by-battery, not
as "no utilisation". n=1 paired observation per (checkpoint, delay); no new controls
beyond the pre-registered five-arm battery.

## 8q. Long-delay maintenance under state-budget-constrained encoder updates (pre-registered)

**Question.** §8n located the destabiliser: a single *unconstrained* encoder step on the
delayed objective destroys the delay-30 rescue. The user-proposed hypothesis: constraining
encoder updates by their effect on the delayed representation (rather than balancing event
counts, §8m) should allow maintenance *under genuine adaptation*. This may also produce the
first architectural-relevant improvement of the programme, if it works without a new memory
module — per the standing decision, no new module, no added state dimensionality; the D800
frozen reference config is untouched.

**Design.** The same 12 captured checkpoints and the *identical* delayed exposure as §8n
`delayed_only`: 100 delayed episodes at delay 30, continuing the saved RNG stream, readout
frozen (head/C/D deltas must be exactly 0.000). Three conditions, one variable at a time:

1. **unconstrained** — §8n `delayed_only`, already measured (0/12, collapse at episode 1).
   Used as the reference arm; *not re-run* (identical seeds, data stream, and optimiser by
   construction).
2. **`state_budget`** (constrained, readout/task-aware) — a per-step backtracking line
   search. After each proposed encoder update, the step is scaled by α ∈ {1, 1/2, …, 1/128,
   0} and accepted at the largest α whose **cumulative displacement** of the response-step
   delayed states on a fixed 16-instance held-out probe batch stays within budget ε of the
   *captured* model's states on the same batch: α = 0 (full rejection) is admissible.
   *Primary budget ε = 0.2* (~10% of the response-step state norm ≈ 2.22, and of the same
   order as the mean-state drifts §8o measured across the 3→30 transition, 0.19–0.34);
   *secondary ε = 0.05* probes a stricter regime. Both are pre-registered here, before any
   run.
3. **`matched_raw`** (magnitude-matched control, no state awareness) — at each step, the
   raw delayed-gradient update is scaled to exactly the per-step accepted-update norm that
   `state_budget` accepted at the same episode (paired runs, same seeds/data/optimiser). Any
   outcome difference between arms 2 and 3 is then attributable to *direction*
   (state-awareness), not to step size.

**Per-step recorded:** accepted α, accepted/update-norm ratio, cumulative displacement,
delayed margin at the response step (all on the probe batch). Probe schedule as §8n
(episodes 1–5, then every 10); pre/post full state summaries; final substitution battery;
full pre-registered rubric (`cr.success_rubric`); parameter deltas; post geometry.

**Success criterion (fixed):** full rubric at post AND delayed accuracy = 1.000 AND GNN
parameter delta ≥ 0.05 (genuine adaptation, not de-facto freezing). Freezing is scored
separately: if > 90% of steps are rejected (α = 0) and GNN delta < 0.05, the outcome is
labelled **no-admissible-direction** (the rescue point is a constrained optimum of the
coupled system), not success.

**Interpretation rules (fixed before running):**

- **12/12 success with real parameter movement** → maintenance under adaptation is
  achievable via state-budget-constrained encoder updates; the `matched_raw` outcome
  attributes it to state-awareness vs step size.
- **`matched_raw` also passes** → the effect is step size (dose), not state-awareness;
  the §8n failure is then simply about update magnitude, and the constrained update adds
  nothing the same norm would not.
- **`state_budget` passes, `matched_raw` fails** → state-aware direction is the operative
  factor: the delayed gradient's *large* components are precisely the code-rewriting ones.
- **0/12 both** → the rescue is not maintainable under any tested encoder update at any
  tested budget; strengthens "readout compatibility is a hairline constraint".
- Partial outcomes: rank-separated as usual, reported per checkpoint.

**Bounded claims.** The constraint is measured on a 16-instance probe batch and at the
response step only — it constrains what the probe can see, and states outside the probe's
support are unconstrained. ε is norm-relative on the *displacement* vector, not
direction-wise; a rotation within budget is admissible by design (boundary translations are
the survivable failure mode, §8n immediate_only). ε = 0.2/0.05 are operational choices,
justified above, not biological claims. The unconstrained reference is §8n's arm, not a
fresh run — if any §8q infrastructure difference emerged it would invalidate that reuse and
be recorded before reporting.

(`scripts/run_state_budget.py`, results in `results/delay30_maintenance/`.)

> **Definitional clarification (recorded before implementation and before any result).**
> The "cumulative displacement" of the state-budget constraint is the mean over the 16
> probe instances of the per-instance L2 distance ||s_i − s_cap,i||_2 between the current
> and captured response-step states, i.e. a per-instance norm (state norm ≈ 2.22 → ε = 0.2
> is ~10% per instance). The α ladder is {1, 1/2, 1/4, …, 1/128, 0}; α = 0 means the step
> is fully rejected.
>
> **Procedural amendment (recorded BEFORE any §8q run completed; all initial shard runs
> died of system memory exhaustion with no completed arm, so no carried-over results
> exist).** The descending α scan costs 4–6 full probe captures per step (~36 s/step,
> >30 h over the grid) and the running machine cannot support the parallelism that pace
> requires. The accepted-α procedure is amended to a **crossing-estimate scan** with the
> same accepted-α definition (largest ladder α with displacement ≤ ε): (1) measure
> disp(α=1), accept 1 if within ε; (2) else measure disp(α=1/128) — if beyond ε, no rung
> qualifies and α=0; (3) else linearly interpolate the crossing in log2-α between the two
> measured points, take the largest ladder rung at or below the estimate, verify it by
> capture, then attempt successive rungs upward while they qualify. Under monotone
> disp(α) this returns exactly the registered scan's α; if disp(α) were non-monotone
> between adjacent rungs the two procedures can differ — a **calibration harness** runs
> both scans on identical step streams for the first 15 episodes of two checkpoints and
> the amendment is adopted only if the accepted α agree rung-for-rung on every step and
> every displacement choice; otherwise the full descending scan is used despite the cost
> (~2–3 captures/step vs 4–6). Concurrency capped at 3 processes.
>
> **Calibration outcome (recorded).** The gate FAILED: rung-for-rung agreement on only 1
> of 4 (checkpoint, ε) pairs. The crossings sit exactly on the budget boundary, and the
> graph-pooling ops are non-deterministic across processes, so borderline acceptances
> flip — precisely the regime the amendment excluded from adoption. Per the recorded
> fallback rule the **full descending ladder scan is the operative procedure** for all
> §8q runs (`fast = False` everywhere). A post-hoc timing profile on the cleaned machine
> measured a probe capture at ~0.1 s (the 7.2 s figure that motivated the amendment was
> an artifact of the memory-starved machine), so the registered scan is affordable after
> all. The rejection of the amended scan is *methodological*, not a result: the
> accepted-α definition is unchanged and the constraint semantics are exactly as
> pre-registered in the definitional clarification.

## 9. Reproduction

Every number above is regenerable: each experiment directory contains the exact `config.yaml`
used, per-seed `seed_summary.json`, raw CSVs (`metrics.csv`, `graph_locus.csv`,
`rewired_control.csv`, `retention.csv`, `memory_curve.csv`, …), checkpoints with environment
metadata, and an aggregated `summary.json`. Commands: `python scripts/run_experiment.py
--config <cfg> [--graph <family>]`, `python scripts/run_curriculum_study.py --config
configs/experiments/curriculum_study.yaml`. 117 unit tests: `pytest`.


