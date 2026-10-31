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

**D_int s0 remains unresolved — narrowed further.** Its coupled readout *can* be captured
(episode 60, full up-phase behaviour) but is destroyed by any subsequent joint-training
regime tested so far — including with the readout completely frozen. Combined with §8d,
the D s0 failure is now localised to **conflict between the two objectives' gradients
through the shared representation pathway** (encoder/pooling/state-generation), not the
readout path and not the state content. Next registered diagnostic: per-group gradient
cosine during stage 4 on this source.

## 9. Reproduction

Every number above is regenerable: each experiment directory contains the exact `config.yaml`
used, per-seed `seed_summary.json`, raw CSVs (`metrics.csv`, `graph_locus.csv`,
`rewired_control.csv`, `retention.csv`, `memory_curve.csv`, …), checkpoints with environment
metadata, and an aggregated `summary.json`. Commands: `python scripts/run_experiment.py
--config <cfg> [--graph <family>]`, `python scripts/run_curriculum_study.py --config
configs/experiments/curriculum_study.yaml`. 117 unit tests: `pytest`.


