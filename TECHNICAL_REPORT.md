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

## 9. Reproduction

Every number above is regenerable: each experiment directory contains the exact `config.yaml`
used, per-seed `seed_summary.json`, raw CSVs (`metrics.csv`, `graph_locus.csv`,
`rewired_control.csv`, `retention.csv`, `memory_curve.csv`, …), checkpoints with environment
metadata, and an aggregated `summary.json`. Commands: `python scripts/run_experiment.py
--config <cfg> [--graph <family>]`, `python scripts/run_curriculum_study.py --config
configs/experiments/curriculum_study.yaml`. 117 unit tests: `pytest`.


