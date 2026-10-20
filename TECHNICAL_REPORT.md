# Technical Report — GNN + SSM Artificial Learning System

**Status:** Working experimental platform, Milestones 1–5 complete (spec §27), graph-dependent
benchmark complete, curriculum-scheduling study *implemented but not yet run*.
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

\* seed 1's unseen-size probe returned 0.505 — the only successful seed where the size
generalization probe did not track the seen-size accuracy; noted as an anomaly to re-check
(probe uses the same model; the discrepancy is unexplained and should be re-run before any claim).

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

## 5. What is implemented but NOT yet run

- **Curriculum-scheduling study** (Phase B–D): 4 budget-matched conditions (A immediate-only,
  B delayed-only, C staged, D interleaved; 400 optimizer episodes each; intact-vs-reset probes
  every 100 episodes; computed-answer retention curve; **delayed counterfactual probe**
  (retain / update-on-new-evidence / irrelevant-new-info); success-rate aggregation against a
  pre-registered 0.9 threshold). Code: `run_curriculum_study`, `run_curriculum_condition`,
  `probe_delayed_intact_vs_reset`, `run_delay_retention`, `run_counterfactual_probe`;
  config: `configs/experiments/curriculum_study.yaml`; CLI: `scripts/run_curriculum_study.py`.
  Covered by a passing end-to-end smoke test (`test_curriculum_study.py`). **Not executed at
  scale** (the launch was blocked; est. ~2.5–3 h for 4 conditions × 3 seeds).
- The counterfactual and retention probes have so far only been exercised on tiny smoke-test
  models — treat their outputs as unvalidated until the study runs.

---

## 6. Next steps (in order)

1. **Run the curriculum study** (`python scripts/run_curriculum_study.py --config
   configs/experiments/curriculum_study.yaml`), 3 seeds first, extend to ≥ 10 seeds if the
   success rate is promising. Pre-registered criteria: ≥ 90% of seeds reach delayed accuracy
   ≥ 0.9; successful seeds show the state-reset drop; unseen generalization holds; the
   degree-preserving control holds; C/D ≥ A/B under matched budgets.
2. **Investigate seed 0** (curriculum failure): compare phase-2 loss trajectories and state
   norms of successful vs failed seeds; test whether more phase-2 episodes or an intermediate
   delay (0→1→2→3) rescues it. This distinguishes optimization sensitivity from curriculum
   dependence (your §4 hypotheses).
3. **Counterfactual probe at scale** — distinguish rigid answer retention from flexible,
   evidence-responsive state (implemented, unrun).
4. **State-content analysis**: partial resets that preserve vs destroy the answer decodable
   from state (the reset machinery already exists; linear probes on the 8-dim state are the
   natural next tool).
5. Only after reliability: delay scaling (3→10→30→100), then richer relational relations
   (paths, multi-hop), then the Level-3 agenda (multiple items, evidence integration).

## 7. Reproduction

Every number above is regenerable: each experiment directory contains the exact `config.yaml`
used, per-seed `seed_summary.json`, raw CSVs (`metrics.csv`, `graph_locus.csv`,
`rewired_control.csv`, `retention.csv`, `memory_curve.csv`, …), checkpoints with environment
metadata, and an aggregated `summary.json`. Commands: `python scripts/run_experiment.py
--config <cfg> [--graph <family>]`, `python scripts/run_curriculum_study.py --config
configs/experiments/curriculum_study.yaml`. 117 unit tests: `pytest`.
