# Artificial Learning System (artificial_cns)

A controlled experimental platform for studying how a **hybrid neural architecture** —
a Graph Neural Network (GNN) for structured relational computation combined with a
State-Space Model (SSM) for temporal state and memory — learns, retains, reverses, and
generalizes behaviour.

Everything is built around one stable graph interface (`GraphData`), so any new graph
data source enters as an **adapter**, never a model rewrite.

---

## What we are trying to do

The central research question is not *whether* a hybrid GNN+SSM architecture performs
well, but **why** — and whether its components make experimentally distinguishable
contributions:

1. **Functional specialization** — can the GNN, the SSM, and the learned weights be shown
   to carry *different*, separately ablatable responsibilities (relational computation,
   temporal bridging, durable associations)?
2. **Reliable composition** — can these capabilities be learned together reliably across
   seeds, curricula, and task variations?
3. **Generalizable cognitive substrate** (long-term) — can the same design principles
   support working memory of computed answers, evidence integration, and rule revision?

The methodology matters as much as the results: every headline claim in this project is
backed by controls designed to kill specific alternative explanations (shortcut detection,
degree statistics, instance memorization, information leakage at response time).

---

## Architecture

```
graph input ──> GNN encoder ──> node embeddings ──> pooling ──> z_t
                                                                  │
                                                                  v
                        readout head <── SSM state s_t = f(A·s_{t-1} + B·z_t)
```

- **GNN** — pure-PyTorch message passing (no PyG dependency), configurable
  mean/sum/max aggregation, residual connections, LayerNorm, node-mask silencing.
- **SSM** — discrete-time linear state space model with spectral-radius-bounded `A`;
  exposes `initial_state / step / forward / reset_state` for state perturbation studies.
- **Readout** — mean / max / attention / selected-node pooling + prediction head.
- **Ablation matrix** — M1 (no GNN, no SSM), M2 (GNN only), M3 (SSM only), M4 (GNN+SSM),
  M5 (Transformer temporal), M6 (per-node MLP + SSM = message passing disabled).

All components communicate exclusively through:

```python
GraphData(node_features, edge_index, edge_features=None, node_mask=None, batch=None, metadata={})
```

---

## Two benchmarks

### 1. Cue-learning task + ARRG protocol (spec §14–§17)

A configurable cue-learning environment (cues written into a graph feature channel,
configurable delays, stochastic rewards, mapping reversal) with automated evaluation of:

- **A**cquisition — performance before vs after training
- **R**etention — accuracy after blank/distractor delays (0 → 500 steps)
- **R**eversal — relearning speed after the cue→action mapping flips
- **G**eneralization — transfer to held-out cues sharing an explicit similarity structure

### 2. Graph-dependent classification (topology causally necessary)

A benchmark where **both classes come from the same connected-graph family** and only the
marked node pair differs (adjacent vs distance ≥ 3). No graph-level statistic — component
count, degree, density — can reveal the label; only marker relations can. Delayed variants
require the system to compute the relational answer and **preserve it in state** after the
markers disappear. Controls include:

- degree-preserving edge swaps (relabeled vs original labels) — separates relational
  structure from degree statistics
- state-reset / partial-reset / noise / truncation probes (temporal)
- storage-locus analysis (state × graph × response-regime)
- computed-answer retention curves and a delayed counterfactual probe
  (retain vs update-on-new-evidence vs irrelevant-new-information)

---

## Key findings (all numbers verified against `results/*/summary.json`; see
[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the full fact-checked account)

| Finding | Evidence |
|---|---|
| Learning survives 10× graph scaling | acquisition criterion ~98/99/100 trials at 100/500/1000 nodes; retention@500 = 1.0 |
| Two separable memory mechanisms | mapping lives in **weights** (durable, delay-500 retention 1.0); active-cue trace lives in **SSM state** (half-life ≈ 2 steps, full reset → chance) |
| The cue task is graph-independent | 15 graph perturbations (up to 75% edge/node removal) caused **zero** behavioural change — motivating the graph-dependent benchmark |
| Message passing is necessary for relational answers | no-message-passing ablations at chance; degree-preserving relabeled control = 1.000; unseen graph size = 1.000 |
| Identity features cause shortcut memorization | 400→10000 instances + weight decay never induced the rule; bare marker features did (test 1.000, unseen size 1.000) |
| The SSM carries computed answers across delays — observed reliably | interventions meet the pre-registered *observed* criterion (≥ 90% success proportion in the sample; population rate not yet tightly bounded): interleaved-800 and GNN-frozen staged training both reach **3/3 seeds** with state reset → chance |
| The retained state is flexible, not a rigid imprint | counterfactual probe: successful seeds update the answer on new evidence (incl. answer-flipping) = 1.000, ignore irrelevant input = 1.000 |
| Working memory generalizes beyond the trained delay | successful seeds retain the computed answer perfectly to delays 10–30 (1–10× the trained delay 3) without training at those delays |
| Mean aggregation oversquashes relational signals | sum aggregation fixed a hard propagation failure (0.58 → 0.81) |
| The state carries the *answer*, not the evidence | held-out linear probe decodes the answer (1.000) at every trial point; instance features are absent from the state (own-class-centroid distance ≈ 1e-4 vs 0.38 cross-class); the code is sparse: exactly 3 of 8 dims flip sign between classes |
| The state's content *causally* controls the prediction | state substitution: donors from other trials at the same computation point — same-answer substitution stays correct (1.000), different-answer donations are **followed** (1.000), zero state → chance, all on blank observations during the delay |
| Decodability ≠ use: failures dissociate cleanly | 2 failed seeds carry a perfectly decodable answer in state yet behave at chance *and* fail every substitution control — answer content present but causally disconnected from the readout; failure modes split into not-computed / decays / present-but-unused |
| The missing piece is learnable readout *coupling* (H5) | readout-only training on labelled state–target pairs can establish the missing functional connection **with frozen state-generation and unchanged state geometry**; scrambled-label control changed nothing (H5); later corrected: the coupled solution is transient (see next row) |
| The coupled readout is a transient, not an attractor (H6/H7 diagnostics) | delayed accuracy oscillates 1.000↔chance with a ~20-episode period; immediate-vs-delayed gradient cosine flips in anti-phase (±0.4) with the cycle; final-probe "rescue vs no-rescue" differences between sources were phase sampling (§8d) |
| Acquire + retain is achievable with schedule-only changes (H7) | capture-and-hold (freeze at first delayed ≥ 0.9 probe, then joint training with frozen readout) meets **all** pre-registered criteria on one source — immediate 1.000, delayed 1.000, substitution follows 1.000, state representation preserved; the difficult source's captured solution is still destroyed by joint training (open, localised to representation-pathway gradient conflict) |

**Scope caveats:** the state-reset result establishes *necessity of the state*, not a
general working-memory faculty; results are 2–4 seeds per condition on one task family;
the reliability interventions (D800 interleaved and GNN-frozen staged training) both reach
**3/3 seeds** and meet the pre-registered *observed* criterion (success proportion ≥ 90% in
the sample) — D800 preserves both capabilities, while GNN-freezing sacrifices the immediate
rule (readout forgetting). 3/3 does not tightly bound the *population* success rate
(P(3/3 | p = 0.7) ≈ 0.34); additional seeds are an explicit open item.

---

## Repository layout

```
artificial_cns/
├── configs/            # base + toy + experiment YAML configs (incl. frozen study baseline)
├── src/artificial_system/
│   ├── data/           # GraphData schema, generators, batching, adapters, graph tasks
│   ├── models/         # GNN, SSM, GraphTemporalModel, readouts, ablation builders
│   ├── environment/    # cue-learning + graph-classification environments
│   ├── training/       # episodic BPTT trainer, losses, curriculum
│   ├── evaluation/     # ARRG, memory curves, perturbations, M7 controls, counterfactual
│   ├── experiments/    # reproducible runner, ablations, curriculum study
│   └── utils/          # seeding, logging, checkpointing
├── scripts/            # run_experiment.py, train.py, evaluate.py, run_curriculum_study.py
├── tests/              # 117 unit + pipeline tests
├── results/            # experiment outputs (gitignored; regenerable from configs)
└── TECHNICAL_REPORT.md # full fact-checked report of all experiments
```

## Install & quickstart

```bash
pip install -e ".[dev]"
pytest                                        # 117 tests

# full pipeline: load graph -> validate -> train -> checkpoint -> ARRG eval
python scripts/run_experiment.py --config configs/toy_small.yaml --graph random --seeds 0

# graph-dependent benchmark (immediate / delayed)
python scripts/run_experiment.py --config configs/experiments/graph_adjacency.yaml
python scripts/run_experiment.py --config configs/experiments/graph_adjacency_delayed.yaml

# curriculum-scheduling study (4 budget-matched conditions)
python scripts/run_curriculum_study.py --config configs/experiments/curriculum_study.yaml

# training / evaluation only
python scripts/train.py --config configs/base.yaml --graph small_world --seeds 0
python scripts/evaluate.py --config configs/base.yaml --graph small_world \
    --checkpoint results/base/seed_000/checkpoint.pt
```

`--graph` accepts a synthetic family (`random`, `regular`, `small_world`, `scale_free`,
`modular`) or an `.npz` export (`node_features`, `edge_index`, optional `edge_features`,
`node_mask`). Production graph data plugs in via a `GraphAdapter` — nothing upstream changes.

## Reproducibility

Every experiment directory contains the exact `config.yaml` used, per-seed summaries,
raw CSVs, and checkpoints with environment metadata (Python/torch versions, git commit).
Every figure and number in `TECHNICAL_REPORT.md` is regenerable from the recorded configs
and seeds.

## Status

- Implemented, tested, and run: cue-task ARRG, scaling (100→1000), memory curves,
  temporal/graph perturbation batteries, storage-locus analysis, graph-dependent
  benchmark (immediate + delayed), ablation matrices, the **curriculum-scheduling
  study** (4 budget-matched conditions × 3 seeds: interleaved 2/3, staged 1/3,
  delayed-only 0/3, immediate-only 0/3 — with failure-mode taxonomy from
  intact-vs-reset trajectories, counterfactual probes, and computed-answer retention),
  and two **bridge-reliability interventions** — interleaved-800 and GNN-frozen staged
  training — both reaching **3/3 seeds** against the pre-registered *observed* criterion
  (≥ 90% success proportion in the sample; population rate not yet tightly bounded).
- Reproducibility verified: `scripts/verify_reproducibility.py` regenerates all published
  metrics from committed checkpoints + configs (12/12 exact matches).
- Open items: targeted interventions on the answer-present-but-unused failure phenotype;
  additional D800 seeds for a tighter population bound; delay scaling (see
  `TECHNICAL_REPORT.md` §8).
- ~~Unseen-size anomaly~~ resolved: was an evaluation-protocol defect (generic blank not
  threaded into the generalization probe). Corrected, all successful seeds generalize to
  n=200 at 0.995–1.000, and the controlled size sweep (100→200, degree-matched) is flat —
  see `scripts/run_size_sweep.py` outputs in `results/*/size_sweep/`.
- State-content analysis (decodability / answer-vs-feature / substitution) complete on
  18 condition-seeds: `scripts/run_state_analysis.py`, results in `results/*/state_content/`.
- H5 readout-coupling rescue run (3 arms × 2 phenotype-(iii) checkpoints, pre-registered
  rubric): see `scripts/run_coupling_rescue.py`, `results/coupling_rescue/`;
  report §8b.
