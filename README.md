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
| The SSM carries computed answers across delays — with the right curriculum | immediate-first curriculum: 3/4 seeds reach 1.000 on the delayed task with state reset → chance (0.480); joint discovery without curriculum: 0/2 seeds |
| Mean aggregation oversquashes relational signals | sum aggregation fixed a hard propagation failure (0.58 → 0.81) |

**Scope caveats:** the state-reset result establishes *necessity of the state*, not a
general working-memory faculty; the working-memory result holds for 3/4 seeds under one
curriculum on one task family; the curriculum-scheduling study (4 budget-matched
conditions × seeds) is implemented and smoke-tested but **not yet run at scale**.

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
  benchmark (immediate + delayed + curriculum), ablation matrices.
- Implemented, not yet run at scale: the curriculum-scheduling study and the
  counterfactual probe (next milestone: ≥ 90% seed success rate under the chosen
  curriculum, verified against the pre-registered criteria in the study config).
