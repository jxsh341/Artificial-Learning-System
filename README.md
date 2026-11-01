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
| The SSM carries computed answers across delays — observed reliably | interventions meet the pre-registered *observed* criterion (≥ 90% success proportion in the sample): interleaved-800 and GNN-frozen staged training both reach **3/3 seeds** with state reset → chance; the interleaved-800 (D800) rate is now bounded by a 20-seed population run (next row) |
| The retained state is flexible, not a rigid imprint | counterfactual probe: successful seeds update the answer on new evidence (incl. answer-flipping) = 1.000, ignore irrelevant input = 1.000 |
| Working memory generalizes beyond the trained delay | successful seeds retain the computed answer perfectly to delays 10–30 (1–10× the trained delay 3) without training at those delays |
| Mean aggregation oversquashes relational signals | sum aggregation fixed a hard propagation failure (0.58 → 0.81) |
| The state carries the *answer*, not the evidence | held-out linear probe decodes the answer (1.000) at every trial point; instance features are absent from the state (own-class-centroid distance ≈ 1e-4 vs 0.38 cross-class); the code is sparse: exactly 3 of 8 dims flip sign between classes |
| The state's content *causally* controls the prediction | state substitution: donors from other trials at the same computation point — same-answer substitution stays correct (1.000), different-answer donations are **followed** (1.000), zero state → chance, all on blank observations during the delay |
| Decodability ≠ use: failures dissociate cleanly | 2 failed seeds carry a perfectly decodable answer in state yet behave at chance *and* fail every substitution control — answer content present but causally disconnected from the readout; failure modes split into not-computed / decays / present-but-unused |
| The missing piece is learnable readout *coupling* (H5) | readout-only training on labelled state–target pairs can establish the missing functional connection **with frozen state-generation and unchanged state geometry**; scrambled-label control changed nothing (H5); later corrected: the coupled solution is transient (see next row) |
| The coupled readout is a transient, not an attractor (H6/H7 diagnostics) | delayed accuracy oscillates 1.000↔chance with a ~20-episode period; immediate-vs-delayed gradient cosine flips in anti-phase (±0.4) with the cycle; final-probe "rescue vs no-rescue" differences between sources were phase sampling (§8d) |
| Acquire + retain is achievable with schedule-only changes (H7/H7b) | capture-and-hold with **per-episode alternating restore** (freeze at first delayed ≥ 0.9 probe; readout static while the representation adapts) meets **all** pre-registered criteria on **both** phenotype-iii sources — immediate 1.000, delayed 1.000, substitution follows 1.000, state representation preserved (norm ratio 0.87/1.07). The blocked-restore failure of H7 was a schedule artifact (§8f), corrected and preserved in the record |
| Compatibility of shared-representation objectives | under alternating joint adaptation the two objectives co-exist (immediate gradients vanish once satisfied); blocked phases are what destroy co-acquired solutions — compatibility is achievable without architecture change |
| Capture-and-hold generalises to the class it targets, and only to it (§8i) | across **all nine** failing checkpoints in the frozen study: "present-but-unused" checkpoints (answer decodes at 1.000, behaviour at chance) capture and pass the full rubric **3/3** (95% CI 0.292–1.000); "never-computed" and "computed-then-decayed" checkpoints (answer absent at response) capture **0/6** and 5/6 stay at chance. Applicability is predictable *in advance* from a state probe, not from the schedule. One answer-absent checkpoint passed by a different route: with the readout exactly frozen, encoder adaptation created the state representation (decode 0.53 → 1.000) — flagged for replication |
| Retention separates into three timescales (§8k) | evaluation-only sweep over 23 checkpoints, no retraining: at delay 10 (3.3× trained) information, functional and behavioural retention all hold **23/23**; at delay 30 (10×) **information survives 23/23 but functional and behavioural retention only 11/23**, bimodally with no intermediates, and the failures are *confidently* wrong (margin 0.74 vs 0.38, state norm 1.65 vs 1.39) — disconnection, not forgetting; at delay 100 (33×) information itself is gone (decode ≈ 0.53) |
| The delay-30 failure is a displaced decision boundary, not a lost or rotated representation (§8l stage A) | all 23 checkpoints × delays 3/10/30: the probe direction and the effective readout direction are the same in every group (cos 0.975–0.982; failing 0.982 vs passing 0.980, so rotation is rejected). At delay 30 every failing checkpoint has **one class entirely misclassified** (\|boundary_vs_mid\| 0.52–1.75; 8 class-0-wrong, 4 class-1-wrong) with probe–readout agreement at chance (0.48/0.52), while passing checkpoints are aligned (max \|boundary_vs_mid\| 0.45, agreement 1.000). Failing margins and norms are larger, not smaller, so decay is rejected too. At delays 3 and 10 all 23 are aligned (max \|boundary_vs_mid\| 0.047 / 0.161) |
| H7b at delay 30: the readout *can* rescue it, but the joint schedule is not delay-invariant (§8l stage B) | 12 failing checkpoints, delay 30 preserved throughout: **capture fired 12/12** (episodes 5–15) and the readout-only stage alone reached the full rescue signature **12/12** (delayed 1.000, decode 1.000, same 1.000, follows 1.000, zero/shuffled at chance). The pre-registered full-rubric outcome is **0/12**: the joint stage then erases it in 12/12 (delayed → 0.48–0.52, decode → 0.29–0.52) with the readout exactly frozen (deltas 0.000). Measured cause: 1:1 alternating *episodes* deliver 32:1 immediate:delayed supervised events at delay 30 (events per 32-step episode: 32/8/2/1 at delays 0/3/10/30) vs 4:1 at delay 3, where H7b passed 3/3 while moving the encoder *more* (GNN delta 1.29–3.64 vs 0.81–1.35) — the delayed task supplies too little gradient to defend a 31-step-compounded state, not too much encoder drift |
| Two distinct acquisition routes, separated by which side of the interface adapts (§8j) | **capture-and-hold** (H7b): the readout learns utilisation *after* the answer is present — 3/3 phenotype-iii checkpoints. **Encoder-side acquisition** (B2): the readout is frozen and uninformative, and the encoder learns to write the answer into the state — 3/3 data-order replications, decode 0.53 → 1.000, head/C/D deltas exactly 0.0, substitution follows 1.000. Kept as separate hypotheses, not merged |
| D800 is a reliable reference baseline (population study, §8h) | 20 fresh seeds, frozen config: **20/20** delayed ≥ 0.9 and immediate 1.000 (95% CI 0.832–1.000); with the original 3 seeds kept separate, **23/23** (95% CI 0.852–1.000). State reset → chance (mean 0.500) in every seed. Unseen size 22/23 ≥ 0.985; the single exception (seed 19, 0.560) is an encoder scale/degree calibration issue — immediate and delayed fail together, reset stays at chance, and degree-matched n=200 restores 1.000 |

**Scope caveats:** the state-reset result establishes *necessity of the state*, not a
general working-memory faculty; results are 2–4 seeds per condition on one task family;
the reliability interventions (D800 interleaved and GNN-frozen staged training) both reach
**3/3 seeds** and meet the pre-registered *observed* criterion (success proportion ≥ 90% in
the sample) — D800 preserves both capabilities, while GNN-freezing sacrifices the immediate
rule (readout forgetting). 3/3 does not tightly bound the *population* success rate
(P(3/3 | p = 0.7) ≈ 0.34); additional seeds are an explicit open item.

---

## Position statement (current)

> The Artificial Learning System demonstrates separable stages of persistent-state
> learning: information representation, functional utilisation, and capability retention.
> Experiments identified cases where answer information was preserved in the recurrent
> state but disconnected from the output pathway. Readout-only training transiently
> restores functional utilisation; diagnostics then revealed oscillatory coupling rather
> than stable convergence (both tested failing checkpoints acquire the coupled solution,
> which re-appears and decays with a ~20-episode period). A pre-registered
> capture-and-hold intervention with per-episode alternating joint adaptation subsequently
> preserved both immediate and delayed capabilities through continued training — meeting the
> full pre-registered rubric (behaviour, causal substitution battery, preserved state
> representation) — without architectural modification, in **all three** checkpoints whose
> state carries the answer at response time (3/3). Schedule granularity (blocked vs
> alternating phases) was identified as the determinant of whether co-adaptation preserves
> or destroys co-acquired solutions. Across all nine failing checkpoints, applicability was
> predicted in advance by a state probe: checkpoints whose state does not carry the answer
> at response time did not capture (0/6) and mostly stayed at chance (5/6), so the
> intervention generalises exactly to the failure class it targets, not to all failures.
> Two distinct acquisition routes were then separated experimentally: **capture-and-hold**,
> where the readout learns utilisation after the answer is already present, and
> **encoder-side acquisition**, where an exactly frozen, uninformative readout is left in
> place and the encoder learns to write the answer into the state (3/3 data-order
> replications, decodability 0.53 → 1.000, head/C/D deltas 0.000, causal substitution
> intact). Acquisition and utilisation therefore need not occur in the same training stage.
> A separate 20-seed population run on the frozen D800 configuration bounds the reference
> baseline: 20/20 seeds acquire the delayed capability (delayed ≥ 0.9) with immediate
> accuracy 1.000 and state reset at chance (23/23 including the original seeds; 95% CI
> 0.852–1.000), so the baseline is reliable within the tested regime rather than a
> small-sample result. Evaluation-only delay scaling separates retention into three
> timescales: perfect on all three measures to delay 10 (23/23), information intact but use
> lost in about half the population at delay 30 (23/23 vs 11/23, bimodal and confidently
> wrong), and information itself lost by delay 100 (0/23) — so the "present but unused"
> dissociation also appears as a function of time, not only of training.

Open items: the mechanism of the ~20-episode readout oscillation; delay scaling; the
single population seed with an unseen-size gap (encoder edge-density calibration); and the
unplanned "encoder-side acquisition" route seen once in `B_delayed:2` (readout frozen,
representation created de novo) — replication across data-order seeds before any claim.

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
  training — both reaching **3/3 seeds** against the pre-registered *observed* criterion.
- **D800 population reliability complete**: 20 fresh seeds on the frozen config, 20/20
  delayed ≥ 0.9 with immediate 1.000 (95% CI 0.832–1.000; 23/23 combined, CI 0.852–1.000),
  state reset at chance in every seed (`scripts/d800_population_analysis.py`, report §8h).
- **H7b generalisation complete** (§8i): on the seven remaining failing checkpoints,
  capture fires only where the state still carries the answer at response time; all three
  such checkpoints in the study pass the full rubric (3/3, CI 0.292–1.000), the six
  answer-absent ones capture 0/6 and mostly stay at chance. One answer-absent checkpoint
  passed by encoder-side acquisition under a frozen readout (flagged for replication).
- **B2 replication complete** (§8j): encoder-side acquisition reproduced **3/3** across
  independent data orders — no capture event, readout exactly frozen (head/C/D deltas
  0.000), decode 0.53 → 1.000, full rubric pass. Treated as a second acquisition route,
  separate from H7b.
- **Delay scaling complete** (§8k): evaluation-only, 23 checkpoints × delays 3/10/30/100.
  Perfect retention on all three measures to delay 10 (23/23); at delay 30 information
  survives 23/23 while functional and behavioural retention survive 11/23 (bimodal,
  confidently wrong failures); at delay 100 information is lost (0/23).
- **Delay-30 geometry + intervention complete (§8l, two stages)**: stage A showed the
  delay-30 failures are *translated* decision boundaries (directions aligned, one class
  entirely wrong) — rotation and decay rejected. Stage B showed readout-only coupling
  rescues all 12 at delay 30 (capture 12/12, full rescue signature 12/12), but the
  pre-registered full rubric is 0/12 because the episode-based joint stage (32:1
  immediate:delayed events at delay 30) erases the rescue while the readout stays frozen.
  Open: an event-balanced joint stage; delay-100 representation loss.
- Reproducibility verified: `scripts/verify_reproducibility.py` regenerates all published
  metrics from committed checkpoints + configs (12/12 exact matches).
- Open items: the mechanism of the ~20-episode readout oscillation; the single population
  seed with an unseen-size gap (encoder edge-density calibration); how far the encoder-side
  acquisition route generalises beyond its one checkpoint; whether the delay-30 failures are
  a rotation of the state relative to the fixed readout (needs a direct geometric measure).
- ~~Unseen-size anomaly~~ resolved: was an evaluation-protocol defect (generic blank not
  threaded into the generalization probe). Corrected, and the controlled size sweep
  (100→200, degree-matched) is flat — see `scripts/run_size_sweep.py` outputs in
  `results/*/size_sweep/`. In the 23-seed population, 22 generalize to n=200 at ≥ 0.985;
  one seed (19, 0.560) has a genuine edge-density calibration gap that vanishes
  (1.000) when mean degree is matched — see §8h.
- State-content analysis (decodability / answer-vs-feature / substitution) complete on
  18 condition-seeds: `scripts/run_state_analysis.py`, results in `results/*/state_content/`.
- H5 readout-coupling rescue run (3 arms × 2 phenotype-(iii) checkpoints, pre-registered
  rubric): see `scripts/run_coupling_rescue.py`, `results/coupling_rescue/`;
  report §8b.
