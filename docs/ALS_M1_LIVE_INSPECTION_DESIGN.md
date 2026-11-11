# ALS M1 Live Inspection — Design (REQ-ALS-M1-1)

Interface contract: **ALS-M1-LIVE-INSPECT/1.0** · Implementation commit: `55d7086`
Status: implemented and verified (`ALS_M1_LIVE_INSPECTION_TEST_REPORT.md`).

## Phase A — what the existing implementation actually does

Findings from reading the code (not assumptions). Files:
`src/artificial_system/inspection.py`, `training/trainer.py`,
`environment/graph_environment.py`, `environment/cue_environment.py`,
`evaluation/graph_classification.py`, `evaluation/state_content.py`,
`utils/checkpoints.py`.

### A1. Active model and session lifecycle

An active ALS session is a set of in-memory objects: one `torch.nn.Module`
(`GraphTemporalModel`), one optimizer shared by two `Trainer` objects
(immediate and delayed roles), and two environment objects. Training proceeds
by `Trainer.train(n)` calls; the interleaved condition-D schedule alternates
single episodes between the two trainers. There is no daemon thread, no
server loop, and no internal concurrency: a session is quiescent between two
`train()` calls. **Every point between two trainer calls is an execution
boundary; the registered M1-a mid-episode pause (`pause_after_steps`) is a
second supported boundary type.**

### A2. Where working state lives

Enumerated exhaustively by the M1-a execution-state contract (§10 of
`TECHNICAL_REPORT.md`):

| State | Holder | Read path |
|---|---|---|
| Parameters + buffers | `model.state_dict()` | read-only |
| Adam moments, step counters | `optimizer.state_dict()` | read-only |
| Per-trainer global step | `Trainer.execution_state()` | read-only |
| Environment stream (rng, batch, within-trial, episode step, trial count, sampled indices; cue/mapping tensors for the cue env) | `env.state_dict()` | read-only |
| Global python/numpy/torch RNG streams | `capture_rng_states()` | read-only |

### A3. What can be queried safely

All of the above, via the read paths listed — each returns copies of tensors
or plain data and writes nothing.

### A4. What requires a checkpoint today

The delivered `SystemHandle` (M1-b) is constructed only from a serialized
checkpoint file (`SystemHandle.load(path)`). It cannot wrap an in-memory
live session; this is exactly the REQ-ALS-M1-1 blocker.

### A5. What can run against a live model

The measurement functions (`probe_dataset`, `capture_trial_states`,
`probe_delayed_intact_vs_reset`, …) accept any `nn.Module` — mathematically
they work on the live object directly.

### A6/A7. Mutation audit of the existing measurement path (the decisive finding)

Every measurement function flips the module mode around its work:
`model.eval()` → forward passes → `model.train()` (10 call sites across
`graph_classification.py` / `state_content.py`). Consequences:

- On the **current architecture** (dropout 0.0, no batch-norm, no other
  train/eval-dependent behaviour) this is *functionally inert* — verified by
  the §8v determinism check (un-instrumented control bit-identical) and again
  below by the new test battery.
- But it is a **strict state mutation of the live module** (`training` flag),
  and the forward passes run on live tensors. In-place live inspection
  therefore cannot honestly claim *bit-read-only*, and a concurrent
  inspection during a training step would interleave badly (there is no
  internal locking; the training loop is single-threaded per process).

So: inspection **between steps** is safe *today* only by architectural
accident, and only if the caller accepts the `training`-flag flip.

### A8. Is a safe snapshot possible?

Yes. `copy.deepcopy(model)` produces a fully detached module (fresh
parameters, no shared storage with the live model). The M1-a machinery
already provides read-only access to optimizer/trainer/env metadata. A
snapshot-based design therefore achieves the observational guarantee **by
construction** rather than by audit of every measurement helper.

## Phase B — design decision

**Chosen: frozen deep-copy snapshot (`LiveSessionHandle.capture`).**

- `capture(model, optimizer=None, trainers=None, envs=None, session_id=None)`
  deep-copies the model, sets the **copy** to eval mode, reads (never
  writes) optimizer / trainer / environment metadata, and records a
  deterministic `parameter_digest` (sha256 over the sorted state dict) as
  the captured execution version.
- All queries run on the frozen copy under `torch.no_grad()`. The live
  session may **continue training after capture**; the handle keeps serving
  the captured state (verified: `test_handle_frozen_while_live_advances`).
- Because the copy is detached, no future change to the measurement helpers
  (including the `eval()/train()` flips, which operate on the copy) can ever
  leak into the live session. The guarantee is structural, not behavioural.

**Alternatives considered and rejected:**

1. *In-place inspection of the live module* — rejected: flips the live
   module's `training` flag (A6); read-only-ness would depend on auditing
   every current and future measurement helper; forward passes on live
   tensors defeat isolation.
2. *Serialize → reload (`save_execution_state` to an in-memory buffer, then
   `SystemHandle.load`)* — works and gives the same isolation, but is slower
   (full serialization round-trip per inspection) and conflates the
   checkpoint format with the live path. It remains available as the
   **checkpoint mode**; the cross-mode agreement test
   (`test_live_and_checkpoint_modes_agree`) proves the two paths measure
   identically.

### The three inspection modes (distinguished, not conflated)

| Mode | Constructed from | Reflects |
|---|---|---|
| `checkpoint` | serialized checkpoint file (`SystemHandle.load`) | a persisted execution |
| `live_snapshot` | an active session at an execution boundary (`LiveSessionHandle.capture`) | the live state **as of capture** |
| `captured_state` | already-captured state tensors (`inspect_state()["states"]`, or execution-state payloads) | recorded tensors, decoupled from any session |

Every response carries `mode` and `contract_version`, so a coordinator can
always tell which surface it is reading.

### Semantic honesty constraints (per assignment §5)

- No new semantic labels are attached to neural state. The only interpreted
  quantities are those already experimentally established (task accuracy,
  trial-point capture geometry, held-out decode of the answer, state norms).
  `summary()` explicitly carries the note *"frozen live snapshot; not a
  persistent-memory claim"*.
- Quantities that cannot be queried reliably are **not exposed**: the live
  session's *future* trajectory, any "current goal", any unvalidated
  interpretation of SSM state content beyond the registered decoders.

## Phase C validation

15 tests in `tests/test_live_inspection.py`, all exact (bit/hash) comparisons
— appropriate because every inspected quantity is deterministic on CPU; no
statistical tolerance is used anywhere. See the test report.
