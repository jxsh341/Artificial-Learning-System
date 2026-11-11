# ALS ⇄ Hybrid Interface Contract — Live Session Inspection

**Contract:** `ALS-M1-LIVE-INSPECT/1.0`
**Delivered at commit:** `55d7086` (branch `main`, repository `artificial_cns`)
**Test evidence:** 15/15 live-inspection tests, full suite **180 passed**
(`docs/ALS_M1_LIVE_INSPECTION_TEST_REPORT.md`)
**Backward compatibility:** the previously delivered `SystemHandle`
(checkpoint mode) API is unchanged; `summary()` gains two additive keys
(`mode`, `contract_version`). No existing operation changes behaviour.

## 1. What Hybrid gets

A read-only handle onto an **active** ALS session, queryable between
training steps without advancing, mutating, or otherwise disturbing the
session. Three inspection surfaces are distinguished by the `mode` tag on
every response:

| mode | source | meaning for the coordinator |
|---|---|---|
| `checkpoint` | `SystemHandle.load(path)` | a persisted execution |
| `live_snapshot` | `LiveSessionHandle.capture(...)` | the active session **as of capture** |
| `captured_state` | `inspect_state(...)["states"]` | already-captured tensors |

## 2. Request schema (Python)

```python
from artificial_system.inspection import LiveSessionHandle, CONTRACT_VERSION

# at an execution boundary (between two Trainer.train calls, or at the
# registered mid-episode pause):
handle = LiveSessionHandle.capture(
    model,                     # the live nn.Module
    optimizer=optimizer,       # optional
    trainers={"immediate": t_imm, "delayed": t_del},   # optional, any subset
    envs={"immediate": env_imm, "delayed": env_del},   # optional
    session_id="sir-session-17",                       # optional, free-form
)
```

Captures a detached frozen snapshot; returns immediately; touches nothing.

## 3. Response schemas

### `query(name, **kwargs) -> dict`

Every response carries:

```
supported: bool
mode: "live_snapshot"
contract_version: "ALS-M1-LIVE-INSPECT/1.0"
```

**`summary`** (`handle.query("summary")` or `handle.summary()`):

```
{supported, mode, contract_version,
 session_id, step, steps: {"immediate": int, "delayed": int},
 captured_digest: <sha256 hex>, num_parameters: int,
 optimizer: {present, param_groups, state_entries} | None,
 envs: {name: {available, trial_count}} | None,
 note: "frozen live snapshot; not a persistent-memory claim"}
```

**`probe`** (`query("probe", dataset=..., delay=0, batch_size=16,
marker_dim=2, generic_blank=None)`): §3 task-accuracy probe fields
(`accuracy`, `accuracy_by_class`, …) + `mode`.

**`inspect_state`** (`query("inspect_state", dataset=..., delay=3,
batch_size=50, marker_dim=2, generic_blank=None, lam=1.0, seed=0)`):

```
{supported, mode, contract_version,
 states: {post_present|delay_1..|pre_response: Tensor[N, state],
          _label, _prediction, _logits, ...},
 decode: {point: {accuracy, majority_baseline, n}},
 state_norms: {point: float}, accuracy: float, delay: int,
 captured_digest}
```

**`state_digest`**: `{captured_digest, frozen_digest}` — both constant after
capture (read-only canary; equal to the live digest at capture time).

**`availability`**: `{model: True, optimizer: bool, trainer_step: bool,
envs: {...} | None, datasets: "caller-supplied per query", note}`.

### Unsupported responses

```
{"supported": False, "mode": ..., "contract_version": ...,
 "reason": "unknown query 'X'; supported queries: summary, probe, ..."}
```

Unknown queries and missing required arguments (`probe`/`inspect_state`
without `dataset`) return this shape; they never raise.

## 4. Lifecycle semantics

1. `capture` at an execution boundary only (no `train()` in flight — ALS
   training is single-threaded per process).
2. The handle is immutable; queries are safe at any later time, including
   while the live session continues training.
3. Query results reflect the capture point; `availability()`/`summary()`
   say so explicitly. To refresh, call `capture` again (a new execution
   version; compare `captured_digest` to detect state change).
4. `capture` performs no I/O. For durable snapshots pair it with
   `save_execution_state` (the two compose; both verified at boundaries).

## 5. Read-only guarantee (what Hybrid may rely on)

Verified by hash/bit-exact tests at commit `55d7086`: after `capture` plus
a full query battery, model parameters (sha256), optimizer state,
per-trainer step counters, environment stream states and RNG streams are
**bit-identical**, and subsequent training matches an never-inspected twin
**bit-for-bit**. Repeated queries of an unchanged captured state return
identical results.

## 6. Explicitly out of scope (no claims)

Live inspection provides *observability only*. It does **not** establish
persistent memory across executions, goal representation, planning, or
multi-step reasoning, and does not modify the frozen D800 reference or any
registered protocol. The `summary()` payload carries this disclaimer.

## 7. Minimal integration example

```python
h = LiveSessionHandle.capture(model, optimizer=opt, trainers=trs, envs=envs,
                              session_id=session_id)
if h.query("availability")["optimizer"]:
    digest = h.query("state_digest")["captured_digest"]   # execution version
    acc = h.query("probe", dataset=hybrid_test_set)["accuracy"]
    states = h.query("inspect_state", dataset=hybrid_test_set)["states"]
# unsupported example:
h.query("forecast")   # -> {"supported": False, "reason": "..."}
```

**Integration note for the Hybrid agent:** consume `query()` for uniform
unsupported handling; treat `captured_digest` as the identity of the
inspected execution version; do not interpret state tensors beyond the
delivered decoders (`decode`, `state_norms`, probe accuracy) — no further
semantic labels are established experimentally.
