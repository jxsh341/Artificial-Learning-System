# ALS M1 Live Inspection — Implementation (REQ-ALS-M1-1)

Contract **ALS-M1-LIVE-INSPECT/1.0** · Implementation commit **`55d7086`** ·
Tests: `tests/test_live_inspection.py` (15) · Full suite **180 passed**.

## What was added (all additive; nothing existing changed behaviour)

`src/artificial_system/inspection.py`:

- `CONTRACT_VERSION = "ALS-M1-LIVE-INSPECT/1.0"` — version tag stamped on
  every response.
- `INSPECTION_MODES = ("checkpoint", "live_snapshot", "captured_state")`.
- `SUPPORTED_LIVE_QUERIES = ("summary", "probe", "inspect_state",
  "state_digest", "availability")`.
- `parameter_digest(model) -> str` — sha256 over the sorted state dict
  (names + shapes + raw bytes). Equal digests ⇔ bit-identical state. This is
  the execution-version / read-only-verification primitive (also usable by
  Hybrid directly).
- `LiveSessionHandle` — frozen live-session snapshot + query surface.
- `SystemHandle.summary()` gains two additive keys (`mode`, `contract_version`);
  no existing key or operation changed.

## API reference

### `LiveSessionHandle.capture(model, optimizer=None, trainers=None, envs=None, session_id=None) -> LiveSessionHandle`

Call at an execution boundary (between two `Trainer.train` calls, or at the
registered mid-episode pause). Deep-copies the model; **reads** (never
writes) optimizer `state_dict()`, each trainer's `execution_state()`, and
each environment's availability/`_trial_count`. Captures
`parameter_digest(model)` as the execution version. Returns a handle fully
detached from the live objects.

### Queries

Uniform dispatcher (recommended for coordinators):

```python
handle.query(name, **kwargs) -> dict
# supported:  {"supported": True,  "mode": "live_snapshot",
#              "contract_version": "...", **result}
# unsupported: {"supported": False, "mode": ..., "contract_version": ...,
#              "reason": "..."}
```

Direct methods exist for the same five queries (`summary`, `probe`,
`inspect_state`, `state_digest`, `availability`).

| Query | Args | Returns |
|---|---|---|
| `summary` | — | `session_id`, `steps` (per trainer), `step` (combined), `captured_digest`, `num_parameters`, `optimizer` summary, `envs` summary, `mode`, `contract_version`, note *"frozen live snapshot; not a persistent-memory claim"* |
| `probe` | `dataset, delay=0, batch_size=16, marker_dim=2, generic_blank=None` | §3 task probe (`accuracy`, `accuracy_by_class`, …) + `mode` |
| `inspect_state` | `dataset, delay=3, batch_size=50, marker_dim=2, generic_blank=None, lam=1.0, seed=0` | §7 capture (`states` dict incl. `post_present/delay_*/pre_response` + `_label/_prediction`), `decode`, `state_norms`, `accuracy`, `delay`, `mode`, `captured_digest` |
| `state_digest` | — | `captured_digest`, `frozen_digest` (constant after capture — read-only canary) |
| `availability` | — | what was captured (`model`, `optimizer`, `trainer_step`, `envs`, `datasets`) + note that queries reflect the state **as of capture** |

Unsupported / under-specified queries return `{"supported": False, "reason": …}`
with the list of supported queries — they never raise.

## Lifecycle and error semantics

- **Creation**: `capture` only; raise-free; validates nothing about the live
  session's future.
- **Valid lifetime**: the handle is immutable after capture; safe to keep
  while the live session continues. It holds the state *as of capture* —
  `availability()` says so explicitly.
- **Concurrency**: ALS training is single-threaded per process; `capture`
  must be called when no `train()` is executing (execution boundary).
  After capture, queries are safe regardless of what the live session does.
- **Errors**: unsupported query → structured `supported: False` response;
  `probe`/`inspect_state` without `dataset` → structured unsupported
  response. Malformed datasets raise from the underlying measurement
  functions (documented §3/§7 behaviour, unchanged).
- **Persistence**: `capture` does not write files. Use
  `save_execution_state` separately if the captured point must be
  checkpointed (they compose — see the boundary test).

## Usage example

```python
from artificial_system.inspection import LiveSessionHandle, CONTRACT_VERSION

# ... a live session: model, optimizer, trainers {"immediate","delayed"}, envs
# between two trainer.train() calls:
handle = LiveSessionHandle.capture(
    model, optimizer=optimizer, trainers=trainers, envs=envs,
    session_id="sir-session-17")

r = handle.query("summary")            # active state as of capture
assert r["supported"] and r["mode"] == "live_snapshot"
print(r["steps"], r["captured_digest"][:12])

probe = handle.query("probe", dataset=test_ds, delay=0, batch_size=16)
state = handle.query("inspect_state", dataset=test_ds, delay=3, batch_size=50)

# the live session keeps training; the handle still serves capture-point state
# and nothing about the session changed (verified by hash in the test suite)
```

## Backward compatibility

- `SystemHandle.load / probe / inspect_state / summary` unchanged (summary
  gained additive keys). Hybrid's existing integration is untouched.
- Version notification to Hybrid: **ALS-M1-LIVE-INSPECT/1.0**, delivered at
  commit `55d7086` (see `ALS_M1_HYBRID_INTERFACE_CONTRACT.md`).
