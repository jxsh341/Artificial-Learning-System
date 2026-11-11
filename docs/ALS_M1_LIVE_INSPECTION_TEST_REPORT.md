# ALS M1 Live Inspection — Test Report (REQ-ALS-M1-1)

Contract **ALS-M1-LIVE-INSPECT/1.0** · Implementation commit **`55d7086`** ·
`tests/test_live_inspection.py`: **15 passed** · Full suite: **180 passed**
(165 before this work + 15 new; `pytest`, CPU, 2026-11-11).

## Validation method

Exact comparisons only — sha256 parameter digests for model state,
`state_dict()` equality for optimizer and environments, `torch.equal` for
RNG streams and captured tensors, and Python `==` on returned floats. This is
appropriate (not merely convenient): every inspected quantity is a
deterministic CPU computation (dropout 0.0; §8v already demonstrated
bit-reproducibility of this pipeline), so exact equality is the *correct*
comparison standard here. No statistical tolerance is used anywhere in the
battery.

## The 10 required validations

| # | Requirement | Test(s) | Result |
|---|---|---|---|
| 1 | Inspecting a live session returns actual active state | `test_capture_reflects_live_session` — frozen params bit-equal to live at capture; `steps == {immediate: 24, delayed: 16}` matches the interleaved schedule exactly; digest matches live; availability populated | PASS |
| 2 | Inspection does not advance execution | `test_inspection_does_not_advance_or_mutate` — trainer step counters, env stream states (rng/within-trial/episode-step/trial-count/idxs) and torch RNG bit-equal before/after the full battery | PASS |
| 3 | Inspection does not modify model parameters | `test_model_parameters_bit_identical` — live digest identical after all six queries | PASS |
| 4 | Inspection does not modify optimizer state | `test_optimizer_state_unchanged` — full `state_dict()` (moments + step counters) identical after the battery | PASS |
| 5 | Inspection does not change subsequent behaviour | `test_subsequent_behaviour_bit_identical` — session continues 4 more episodes after a full battery; final digest **bit-equal** to a never-inspected twin | PASS |
| 6 | Repeated inspection of unchanged state is consistent | `test_repeated_inspection_identical` — two probes equal; digests equal; captured state tensors and decodes `torch.equal` | PASS |
| 7 | Unsupported queries explicitly reported | `test_unsupported_queries_explicit` — unknown query and missing-`dataset` cases return `{"supported": False, "reason": …}` incl. contract version | PASS |
| 8 | Correct behaviour at supported execution boundaries | `test_capture_at_episode_boundary_midrun` (between trainer calls) and `test_capture_at_mid_episode_pause_boundary` (registered M1-a pause): capture + battery at the boundary, session continues / replays, final digest bit-equal to control in both | PASS |
| 9 | Existing ALS functionality preserved | `test_existing_systemhandle_flows_preserved` + full suite (all 165 prior tests green, incl. both M1-a bit-exactness tests) | PASS |
| 10 | Reproducibility tests continue to pass | full suite green (includes the §8-series determinism tests); suite count 165 → 180 | PASS |

## Additional validations (beyond the required ten)

| Property | Test | Result |
|---|---|---|
| Frozen copy stays frozen while the live session advances | `test_handle_frozen_while_live_advances` — handle summary/probe unchanged after 4 further live episodes; live digest diverges from captured digest (as it must) | PASS |
| Live module `training` flag never flipped by inspection | `test_live_module_training_flag_untouched` | PASS |
| Optimizer *not captured* is reported honestly | `test_availability_reports_what_was_captured` | PASS |
| Cross-mode agreement | `test_live_and_checkpoint_modes_agree` — same state inspected live vs via execution-state checkpoint gives identical probe accuracy; modes correctly tagged `live_snapshot` / `checkpoint` | PASS |
| Version/mode tagging on every response | `test_contract_version_and_modes` | PASS |

## During-development corrections (recorded for honesty)

Two test-side bugs were found and fixed before delivery (implementation was
not changed after its first commit): a helper iterated state-dict values as
dicts (TypeError), and the pause-boundary test installed its snapshot hook
after an extra `interleaved(2)` call — a controlled three-way comparison
(pause+resume vs pause+battery+resume vs control, all digests identical) proved
the divergence was the test's own episode arithmetic, not the inspection
machinery. The corrected test now isolates exactly the property it claims.

## Known boundaries (documented, not defects)

- `capture` must be called at an execution boundary (single-threaded training
  loop; no internal locking — by design, see design doc §A6/A7).
- Queries reflect the state **as of capture**; the live session may have
  advanced. `availability()` states this explicitly.
- `probe`/`inspect_state` on the frozen copy flip the *copy's* module flag
  (the existing measurement helpers do that); the copy is never trained, so
  this cannot affect any session. The **live** module's flag is asserted
  untouched.
