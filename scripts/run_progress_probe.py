"""M1-c registered progress-field probe (TECHNICAL_REPORT §10).

Question: is the recurrent state a *progress representation* — does it encode,
and causally use, the position of the current trial relative to the response?

Per seed (0..7), one fresh control run of the registered E1 shape (400
episodes, condition D interleaved, same constants as run_resume_exactness —
imported from it so the two registered protocols cannot drift), then:

  - delayed-regime capture   : capture_trial_states, delay 3, batch 50 ->
    50 states x {post_present, delay_1, delay_2, pre_response}
  - immediate-regime capture : capture_trial_states, delay 0 -> 2 points
  - PC1 (decodable)          : multiclass ridge (lam 1.0, 5-fold CV, seed 0)
    decoding phase p in {0,1,2,3} from the 200 pooled delayed states;
    registered PASS line = CV accuracy - majority baseline >= 0.15
  - PC2 (causal)             : substitution at pre_response. Treatment donor:
    same instance's delay_1 state (phase-mismatched, same answer). Control
    donor: different same-answer instance's pre_response state (phase-matched,
    different content). Metric: fraction of instances whose prediction
    changes vs the intact prediction, treatment vs control, paired per seed.

Raw captured states are saved alongside the summary for audit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_resume_exactness import (  # noqa: E402  (registered shared shape)
    DELAY,
    EPISODES,
    GEN,
    BATCH,
    build_arm,
    run_interleaved,
)

from artificial_system.evaluation import graph_classification as graph_eval  # noqa: E402
from artificial_system.evaluation import state_content as sc  # noqa: E402

# ---- registered constants (TECHNICAL_REPORT §10 M1-c) ----
SEEDS = tuple(range(8))
POINT_LABELS = ("post_present", "delay_1", "delay_2", "pre_response")
PC1_THRESHOLD = 0.15          # delta over majority baseline
PC1_MIN_SEEDS = 6             # of 8
PC2_MIN_SEEDS = 6             # of 8


def pc2_substitution(
    model,
    dataset,
    donors: dict,
    marker_dim: int,
    generic_blank,
    seed: int,
) -> dict:
    """Registered PC2: swap the state fed to the response step.

    The response step consumes a blank observation (instance-independent), so
    its prediction is a function of the substituted state alone; intact
    predictions come from the donor capture pass (same protocol).
    """
    n = len(dataset)
    labels = donors["_label"].numpy()
    intact = donors["_prediction"].numpy()
    rng = np.random.default_rng(seed)

    # control donor: different instance, same answer, pre_response capture
    ctrl_donor = np.empty(n, dtype=np.int64)
    for i in range(n):
        pool = np.where(labels == labels[i])[0]
        pool = pool[pool != i]
        ctrl_donor[i] = int(rng.choice(pool)) if len(pool) else i

    treat_states = donors["delay_1"]                      # same instance
    ctrl_states = donors["pre_response"][torch.as_tensor(ctrl_donor)]

    change_t = np.zeros(n, dtype=bool)
    change_c = np.zeros(n, dtype=bool)
    for idxs in graph_eval._chunks(n, 50):
        chunk = list(idxs)
        blank = graph_eval.observe_instances(
            dataset, chunk, keep_markers=False, marker_dim=marker_dim,
            generic_blank=generic_blank,
        )
        lt, _, _ = model.step(blank, treat_states[chunk])
        lc, _, _ = model.step(blank, ctrl_states[chunk])
        pred_t = lt.argmax(-1).numpy()
        pred_c = lc.argmax(-1).numpy()
        change_t[np.asarray(chunk)] = pred_t != intact[np.asarray(chunk)]
        change_c[np.asarray(chunk)] = pred_c != intact[np.asarray(chunk)]

    return {
        "treatment_change_rate": float(change_t.mean()),
        "control_change_rate": float(change_c.mean()),
        "treatment_changed": int(change_t.sum()),
        "control_changed": int(change_c.sum()),
        "n": int(n),
        "control_donor_index": ctrl_donor.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", default="results/m1_progress")
    args = ap.parse_args()

    run_dir = Path(args.outdir) / f"P_seed{args.seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- registered E1 control shape (fresh run; no checkpoint reuse) ----
    model, _optimizer, tr_imm, tr_del, envs, test_ds = build_arm(args.seed)
    run_interleaved(tr_imm, tr_del, 0, EPISODES)

    generic = envs["delayed"]._generic_graph
    model.eval()

    # ---- delayed-regime capture (4 trial points x 50 instances) ----
    delayed = sc.capture_trial_states(
        model, test_ds, delay=DELAY, batch_size=50,
        marker_dim=GEN["marker_dim"], generic_blank=generic,
    )
    # ---- immediate-regime capture (descriptive; delay 0 -> 2 points) ----
    immediate = sc.capture_trial_states(
        model, test_ds, delay=0, batch_size=50, marker_dim=GEN["marker_dim"],
    )

    # ---- PC1: decode phase p from the pooled delayed states ----
    X, y = [], []
    for p, point in enumerate(POINT_LABELS):
        X.append(delayed[point].numpy())
        y.append(np.full(len(test_ds), p, dtype=np.int64))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    pc1 = sc.ridge_decode_cv_multiclass(X, y, lam=1.0, n_folds=5, seed=0)
    pc1["delta_over_baseline"] = pc1["accuracy"] - pc1["majority_baseline"]

    # remaining-steps r = 3 - p (registered sanity check, not a criterion)
    r = 3 - y
    pc1_r = sc.ridge_decode_cv_multiclass(X, r, lam=1.0, n_folds=5, seed=0)

    # ---- PC2: causal substitution at pre_response ----
    pc2 = pc2_substitution(
        model, test_ds, delayed, GEN["marker_dim"], generic, seed=args.seed,
    )

    # ---- descriptive: immediate regime + answer decodability + norms ----
    imm_decode = sc.decode_at_points(
        immediate, ["post_present"], lam=1.0, seed=0,
    )
    dly_answer = sc.decode_at_points(
        delayed, ["pre_response"], lam=1.0, seed=0,
    )
    norms = {
        f"delayed_{pt}": float(delayed[pt].norm(dim=-1).mean())
        for pt in POINT_LABELS
    }
    norms["immediate_post_present"] = float(
        immediate["post_present"].norm(dim=-1).mean()
    )

    summary = {
        "seed": args.seed,
        "episodes": EPISODES,
        "delay": DELAY,
        "pc1": pc1,
        "pc1_remaining_sanity": {k: pc1_r[k] for k in
                                 ("accuracy", "majority_baseline", "n")},
        "pc2": pc2,
        "immediate_decode_post_present": imm_decode["post_present"],
        "delayed_answer_decode_pre_response": dly_answer["pre_response"],
        "delayed_accuracy": float(
            (delayed["_prediction"] == delayed["_label"]).float().mean()
        ),
        "immediate_accuracy": float(
            (immediate["_prediction"] == immediate["_label"]).float().mean()
        ),
        "state_norms": norms,
    }
    with open(run_dir / "probe_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    torch.save(
        {pt: delayed[pt] for pt in POINT_LABELS}
        | {"_label": delayed["_label"], "_prediction": delayed["_prediction"]},
        run_dir / "delayed_states.pt",
    )
    print(f"[P_seed{args.seed:03d}] done -> {run_dir / 'probe_summary.json'} "
          f"(pc1 delta={pc1['delta_over_baseline']:.3f}, "
          f"pc2 treat={pc2['treatment_change_rate']:.2f} "
          f"ctrl={pc2['control_change_rate']:.2f})")


if __name__ == "__main__":
    main()
