"""State-content analysis: what does the SSM state carry? (report §7 item 3)

Four analyses on saved checkpoints (nothing retrained):

``capture_trial_states``
    One trial per instance, states captured at fixed points of the
    computation: ``pre_info`` (after one generic-blank step, before any
    instance information), ``post_present`` (right after the answer is
    computable), per-delay-step states, ``pre_response``.
``decode_answer``
    Held-out linear (ridge) decodability of the binary answer from the state
    at each capture point, against the majority-class baseline.
``answer_vs_feature``
    Does the state encode the *answer* or retain *input features*?
    Centroid/state distances within-answer vs between-answer, plus
    instance-identity (nearest-state) recognition among same-answer states
    vs shuffled-pair baseline. A mixture is expected; the quantities are
    reported separately, not forced into one verdict.
``state_substitution``
    Causal test: after instance i's presentation, the carried state is
    replaced (from a second run at the *same* computation point, so
    in-distribution) by:
      - same answer (different instance): correct prediction should survive;
      - different answer: prediction should follow the substituted state;
      - shuffled states: in-distribution values, wrong pairing → chance-ish;
      - zero state (fresh reset): chance expected.
    Substitution answers WHERE the prediction's information lives; reset
    only showed THAT it lives in the state.

All probes evaluate on held-out instances; no training data is used.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from artificial_system.evaluation.graph_classification import _chunks, observe_instances


# --------------------------------------------------------------------- capture

@torch.no_grad()
def capture_trial_states(
    model,
    dataset,
    delay: int = 3,
    batch_size: int = 50,
    marker_dim: int = 2,
    generic_blank=None,
) -> dict[str, Tensor]:
    """States for every instance at each trial point, concatenated [N, state].

    Points (with trained delay D):
      ``pre_info``      : generic blank BEFORE presentation (no info);
      ``post_present``  : right after the presentation step;
      ``delay_1..D``    : after each blank delay step;
      ``pre_response``  : the state the response step consumes.
    Also returns per-instance logits at the response step and the accuracy.
    """
    model.eval()
    out_lists: dict[str, list[Tensor]] = {k: [] for k in ["pre_info", "post_present"]}
    out_lists.update({f"delay_{t+1}": [] for t in range(delay)})
    out_lists["pre_response"] = []
    logits_all: list[Tensor] = []
    for idxs in _chunks(len(dataset), batch_size):
        b = len(idxs)
        generic = observe_instances(
            dataset, idxs, keep_markers=False, marker_dim=marker_dim,
            generic_blank=generic_blank,
        )
        state = model.initial_state(b)
        _, state, _ = model.step(generic, state)   # pre_info
        out_lists["pre_info"].append(state.detach().cpu())

        obs = observe_instances(dataset, idxs, True, marker_dim)
        _, state, _ = model.step(obs, state)       # presentation
        out_lists["post_present"].append(state.detach().cpu())

        for t in range(delay):
            blank = observe_instances(
                dataset, idxs, keep_markers=False, marker_dim=marker_dim,
                generic_blank=generic_blank,
            )
            _, state, _ = model.step(blank, state)
            out_lists[f"delay_{t+1}"].append(state.detach().cpu())
        out_lists["pre_response"].append(state.detach().cpu())

        logits, _, _ = model.step(generic, state)
        logits_all.append(logits.argmax(-1).cpu())
    model.train()
    result = {k: torch.cat(v) for k, v in out_lists.items()}
    result["_prediction"] = torch.cat(logits_all)
    result["_label"] = dataset.labels
    return result


# ----------------------------------------------------------------- A: decoding

def ridge_decode_cv(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1.0,
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Held-out binary ridge classification (closed form). Returns accuracy
    and the majority-class baseline (frequency of the more common class)."""
    y = (np.asarray(y) >= 1).astype(np.float64) * 2 - 1  # {-1, +1}
    n, d = X.shape
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(n) % n_folds
    correct = 0
    for f in range(n_folds):
        tr, te = fold_of != f, fold_of == f
        if tr.sum() < 5 or te.sum() == 0:
            continue
        mu = X[tr].mean(0)
        Xtr = X[tr] - mu
        ytr = y[tr]
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ ytr)
        b = ytr.mean()
        pred = np.sign((X[te] - mu) @ w + b)
        correct += int((np.sign(pred) == np.sign(y[te])).sum())
    acc = correct / n
    baseline = max(float((y > 0).mean()), float((y < 0).mean()))
    return {"accuracy": acc, "majority_baseline": baseline, "n": int(n)}


def decode_at_points(states: dict, points: list[str], lam: float = 1.0, seed: int = 0) -> dict:
    """Decode the answer from the state at each named capture point."""
    labels = states["_label"].numpy()
    out = {}
    for pt in points:
        X = states[pt].numpy()
        out[pt] = ridge_decode_cv(X, labels, lam=lam, seed=seed)
    return out


# --------------------------------------------------- B: answer vs feature

def pairwise_cosine(X: Tensor) -> Tensor:
    Xc = X - X.mean(dim=0, keepdim=True)
    Xn = Xc / Xc.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return Xn @ Xn.t()


def answer_vs_feature(states: dict, seed: int = 0) -> dict:
    """Separation of same-answer vs between-answer state similarity, and
    instance-identity recognition within answer classes."""
    labels = np.asarray(states["_label"].numpy())
    X = states["pre_response"]
    n = X.shape[0]
    C = pairwise_cosine(X).numpy()
    iu = np.triu_indices(n, 1)
    cos = C[iu]
    same = labels[iu[0]] == labels[iu[1]]

    sim = C.copy()
    np.fill_diagonal(sim, -2.0)
    nearest = sim.argmax(axis=1)
    identity_same_answer_counts = {
        "nearest_same_label_fraction": float((labels[nearest] == labels[np.arange(n)]).mean()),
        "chance_same_label_fraction": float(
            ((labels == 1).sum() * ((labels == 1).sum() - 1)
             + (labels == 0).sum() * ((labels == 0).sum() - 1))
            / (n * (n - 1))
        ) if n > 1 else 0.5,
    }
    return {
        "mean_cos_same_answer": float(cos[same].mean()) if same.any() else float("nan"),
        "mean_cos_diff_answer": float(cos[~same].mean()) if (~same).any() else float("nan"),
        "answer_separation_cos": float(cos[~same].mean() - cos[same].mean()) if same.any() and (~same).any() else float("nan"),
        "mean_pre_response_norm": float(X.norm(dim=1).mean()),
        **identity_same_answer_counts,
    }


# ------------------------------------------------------------------- D: substitution

@torch.no_grad()
def state_substitution(
    model,
    dataset,
    delay: int = 3,
    batch_size: int = 50,
    marker_dim: int = 2,
    generic_blank=None,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Replace the carried state right after instance i's presentation with a
    donor state captured at the SAME computation point (in-distribution),
    then finish i's delay steps (generic blanks) and read the response.

    Conditions:
      ``own``          : no substitution (temporal control, should be ~1.0);
      ``same_answer``  : donor label == i's label → own answer should survive;
      ``diff_answer``  : donor label != i's label → prediction should FOLLOW
                         the substituted (donated) answer;
      ``shuffled``     : own states, rows shuffled within batch (in-distribution
                         values, wrong pairing) → chance-ish;
      ``zero``        : fresh reset → chance.
    Note the run uses blank observations after presentation, so the donated
    state is the ONLY instance information in the second half of the trial;
    ``diff_answer`` following the donated answer is the causal signature."""
    n = len(dataset)
    labels = np.asarray(dataset.labels.numpy())
    rng = np.random.default_rng(seed)

    # one shared donor capture pass (same capture point for every condition)
    donors = capture_trial_states(model, dataset, delay, batch_size, marker_dim, generic_blank)
    donor_states = donors["post_present"]  # [N, state]

    # per-instance donor assignment, deterministic given seed
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]

    def pick(i: int, want: int) -> int:
        pool = pos if (want == 1) else neg
        excl = pool[pool != i]
        return int(excl[rng.integers(len(excl))]) if len(excl) else int(pool[0])

    donor_same = np.array([pick(i, labels[i]) for i in range(n)])
    donor_diff = np.array([pick(i, 1 - labels[i]) for i in range(n)])

    model.eval()
    conditions = ["own", "same_answer", "diff_answer", "shuffled", "zero"]
    counts = {c: 0 for c in conditions}
    follow_donated = {"same_answer": [0, 0], "diff_answer": [0, 0]}
    totals = {c: 0 for c in conditions}
    norm_ratio: dict[str, list[float]] = {c: [] for c in conditions}
    rng2 = np.random.default_rng(seed)

    for idxs in _chunks(n, batch_size):
        b = len(idxs)
        own_labels = labels[np.asarray(idxs)]
        generic = observe_instances(dataset, idxs, False, marker_dim, generic_blank)
        obs = observe_instances(dataset, idxs, True, marker_dim)
        state = model.initial_state(b)
        _, state, _ = model.step(obs, state)
        own = state.detach().clone()

        subs = {
            "own": own,
            "same_answer": donor_states[donor_same[idxs]],
            "diff_answer": donor_states[donor_diff[idxs]],
            "shuffled": own[torch.randperm(b, generator=torch.Generator().manual_seed(seed))],
            "zero": torch.zeros_like(own),
        }
        targets = {
            "own": own_labels,
            "same_answer": labels[donor_same[idxs]],  # == own labels by construction
            "diff_answer": labels[donor_diff[idxs]],  # the DONATED answer
            "shuffled": own_labels,
            "zero": own_labels,
        }

        for cond in conditions:
            s = subs[cond].clone()
            norm_ratio[cond].append(
                float((s.norm(dim=1) / own.norm(dim=1).clamp(min=1e-9)).mean())
            )
            state = s
            for _ in range(delay):
                _, state, _ = model.step(generic, state)
            logits, _, _ = model.step(generic, state)  # response on blank obs
            pred = logits.argmax(-1).numpy()
            hit = pred == targets[cond]
            counts[cond] += int(hit.sum())
            totals[cond] += b
            if cond in ("same_answer", "diff_answer"):
                follow_donated[cond][0] += int(hit.sum())
                follow_donated[cond][1] += b
    model.train()
    rows = []
    for cond in conditions:
        acc = counts[cond] / max(totals[cond], 1)
        row = {
            "condition": cond,
            "accuracy": acc,
            "follows_substituted_state": (
                follow_donated[cond][0] / max(follow_donated[cond][1], 1)
                if cond in ("same_answer", "diff_answer") else float(acc)
            ),
            "mean_norm_ratio": float(np.mean(norm_ratio[cond])) if norm_ratio[cond] else float("nan"),
        }
        rows.append(row)
    summary = {f"subst_{r['condition']}_accuracy": r["accuracy"] for r in rows}
    summary["subst_same_follows"] = rows[1]["follows_substituted_state"]
    summary["subst_diff_follows"] = rows[2]["follows_substituted_state"]
    return rows, summary


def drift_metrics(states: dict, delay: int) -> dict:
    """Temporal dynamics of the state trajectory (delay = trained delay).

    ``blank_step_drift``: mean displacement of the state across the blank
    delay steps — the system's response to irrelevant input (should be small
    if the answer is protected);
    ``revision_gain``: displacement caused by the presentation step, divided
    by blank-step drift — sensitivity to relevant evidence vs irrelevant
    filler (≫1 when evidence is absorbed, ≈1 when state barely moves for
    input changes)."""
    pre = states["pre_info"]
    post = states["post_present"]
    prev = post
    drifts = []
    for t in range(1, int(delay) + 1):
        cur = states[f"delay_{t}"]
        drifts.append((cur - prev).norm(dim=1))
        prev = cur
    blank_drift = torch.stack(drifts).mean().item()
    presentation = (post - pre).norm(dim=1).mean().item()
    return {
        "blank_step_drift": blank_drift,
        "presentation_displacement": presentation,
        "revision_gain": presentation / max(blank_drift, 1e-12),
    }


def run_state_content(
    model,
    dataset,
    delay: int = 3,
    batch_size: int = 50,
    marker_dim: int = 2,
    generic_blank=None,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Full A + B + D analysis for one checkpoint. Returns (rows, summary)."""
    states = capture_trial_states(
        model, dataset, delay, batch_size, marker_dim, generic_blank
    )
    points = ["pre_info", "post_present", "pre_response"] + [
        f"delay_{t+1}" for t in range(int(delay))
    ]
    dec = decode_at_points(states, points, seed=seed)
    avf = answer_vs_feature(states, seed=seed)
    acc = float((states["_prediction"] == states["_label"]).float().mean())
    drift = drift_metrics(states, delay)
    subst_rows, subst_summary = state_substitution(
        model, dataset, delay, batch_size, marker_dim, generic_blank, seed
    )
    summary: dict = {"behavioural_accuracy": acc}
    for pt, d in dec.items():
        summary[f"decode_{pt}"] = d["accuracy"]
        summary[f"decode_{pt}_baseline"] = d["majority_baseline"]
        summary[f"decode_{pt}_gap"] = d["accuracy"] - d["majority_baseline"]
    summary.update({f"avf_{k}": v for k, v in avf.items()})
    summary.update({f"drift_{k}": v for k, v in drift.items()})
    avf.update(drift)
    summary.update(subst_summary)
    rows = (
        [{"analysis": "decode", "point": pt, "accuracy": d["accuracy"],
          "baseline": d["majority_baseline"]} for pt, d in dec.items()]
        + [{"analysis": "avf", "metric": k, "value": v} for k, v in avf.items()]
        + [{"analysis": "substitution", **r} for r in subst_rows]
    )
    return rows, summary
