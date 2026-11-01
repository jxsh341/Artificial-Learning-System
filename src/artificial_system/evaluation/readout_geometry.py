"""Readout–probe geometry (report §8l, stage A).

The held-out linear probe shows the answer is *recoverable* from the state. It
does not show that the answer occupies the direction the trained readout uses.
This module measures both directions in the same (raw) state coordinate system:

- ``w_probe``  : ridge direction fitted on the captured response-step states
                 (translation-invariant, so centring does not change its
                 direction);
- ``w_readout``: gradient of the trained logit difference with respect to the
                 state at the mean state, ``d(logit1 - logit0)/ds`` — this
                 includes the SSM output projection C and any head
                 nonlinearity, so it is the *effective* readout direction.

Because the head is linear, ``logit1 - logit0 = w_readout . s + c`` exactly, so
the readout's decision boundary is the hyperplane ``w_readout . s = -c``.  The
probe's boundary is ``w_probe . (s - mean) = 0``.  Two geometrically distinct
failure modes are therefore separable:

- **rotation**   : the two directions diverge (low cosine);
- **translation**: the directions agree but the readout boundary is displaced
                   relative to the class means (``boundary_vs_mid`` outside
                   +/-0.5 means one class is entirely on the wrong side).

Reported per checkpoint x delay: cosine, per-class and overall readout
accuracy, probe accuracy (held-out CV and in-sample), per-class projections,
signed and absolute margin, state norm, prediction agreement, and the
normalised boundary position.
"""
from __future__ import annotations

import numpy as np
import torch

from artificial_system.evaluation.graph_classification import observe_instances
from artificial_system.evaluation.state_content import (
    capture_trial_states,
    ridge_decode_cv,
)


def probe_direction(X: torch.Tensor, y: np.ndarray, lam: float = 1.0):
    """Ridge direction fitted on all instances (for direction only; accuracy is
    measured separately with held-out CV)."""
    Xn = X.numpy().astype(np.float64)
    yy = (np.asarray(y) >= 1).astype(np.float64) * 2.0 - 1.0
    mu = Xn.mean(0)
    Xc = Xn - mu
    d = Xn.shape[1]
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yy)
    return w, mu


def readout_direction(model, blank_graph, state_vec: torch.Tensor) -> torch.Tensor:
    """Effective direction of the trained readout at ``state_vec``:
    ``d(logit1 - logit0)/ds`` (accounts for C and head nonlinearity)."""
    s = state_vec.clone().detach().reshape(1, -1).requires_grad_(True)
    logits, _, _ = model.step(blank_graph, s)
    margin = logits[0, 1] - logits[0, 0]
    (grad,) = torch.autograd.grad(margin, s)
    return grad.reshape(-1).detach()


def margin_at(model, blank_graph, state_vec: torch.Tensor) -> float:
    with torch.no_grad():
        s = state_vec.detach().reshape(1, -1)
        logits, _, _ = model.step(blank_graph, s)
        return float(logits[0, 1] - logits[0, 0])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def analyse(
    model,
    dataset,
    delay: int,
    marker_dim: int = 2,
    generic_blank=None,
    seed: int = 0,
    lam: float = 1.0,
) -> dict:
    """Full geometry analysis for one checkpoint at one delay."""
    model.eval()
    states = capture_trial_states(
        model, dataset, delay=delay, batch_size=50,
        marker_dim=marker_dim, generic_blank=generic_blank,
    )
    Xt = states["pre_response"]
    X = Xt.numpy().astype(np.float64)
    labels = states["_label"].numpy().astype(int)
    dec = ridge_decode_cv(Xt.numpy(), labels, lam=lam, seed=seed)

    w_p, mu = probe_direction(Xt, labels, lam=lam)
    blank = observe_instances(
        dataset, [0], keep_markers=False, marker_dim=marker_dim,
        generic_blank=generic_blank,
    )
    s_bar = Xt.mean(0)
    w_r = readout_direction(model, blank, s_bar).numpy()
    c = margin_at(model, blank, s_bar) - float(w_r @ s_bar.numpy())

    margin = X @ w_r + c                      # exact for a linear head
    probe_score = (X - mu) @ w_p              # probe boundary at the data mean
    pred_readout = (margin > 0).astype(int)
    pred_probe = (probe_score > 0).astype(int)
    signed = np.where(labels >= 1, margin, -margin)

    # boundary position relative to the class means, in units of class separation
    t0 = float(margin[labels == 0].mean()) if (labels == 0).any() else float("nan")
    t1 = float(margin[labels == 1].mean()) if (labels == 1).any() else float("nan")
    sep = t1 - t0
    mid = 0.5 * (t0 + t1)

    return {
        "delay": int(delay),
        "n_instances": int(len(labels)),
        "decode_accuracy": float(dec["accuracy"]),
        "decode_baseline": float(dec["majority_baseline"]),
        "probe_accuracy_insample": float((pred_probe == labels).mean()),
        "readout_accuracy": float((pred_readout == labels).mean()),
        "readout_accuracy_class0": float((pred_readout[labels == 0] == 0).mean())
            if (labels == 0).any() else float("nan"),
        "readout_accuracy_class1": float((pred_readout[labels == 1] == 1).mean())
            if (labels == 1).any() else float("nan"),
        "cos_probe_readout": _cos(w_p, w_r),
        "probe_norm": float(np.linalg.norm(w_p)),
        "readout_norm": float(np.linalg.norm(w_r)),
        "state_norm_mean": float(np.linalg.norm(X, axis=1).mean()),
        "mean_state_norm_centred": float(np.linalg.norm(X - mu, axis=1).mean()),
        "signed_margin_mean": float(signed.mean()),
        "abs_margin_mean": float(np.abs(margin).mean()),
        "margin_at_mean_state": float(w_r @ mu + c),
        "class_margin_separation": float(sep),
        "boundary_vs_mid": float(-mid / sep) if sep != 0 else float("nan"),
        "proj_probe_class0": float(probe_score[labels == 0].mean()) if (labels == 0).any() else float("nan"),
        "proj_probe_class1": float(probe_score[labels == 1].mean()) if (labels == 1).any() else float("nan"),
        "proj_readout_class0": t0,
        "proj_readout_class1": t1,
        "prediction_agreement": float((pred_probe == pred_readout).mean()),
    }
