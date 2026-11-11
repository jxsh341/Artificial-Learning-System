"""System-level query API (M1-b, TECHNICAL_REPORT §10).

Two handle types:

- :class:`SystemHandle` wraps a **checkpoint** (old or new format) as a
  queryable handle for an external coordinator: the model is rebuilt from the
  embedded configuration, and validated measurements (task probe, trial-point
  state capture + held-out decode, execution summary) are exposed as methods.
  The API adds no new behaviour — every measurement is already registered
  elsewhere (§3, §7).

- :class:`LiveSessionHandle` (REQ-ALS-M1-1, ALS-M1-LIVE-INSPECT/1.0) wraps a
  **live session**: :meth:`LiveSessionHandle.capture` takes a fully detached
  frozen snapshot of the active model (plus optimizer / trainer / environment
  metadata) at an execution boundary; every query then runs against the frozen
  copy only. The live objects are never written to, never advanced, and never
  have their module ``training`` flag flipped — the live session may continue
  training after capture while the handle keeps serving the captured state.

Limitation (recorded): legacy checkpoints do not embed the model's input
dimensions, so :meth:`SystemHandle.load` requires them explicitly for
old-format checkpoints; execution-state checkpoints carry ``model_dims``.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch

from artificial_system.evaluation import graph_classification as graph_eval
from artificial_system.evaluation import state_content as sc
from artificial_system.models.gnn_ssm import build_model
from artificial_system.utils.checkpoints import load_checkpoint

# Interface contract version (ALS-M1-LIVE-INSPECT). The checkpoint-backed
# SystemHandle API is unchanged (backward compatible); the live-session
# extension is additive and versioned here.
CONTRACT_VERSION = "ALS-M1-LIVE-INSPECT/1.0"

# The three inspection modes distinguished by the contract:
#   checkpoint      : inspection of a serialized checkpoint (SystemHandle)
#   live_snapshot   : frozen snapshot of a live session (LiveSessionHandle)
#   captured_state  : measurement over already-captured state tensors
INSPECTION_MODES = ("checkpoint", "live_snapshot", "captured_state")

# Queries supported by LiveSessionHandle.query(); anything else returns an
# explicit unsupported response rather than raising.
SUPPORTED_LIVE_QUERIES = (
    "summary", "probe", "inspect_state", "state_digest", "availability",
)


def parameter_digest(model: torch.nn.Module) -> str:
    """Deterministic digest of a model's parameters and buffers (sha256 over
    sorted state-dict entries). Suitable for exact read-only verification:
    equal digests <=> bit-identical state."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class SystemHandle:
    """A loaded checkpoint exposed as a queryable system."""

    def __init__(self, model: torch.nn.Module, payload: dict):
        self.model = model
        self._payload = payload

    # ------------------------------------------------------------- loading

    @classmethod
    def load(
        cls,
        path: str | Path,
        node_feature_dim: int | None = None,
        edge_feature_dim: int | None = None,
        num_actions: int | None = None,
        num_nodes: int | None = None,
        map_location: str | torch.device = "cpu",
    ) -> "SystemHandle":
        """Load a checkpoint (``checkpoint.pt`` or an execution state) and
        rebuild the model from its embedded config. Input dimensions come from
        the payload's ``model_dims`` (execution states) or the explicit
        arguments (legacy checkpoints)."""
        payload = load_checkpoint(path, map_location=map_location)
        cfg = payload.get("config") or {}
        dims = (payload.get("extra") or {}).get("model_dims") or {}
        nfd = node_feature_dim if node_feature_dim is not None else dims.get("node_feature_dim")
        efd = edge_feature_dim if edge_feature_dim is not None else dims.get("edge_feature_dim", 0)
        na = num_actions if num_actions is not None else dims.get(
            "num_actions", int(cfg.get("task", {}).get("num_actions", 2))
        )
        nn = num_nodes if num_nodes is not None else dims.get("num_nodes")
        if nfd is None:
            raise ValueError(
                "model input dims unknown: pass node_feature_dim explicitly "
                "for legacy checkpoints"
            )
        model = build_model(
            cfg, node_feature_dim=int(nfd), edge_feature_dim=int(efd or 0),
            num_actions=int(na), num_nodes=nn,
        )
        load_checkpoint(path, model=model, map_location=map_location)
        return cls(model, payload)

    # ------------------------------------------------------------- queries

    @torch.no_grad()
    def probe(
        self,
        dataset,
        delay: int = 0,
        batch_size: int = 16,
        marker_dim: int = 2,
        generic_blank=None,
    ) -> dict:
        """Task-accuracy probe on a dataset (delegates to the registered
        §3 probe)."""
        return graph_eval.probe_dataset(
            self.model, dataset, delay=delay, batch_size=batch_size,
            marker_dim=marker_dim, generic_blank=generic_blank,
        )

    @torch.no_grad()
    def inspect_state(
        self,
        dataset,
        delay: int = 3,
        batch_size: int = 50,
        marker_dim: int = 2,
        generic_blank=None,
        lam: float = 1.0,
        seed: int = 0,
    ) -> dict:
        """Capture trial-point states and decode the answer from them
        (§7 machinery). Returns capture tensors, per-point decode metrics,
        state norms, and the probe accuracy."""
        states = sc.capture_trial_states(
            self.model, dataset, delay=delay, batch_size=batch_size,
            marker_dim=marker_dim, generic_blank=generic_blank,
        )
        decode = sc.decode_at_points(
            states, ["post_present", "pre_response"], lam=lam, seed=seed
        )
        norms = {
            pt: float(states[pt].norm(dim=-1).mean())
            for pt in states if not pt.startswith("_")
        }
        accuracy = float(
            (states["_prediction"] == states["_label"]).float().mean()
        )
        return {
            "states": states,
            "decode": decode,
            "state_norms": norms,
            "accuracy": accuracy,
            "delay": int(delay),
        }

    def summary(self) -> dict:
        """Execution summary stored with the checkpoint."""
        n_params = sum(p.numel() for p in self.model.parameters())
        return {
            "step": self._payload.get("step", self._payload.get("global_step")),
            "seed": self._payload.get("seed"),
            "num_parameters": n_params,
            "env_info": self._payload.get("env_info"),
            "extra": self._payload.get("extra") or {},
            # additive contract tags (ALS-M1-LIVE-INSPECT/1.0)
            "mode": "checkpoint",
            "contract_version": CONTRACT_VERSION,
        }


class LiveSessionHandle:
    """Read-only inspection of a **live** ALS session (REQ-ALS-M1-1).

    ``capture`` takes a fully detached snapshot of the active session at an
    execution boundary (between trainer calls / at a pause): the model is
    deep-copied, the optimizer / trainer / environments are only *read*
    (``state_dict`` / step counters — never mutated), and the parameter digest
    fixes the captured execution version. All queries run against the frozen
    copy under ``torch.no_grad()``, so the live session can continue training
    after capture without any interference from inspection.

    Observational guarantee: no live tensor is written, no step counter or
    environment stream is advanced, no module ``training`` flag is flipped on
    the live model (the frozen copy is switched to eval locally).
    """

    def __init__(self, frozen_model: torch.nn.Module, meta: dict):
        self.model = frozen_model  # frozen copy; owned by this handle
        self._meta = dict(meta)

    # ------------------------------------------------------------- capture

    @classmethod
    def capture(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        trainers: dict | None = None,
        envs: dict | None = None,
        session_id: str | None = None,
    ) -> "LiveSessionHandle":
        """Snapshot a live session. Call only at an execution boundary (the
        caller must guarantee no training step is running concurrently — ALS
        training is single-threaded per process, so any point between two
        ``Trainer.train`` calls is a boundary).

        ``trainers`` maps role -> Trainer (e.g. immediate/delayed); each
        trainer's own global step counter is read (never written)."""
        frozen = copy.deepcopy(model)
        frozen.eval()  # the copy only; the live module's mode is untouched

        steps: dict | None = None
        if trainers:
            steps = {name: int(t.execution_state()["global_step"])
                     for name, t in trainers.items()}

        optimizer_summary = None
        if optimizer is not None:
            state = optimizer.state_dict()
            optimizer_summary = {
                "present": True,
                "param_groups": len(state.get("param_groups", [])),
                "state_entries": len(state.get("state", {})),
            }

        env_summary = None
        if envs:
            env_summary = {}
            for name, env in envs.items():
                ok = hasattr(env, "state_dict")
                env_summary[name] = {
                    "available": bool(ok),
                    "trial_count": int(getattr(env, "_trial_count", -1)),
                }

        meta = {
            "mode": "live_snapshot",
            "contract_version": CONTRACT_VERSION,
            "session_id": session_id,
            "steps": steps,
            "step": sum(steps.values()) if steps else None,
            "captured_digest": parameter_digest(model),
            "num_parameters": sum(p.numel() for p in frozen.parameters()),
            "optimizer": optimizer_summary,
            "envs": env_summary,
        }
        return cls(frozen, meta)

    # ------------------------------------------------------------- queries

    def query(self, name: str, **kwargs) -> dict:
        """Uniform query dispatch with explicit unsupported responses.

        Returns ``{"supported": True, mode/contract_version, **result}`` for
        supported queries and ``{"supported": False, "reason": ...}``
        otherwise. Supported: ``summary``, ``probe`` (needs ``dataset``),
        ``inspect_state`` (needs ``dataset``), ``state_digest``,
        ``availability``.
        """
        base = {"mode": self._meta["mode"],
                "contract_version": self._meta["contract_version"]}
        if name not in SUPPORTED_LIVE_QUERIES:
            return {"supported": False, **base,
                    "reason": f"unknown query {name!r}; supported queries: "
                              f"{', '.join(SUPPORTED_LIVE_QUERIES)}"}
        if name in ("probe", "inspect_state") and kwargs.get("dataset") is None:
            return {"supported": False, **base,
                    "reason": f"{name} requires a dataset argument"}
        result = getattr(self, name)(**kwargs)
        return {"supported": True, **base, **result}

    # ---------------------------------------------------- measurement ops

    @torch.no_grad()
    def probe(self, dataset, delay: int = 0, batch_size: int = 16,
              marker_dim: int = 2, generic_blank=None) -> dict:
        """Task-accuracy probe against the frozen snapshot (same §3
        machinery as SystemHandle.probe)."""
        result = graph_eval.probe_dataset(
            self.model, dataset, delay=delay, batch_size=batch_size,
            marker_dim=marker_dim, generic_blank=generic_blank,
        )
        result["mode"] = self._meta["mode"]
        return result

    @torch.no_grad()
    def inspect_state(self, dataset, delay: int = 3, batch_size: int = 50,
                      marker_dim: int = 2, generic_blank=None,
                      lam: float = 1.0, seed: int = 0) -> dict:
        """Trial-point state capture + decode against the frozen snapshot
        (same §7 machinery as SystemHandle.inspect_state). The returned
        ``states`` tensors are the captured-state inspection surface."""
        states = sc.capture_trial_states(
            self.model, dataset, delay=delay, batch_size=batch_size,
            marker_dim=marker_dim, generic_blank=generic_blank,
        )
        decode = sc.decode_at_points(
            states, ["post_present", "pre_response"], lam=lam, seed=seed
        )
        norms = {
            pt: float(states[pt].norm(dim=-1).mean())
            for pt in states if not pt.startswith("_")
        }
        accuracy = float(
            (states["_prediction"] == states["_label"]).float().mean()
        )
        return {
            "states": states,
            "decode": decode,
            "state_norms": norms,
            "accuracy": accuracy,
            "delay": int(delay),
            "mode": self._meta["mode"],
            "captured_digest": self._meta["captured_digest"],
        }

    def state_digest(self) -> dict:
        """Digest of the frozen state (constant after capture — a read-only
        canary; equal to the live digest at capture time)."""
        return {"captured_digest": self._meta["captured_digest"],
                "frozen_digest": parameter_digest(self.model)}

    def availability(self) -> dict:
        """What this handle can and cannot answer (state availability)."""
        return {
            "model": True,
            "optimizer": self._meta["optimizer"] is not None,
            "trainer_step": self._meta["step"] is not None,
            "envs": self._meta["envs"],
            "datasets": "caller-supplied per query",
            "note": "queries reflect the state as of capture; the live "
                    "session may have advanced since",
        }

    def summary(self) -> dict:
        """Live-session summary as of capture."""
        return {**self._meta,
                "supported": True,
                "note": "frozen live snapshot; not a persistent-memory claim"}
