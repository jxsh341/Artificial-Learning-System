"""System-level query API (M1-b, TECHNICAL_REPORT §10).

Wraps a checkpoint as a queryable handle for an external coordinator: the
model is rebuilt from the embedded configuration, and validated measurements
(task probe, trial-point state capture + held-out decode, execution summary)
are exposed as methods. The API adds no new behaviour — every measurement is
already registered elsewhere (§3, §7).

Limitation (recorded): legacy checkpoints do not embed the model's input
dimensions, so :meth:`SystemHandle.load` requires them explicitly for
old-format checkpoints; execution-state checkpoints carry ``model_dims``.
"""
from __future__ import annotations

from pathlib import Path

import torch

from artificial_system.evaluation import graph_classification as graph_eval
from artificial_system.evaluation import state_content as sc
from artificial_system.models.gnn_ssm import build_model
from artificial_system.utils.checkpoints import load_checkpoint


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
        }
