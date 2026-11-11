"""Training loop (spec §13, §23).

Episodic backpropagation-through-time: for each episode the model is
unrolled for ``sequence_length`` environment steps, losses are accumulated,
and a single optimizer step is taken. Per-step records (loss, accuracy,
reward, state norm, action probabilities) are returned and can be streamed
to an :class:`ExperimentLogger`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from artificial_system.environment.base import Environment
from artificial_system.training.losses import IGNORE_INDEX, classification_loss
from artificial_system.utils.logging import ExperimentLogger


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        env: Environment,
        optimizer: torch.optim.Optimizer,
        config: dict,
        logger: ExperimentLogger | None = None,
    ):
        self.model = model
        self.env = env
        self.optimizer = optimizer
        self.logger = logger
        cfg = dict(config)
        self.batch_size = int(cfg.get("batch_size", 16))
        self.seq_len = int(cfg.get("sequence_length", 32))
        self.grad_clip = float(cfg.get("grad_clip", 0.0))
        self.record_aux_every = int(cfg.get("record_aux_every", 0))
        self.ignore_index = IGNORE_INDEX
        # ---- execution-state support (M1-a; both disabled by default) ----
        # Called at the start of every episode, BEFORE env.reset(), with
        # (episode_index_in_call, trainer). Used to snapshot the execution
        # state for mid-episode resume (registered semantics: episode restart).
        self.episode_snapshot_fn = None
        self._current_step = 0  # global step at the start of the live episode

    # ------------------------------------------------------------ training

    def train(self, num_episodes: int, pause_after_steps: int | None = None) -> list[dict]:
        """Run ``num_episodes`` training episodes; returns per-step records.

        ``pause_after_steps`` (M1-a, default disabled): stop mid-episode after
        this many steps of an episode have been taken within this call (the
        first episode reaches the limit first) and return early. The
        trainer is then left with an incomplete episode — the caller must
        either discard it or restore a snapshot (see
        ``self.episode_snapshot_fn``); the partial autograd graph is NOT
        preserved across the pause.
        """
        self.model.train()
        records: list[dict] = []
        step = getattr(self, "_global_step", 0)
        for episode in range(num_episodes):
            self._current_step = step
            if self.episode_snapshot_fn is not None:
                self.episode_snapshot_fn(episode, self)
            obs = self.env.reset(self.batch_size)
            state = self.model.initial_state(self.batch_size)
            losses: list[Tensor] = []
            paused = False
            for k in range(self.seq_len):
                if pause_after_steps is not None and (step - self._current_step) >= pause_after_steps:
                    paused = True
                    break
                record_aux = (
                    self.record_aux_every > 0 and step % self.record_aux_every == 0
                )
                logits, state, aux = self.model.step(obs, state, return_aux=record_aux)
                targets = self.env.current_targets()
                ce = classification_loss(logits, targets, self.ignore_index)
                losses.append(ce)

                with torch.no_grad():
                    actions = logits.argmax(-1)
                    active = targets != self.ignore_index
                    rec = {
                        "step": step,
                        "episode": episode,
                        "loss": float(ce.item()),
                        "active": bool(active.all().item()),
                    }
                    if bool(active.any()):
                        rec["accuracy"] = float(
                            (actions[active] == targets[active]).float().mean().item()
                        )
                        probs = F.softmax(logits.detach(), dim=-1)
                        rec["action_prob_mean"] = float(probs.max(-1).values.mean().item())
                    else:
                        rec["accuracy"] = float("nan")
                    if isinstance(state, Tensor):
                        rec["state_norm"] = float(
                            state.detach().norm(dim=-1).mean().item()
                        )
                    if record_aux and aux:
                        rec["gnn_emb_norm"] = float(
                            aux["node_embeddings"].detach().norm(dim=-1).mean().item()
                        )
                        rec["pooled_norm"] = float(
                            aux["pooled"].detach().norm(dim=-1).mean().item()
                        )

                obs, rewards, _done, _info = self.env.step(actions)
                rec["reward"] = float(rewards.mean().item())
                records.append(rec)
                step += 1

            if paused:
                self._global_step = step
                return records

            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if self.logger is not None:
                self.logger.log_scalar("train/episode_loss", float(loss.item()), episode)
        self._global_step = step
        return records

    # -------------------------------------------------- execution state (M1-a)

    def execution_state(self) -> dict:
        """Global step counter (the only trainer state that must survive an
        episode-boundary resume; episode tensors are re-derived from the
        environment streams and RNG states, which the checkpoint carries)."""
        return {"global_step": getattr(self, "_global_step", 0)}

    def load_execution_state(self, state: dict) -> None:
        """Restore the global step counter saved by :meth:`execution_state`."""
        self._global_step = int(state["global_step"])
        self._current_step = self._global_step

    # ------------------------------------------------------------ evaluation

    @torch.no_grad()
    def evaluate(self, num_steps: int, env: Environment | None = None) -> list[dict]:
        """Run ``num_steps`` without learning; returns per-step records."""
        self.model.eval()
        env = env or self.env
        records: list[dict] = []
        obs = env.reset(self.batch_size)
        state = self.model.initial_state(self.batch_size)
        for step in range(num_steps):
            logits, state, _ = self.model.step(obs, state)
            targets = env.current_targets()
            actions = logits.argmax(-1)
            active = targets != self.ignore_index
            rec = {"step": step, "active": bool(active.all().item())}
            if bool(active.any()):
                rec["accuracy"] = float(
                    (actions[active] == targets[active]).float().mean().item()
                )
                rec["loss"] = float(
                    classification_loss(logits, targets, self.ignore_index).item()
                )
            else:
                rec["accuracy"] = float("nan")
            obs, rewards, _done, _info = env.step(actions)
            rec["reward"] = float(rewards.mean().item())
            if isinstance(state, Tensor):
                rec["state_norm"] = float(state.norm(dim=-1).mean().item())
            records.append(rec)
        self.model.train()
        return records
