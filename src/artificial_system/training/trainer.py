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

    # ------------------------------------------------------------ training

    def train(self, num_episodes: int) -> list[dict]:
        """Run ``num_episodes`` training episodes; returns per-step records."""
        self.model.train()
        records: list[dict] = []
        step = getattr(self, "_global_step", 0)
        for episode in range(num_episodes):
            obs = self.env.reset(self.batch_size)
            state = self.model.initial_state(self.batch_size)
            losses: list[Tensor] = []
            for _ in range(self.seq_len):
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
