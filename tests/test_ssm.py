"""SSM tests (spec §26): step/sequence equivalence, reset, carry-over, grads."""
from __future__ import annotations

import pytest
import torch

from artificial_system.models.ssm import StateSpaceModel


def make_ssm(**kw) -> StateSpaceModel:
    defaults = dict(input_dim=8, state_dim=12, output_dim=6)
    defaults.update(kw)
    torch.manual_seed(0)
    return StateSpaceModel(**defaults)


class TestInterface:
    def test_initial_state_shape(self):
        ssm = make_ssm()
        s = ssm.initial_state(4)
        assert s.shape == (4, 12)
        assert torch.equal(s, torch.zeros_like(s))

    def test_step_shapes(self):
        ssm = make_ssm()
        y, s = ssm.step(torch.randn(4, 8), ssm.initial_state(4))
        assert y.shape == (4, 6)
        assert s.shape == (4, 12)

    def test_step_accepts_none_state(self):
        ssm = make_ssm()
        y, s = ssm.step(torch.randn(4, 8), None)
        assert s.shape == (4, 12)

    def test_sequence_matches_manual_unroll(self):
        ssm = make_ssm()
        seq = torch.randn(7, 3, 8)
        out, final = ssm(seq)
        state = ssm.initial_state(3)
        manual = []
        for t in range(7):
            y, state = ssm.step(seq[t], state)
            manual.append(y)
        assert torch.allclose(out, torch.stack(manual, 0), atol=1e-6)
        assert torch.allclose(final, state, atol=1e-6)

    def test_return_states(self):
        ssm = make_ssm()
        out, _, states = ssm(torch.randn(5, 2, 8), return_states=True)
        assert states.shape == (5, 2, 12)


class TestTemporalBehaviours:
    def test_state_carryover_makes_output_history_dependent(self):
        """Same stimulus, different history -> different output (spec §18)."""
        ssm = make_ssm(nonlinearity="none")
        z = torch.ones(1, 8)
        y1, _ = ssm.step(z, ssm.initial_state(1))
        _, s = ssm.step(torch.randn(1, 8) * 5, ssm.initial_state(1))
        y2, _ = ssm.step(z, s)
        assert not torch.allclose(y1, y2)

    def test_state_reset(self):
        ssm = make_ssm()
        _, s = ssm.step(torch.randn(4, 8), None)
        assert float(s.detach().abs().sum()) > 0
        fresh = ssm.reset_state(s)
        assert torch.equal(fresh, torch.zeros_like(fresh))

    def test_partial_state_reset(self):
        ssm = make_ssm()
        _, s = ssm.step(torch.randn(4, 8), None)
        mask = torch.tensor([True, False, True, False])
        s2 = ssm.reset_state(s, mask)
        assert float(s2[0].detach().abs().sum()) == 0
        assert float(s2[1].detach().abs().sum()) > 0

    def test_diagonal_decay_bounded(self):
        ssm = make_ssm(max_radius=0.95)
        a = ssm.effective_decay()
        assert float(a.abs().max()) <= 0.95 + 1e-6

    @pytest.mark.parametrize("diagonal", [True, False])
    def test_long_sequence_stable(self, diagonal):
        """State must not explode over long sequences (stability constraint)."""
        ssm = make_ssm(diagonal=diagonal, max_radius=0.99)
        _, s = ssm(torch.randn(500, 2, 8))
        assert torch.isfinite(s).all()
        assert float(s.abs().max()) < 100.0


class TestGradients:
    def test_gradient_propagates_through_time(self):
        ssm = make_ssm()
        seq = torch.randn(10, 3, 8)
        out, _ = ssm(seq)
        out[-1].sum().backward()
        assert ssm.B.weight.grad is not None
        assert float(ssm.B.weight.grad.abs().sum()) > 0
        assert float(ssm.C.weight.grad.abs().sum()) > 0

    def test_distant_input_influences_late_output(self):
        """BPTT reach: first input must carry gradient to the last output."""
        ssm = make_ssm(nonlinearity="none")
        seq = torch.zeros(8, 1, 8)
        seq[0] = 1.0
        seq.requires_grad_(True)
        out, _ = ssm(seq)
        out[-1].sum().backward()
        assert float(seq.grad[0].abs().sum()) > 0
