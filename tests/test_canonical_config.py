"""The paper configuration is the code default, and values outside it raise
an error naming the option."""

from __future__ import annotations

import pytest

from train.config import LBITrainingConfig


def _lbi(**overrides) -> LBITrainingConfig:
    base = dict(variants="lbi", backbone="transformer", vocab_size=64, seq_len=16,
                layers=2, dim=16, n_heads=4)
    base.update(overrides)
    return LBITrainingConfig(**base)


def test_defaults_are_the_paper_configuration() -> None:
    cfg = _lbi()
    assert cfg.interface_type == "vector_mlp"
    assert cfg.message_dim == 16 and cfg.region_size == 2
    assert cfg.tie_embeddings and cfg.text_corpus == "fineweb_edu"
    assert cfg.lr_schedule == "cosine" and cfg.canvas_grad_window == 0


def test_options_outside_the_paper_raise_by_name() -> None:
    with pytest.raises(ValueError, match="interface_type"):
        _lbi(interface_type="linear")
    with pytest.raises(ValueError, match="interface_jacobian_mode"):
        _lbi(interface_jacobian_mode="recompute")
    with pytest.raises(ValueError, match="text_corpus"):
        _lbi(text_corpus="tiny_shakespeare")
    with pytest.raises(TypeError):
        _lbi(unknown_option=1)


def test_window_multiplier_requires_a_window() -> None:
    with pytest.raises(ValueError, match="canvas_grad_window_lr_mult"):
        _lbi(canvas_grad_window_lr_mult=8.0)
    assert _lbi(canvas_grad_window=32, canvas_grad_window_lr_mult=8.0).canvas_grad_window == 32
