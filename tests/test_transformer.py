"""Lightweight smoke test for the PyTorch forecasting architecture."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_transformer import TemporalTransformer, make_dataset  # noqa: E402


def test_transformer_produces_one_value_per_forecast_horizon() -> None:
    """Check tensor dimensions without running a costly training loop."""
    generator = np.random.default_rng(42)
    values = generator.normal(size=(24, 5))
    x, y, _scale = make_dataset(values, window=8, horizon=4)
    model = TemporalTransformer(window=8, horizon=4, input_size=6)

    prediction = model(x[:3])

    assert x.shape == (13, 8, 6)
    assert y.shape == (13, 4)
    assert prediction.shape == (3, 4)
    assert torch.isfinite(prediction).all()
