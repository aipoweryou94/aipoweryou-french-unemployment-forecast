"""Train a compact multivariate temporal Transformer and forecast four quarters.

This script produces the raw statistical trajectory. Economic scenario
adjustments remain explicit in ``data/scenarios.csv``.

Usage:
    python train_transformer.py
    python train_transformer.py --epochs 1500 --seeds 10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from aipoweryou_forecast.modeling import (
    FEATURE_COLUMNS,
    Scale,
    build_window,
    make_supervised_arrays,
    next_periods,
    validate_model_frame,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "model_features.csv"
DEFAULT_OUTPUT = ROOT / "data" / "transformer_raw_forecast.csv"


class TemporalTransformer(nn.Module):
    """Small encoder-only Transformer with a direct four-step head."""

    def __init__(
        self,
        window: int,
        horizon: int,
        input_size: int,
        d_model: int = 32,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.position = nn.Parameter(torch.zeros(1, window, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(x) + self.position
        encoded = self.encoder(encoded)
        # ``nn.Sequential`` is typed as ``Any`` by some PyTorch Windows wheels.
        # The head is constructed exclusively from Tensor-to-Tensor layers, so
        # this cast documents the contract and keeps mypy consistent on all OSes.
        return cast(torch.Tensor, self.head(encoded[:, -1, :]))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(
    values: np.ndarray,
    window: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, Scale]:
    """Convert the tested NumPy preparation into PyTorch tensors."""
    features, targets, scale = make_supervised_arrays(values, window, horizon)
    return (
        torch.from_numpy(features),
        torch.from_numpy(targets),
        scale,
    )


def train_one(
    values: np.ndarray,
    window: int,
    horizon: int,
    epochs: int,
    seed: int,
) -> tuple[TemporalTransformer, Scale]:
    set_seed(seed)
    x, y, scale = make_dataset(values, window, horizon)
    model = TemporalTransformer(window=window, horizon=horizon, input_size=values.shape[1] + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_function = nn.SmoothL1Loss()

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_function(prediction, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        current = float(loss.detach())
        if current < best_loss - 1e-6:
            best_loss = current
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 200:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid state")
    model.load_state_dict(best_state)
    return model, scale


def forecast_one(
    model: TemporalTransformer,
    values: np.ndarray,
    scale: Scale,
    window: int,
) -> np.ndarray:
    features = build_window(values, len(values) - window, len(values), scale)
    x = torch.tensor(features[None, :, :], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        scaled_movement = model(x).numpy()[0]
    forecast = values[-1, 0] + scaled_movement * scale.target_diff_std
    return np.asarray(forecast, dtype=float)


def ensemble_forecast(
    values: np.ndarray,
    window: int,
    horizon: int,
    epochs: int,
    seeds: int,
) -> tuple[np.ndarray, np.ndarray]:
    forecasts = []
    for seed in range(seeds):
        model, scale = train_one(values, window, horizon, epochs, seed)
        forecasts.append(forecast_one(model, values, scale, window))
    forecasts_array = np.asarray(forecasts)
    return forecasts_array.mean(axis=0), forecasts_array.std(axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = pd.read_csv(args.data)
    history = validate_model_frame(
        history,
        window=args.window,
        horizon=args.horizon,
    )
    values = history.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    mean, std = ensemble_forecast(
        values,
        window=args.window,
        horizon=args.horizon,
        epochs=args.epochs,
        seeds=args.seeds,
    )
    result = pd.DataFrame(
        {
            "period": next_periods(str(history.iloc[-1]["period"]), args.horizon),
            "transformer_mean": mean.round(3),
            "initialization_std": std.round(3),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print("\nRaw statistical output: economic scenario adjustments are separate.")


if __name__ == "__main__":
    main()
