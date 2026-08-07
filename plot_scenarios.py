"""Generate the AiPowerYou unemployment-scenario chart.

Usage:
    python plot_scenarios.py
    python plot_scenarios.py --output assets/chomage_france_scenarios.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "scenarios.csv"
DEFAULT_OUTPUT = ROOT / "assets" / "chomage_france_scenarios.png"

NAVY = "#12344D"
BLUE = "#246B91"
RED = "#C73532"
GREEN = "#3E7562"
HIGH = "#9D514D"
GRID = "#D8DEE5"
TEXT = "#17212B"
MUTED = "#66727E"
PANEL = "#F4F6F8"


def fr_percent(value: float, _position: int | None = None) -> str:
    """Format a number as a French percentage."""
    return f"{value:.1f}".replace(".", ",") + " %"


def build_chart(data_path: Path, output_path: Path) -> None:
    df = pd.read_csv(data_path)
    x = np.arange(len(df))
    forecast_start = int(df.index[df["status"] == "forecast"][0]) - 1

    fig, ax = plt.subplots(figsize=(10, 11.25), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Forecast area and uncertainty envelope.
    ax.axvspan(forecast_start - 0.05, len(df) - 0.55, color=PANEL, zorder=0)
    scenario_mask = df["median"].notna().to_numpy()
    xs = x[scenario_mask]
    low = df.loc[scenario_mask, "low"].to_numpy(float)
    median = df.loc[scenario_mask, "median"].to_numpy(float)
    high = df.loc[scenario_mask, "high"].to_numpy(float)
    ax.fill_between(xs, low, high, color=BLUE, alpha=0.10, linewidth=0, zorder=1)

    # Observed series and scenario paths.
    observed_mask = df["observed"].notna().to_numpy()
    ax.plot(
        x[observed_mask],
        df.loc[observed_mask, "observed"],
        color=NAVY,
        linewidth=3.2,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2.2,
        label="Observé",
        zorder=5,
    )
    ax.plot(
        xs, low, color=GREEN, linewidth=2.3, linestyle=(0, (5, 4)), label="Scénario bas", zorder=3
    )
    ax.plot(
        xs,
        median,
        color=RED,
        linewidth=3.5,
        marker="o",
        markersize=6,
        label="Scénario central",
        zorder=6,
    )
    ax.plot(
        xs, high, color=HIGH, linewidth=2.3, linestyle=(0, (5, 4)), label="Scénario haut", zorder=3
    )

    # Separate history from forecasts.
    ax.axvline(forecast_start, color=MUTED, linewidth=1.2, linestyle=(0, (2, 4)), zorder=2)
    ax.text(
        forecast_start + 0.12,
        9.13,
        "PROJECTION",
        color=MUTED,
        fontsize=9,
        fontweight="bold",
        va="top",
    )

    # Key labels, kept deliberately sparse.
    ax.annotate(
        "8,3 %\ndernier point observé",
        xy=(forecast_start, 8.3),
        xytext=(forecast_start - 1.45, 8.55),
        color=NAVY,
        fontsize=10,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": NAVY, "lw": 1.2},
        ha="center",
    )
    for value, color, label, dy in [
        (8.2, GREEN, "8,2 %  bas", -0.03),
        (8.5, RED, "8,5 %  central", 0.0),
        (9.0, HIGH, "9,0 %  haut", 0.03),
    ]:
        ax.text(
            len(df) - 0.82,
            value + dy,
            label,
            color=color,
            fontsize=10.5,
            fontweight="bold",
            va="center",
        )

    # Editorial styling.
    ax.set_xlim(-0.35, len(df) - 0.45)
    ax.set_ylim(7.35, 9.18)
    ax.set_xticks(x)
    ax.set_xticklabels(df["period"], rotation=35, ha="right", fontsize=9.5, color=MUTED)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(fr_percent))
    ax.tick_params(axis="y", labelsize=9.5, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.08, 0.955, "CHÔMAGE EN FRANCE", color=RED, fontsize=10, fontweight="bold")
    fig.text(
        0.08,
        0.913,
        "Trois trajectoires à un an",
        color=TEXT,
        fontsize=25,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.879,
        "Un modèle aide à éclairer le futur. Il ne remplace toujours pas la boule de cristal.",
        color=MUTED,
        fontsize=11.2,
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.847),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=2.8,
        columnspacing=1.8,
    )

    fig.text(0.08, 0.055, "AiPowerYou", color=RED, fontsize=12, fontweight="bold")
    fig.text(0.08, 0.036, "DATA  •  IA  •  DÉCISION", color=MUTED, fontsize=8.5, fontweight="bold")
    fig.text(
        0.92,
        0.044,
        "Source historique : Insee, T2 2026\nScénarios indicatifs — prévision non officielle",
        color=MUTED,
        fontsize=8,
        ha="right",
        linespacing=1.45,
    )

    plt.subplots_adjust(left=0.12, right=0.90, top=0.80, bottom=0.14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_chart(args.data, args.output)
