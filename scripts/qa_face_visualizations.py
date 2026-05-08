#!/usr/bin/env python3
"""
One-off QA visualizations for stage-05 face timestamps.

Generates per-season diagnostic plots from the per-second character presence
data in $SCRATCH_DIR/output/05_character_timestamps/. Not part of the
numbered pipeline — this is a sanity-check pass before downstream consumers
(e.g. te-charnet) ingest stage 05 across s1-s7.

Plots per season:
  1. screentime              — total seconds-of-presence per main character
  2. cooccurrence            — 6x6 heatmap (seconds both visible)
  3. per_episode_screentime  — episode × character heatmap

Usage:
  uv run python scripts/qa_face_visualizations.py --season 1
  uv run python scripts/qa_face_visualizations.py --all-seasons
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import constants  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qa_face_viz")

CHARACTERS = sorted(constants.MAIN_CHARACTERS)
CHAR_COLORS = {
    "chandler": "#1f77b4",
    "joey":     "#ff7f0e",
    "monica":   "#d62728",
    "phoebe":   "#e377c2",
    "rachel":   "#9467bd",
    "ross":     "#2ca02c",
}


def season_episode_files(scratch_dir: str, season: int) -> list[str]:
    pattern = os.path.join(
        scratch_dir, "output", "05_character_timestamps",
        f"friends_s{season:02d}e*_timestamps.json",
    )
    return sorted(glob(pattern))


def load_episode(path: str) -> tuple[str, int, dict[str, int]]:
    """Return (episode_id, total_seconds, {char: seconds_present})."""
    with open(path) as f:
        data = json.load(f)
    meta = data["metadata"]
    counts: dict[str, int] = defaultdict(int)
    for chars in data["timestamps"].values():
        for c in chars:
            counts[c] += 1
    return meta["episode_id"], meta["total_seconds"], dict(counts)


def cooccurrence_matrix(timestamps_path: str) -> np.ndarray:
    with open(timestamps_path) as f:
        data = json.load(f)
    n = len(CHARACTERS)
    idx = {c: i for i, c in enumerate(CHARACTERS)}
    mat = np.zeros((n, n), dtype=int)
    for chars in data["timestamps"].values():
        ids = [idx[c] for c in chars if c in idx]
        for i in ids:
            mat[i, i] += 1
            for j in ids:
                if i != j:
                    mat[i, j] += 1
    return mat


def plot_screentime(season_total: dict[str, int], season: int, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [season_total.get(c, 0) for c in CHARACTERS]
    colors = [CHAR_COLORS[c] for c in CHARACTERS]
    bars = ax.bar([c.title() for c in CHARACTERS], vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Seconds visible")
    ax.set_title(f"Season {season} — total screentime per main character")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "screentime.png"), dpi=150)
    plt.close(fig)


def plot_cooccurrence(cooc: np.ndarray, season: int, out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    # Normalize off-diagonal by sqrt(diag_i * diag_j) for an interpretable
    # co-presence rate; keep raw counts in cell labels.
    diag = np.diag(cooc).astype(float)
    norm = np.zeros_like(cooc, dtype=float)
    for i in range(len(CHARACTERS)):
        for j in range(len(CHARACTERS)):
            d = np.sqrt(diag[i] * diag[j])
            norm[i, j] = cooc[i, j] / d if d > 0 else 0
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    labels = [c.title() for c in CHARACTERS]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(CHARACTERS)):
        for j in range(len(CHARACTERS)):
            ax.text(j, i, f"{cooc[i, j]:,}",
                    ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black",
                    fontsize=8)
    ax.set_title(f"Season {season} — co-presence (raw seconds; color = sym. norm.)")
    fig.colorbar(im, ax=ax, label="cooc / sqrt(s_i × s_j)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cooccurrence.png"), dpi=150)
    plt.close(fig)


def plot_per_episode_screentime(per_ep: dict[str, dict[str, int]],
                                per_ep_total: dict[str, int],
                                season: int, out_dir: str):
    eps = sorted(per_ep.keys())
    mat = np.zeros((len(eps), len(CHARACTERS)), dtype=float)
    for i, ep in enumerate(eps):
        total = per_ep_total[ep] or 1
        for j, c in enumerate(CHARACTERS):
            mat[i, j] = per_ep[ep].get(c, 0) / total
    fig, ax = plt.subplots(figsize=(7, max(4, len(eps) * 0.18)))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(CHARACTERS)))
    ax.set_xticklabels([c.title() for c in CHARACTERS], rotation=45, ha="right")
    ax.set_yticks(range(len(eps)))
    short = [ep.replace("friends_", "") for ep in eps]
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title(f"Season {season} — fraction of episode each character is on screen")
    fig.colorbar(im, ax=ax, label="fraction of seconds")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_episode_screentime.png"), dpi=150)
    plt.close(fig)


def aggregate_cooccurrence(paths: list[str]) -> np.ndarray:
    total = np.zeros((len(CHARACTERS), len(CHARACTERS)), dtype=int)
    for p in paths:
        total += cooccurrence_matrix(p)
    return total


def run_season(scratch_dir: str, season: int):
    paths = season_episode_files(scratch_dir, season)
    if not paths:
        log.warning("season %d: no stage-05 files found, skipping", season)
        return
    log.info("season %d: %d episodes", season, len(paths))

    per_ep: dict[str, dict[str, int]] = {}
    per_ep_total: dict[str, int] = {}
    season_total: dict[str, int] = defaultdict(int)
    for p in paths:
        ep_id, total_sec, counts = load_episode(p)
        per_ep[ep_id] = counts
        per_ep_total[ep_id] = total_sec
        for c, n in counts.items():
            season_total[c] += n

    out_dir = os.path.join(scratch_dir, "output", "qa_face_viz", f"s{season}")
    os.makedirs(out_dir, exist_ok=True)

    plot_screentime(season_total, season, out_dir)
    cooc = aggregate_cooccurrence(paths)
    plot_cooccurrence(cooc, season, out_dir)
    plot_per_episode_screentime(per_ep, per_ep_total, season, out_dir)
    log.info("season %d: wrote 3 plots to %s", season, out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--season", type=int, help="single season number (1-7)")
    g.add_argument("--all-seasons", action="store_true",
                   help="run for every season that has stage-05 output")
    args = ap.parse_args()

    scratch = os.environ.get("SCRATCH_DIR")
    if not scratch:
        log.error("SCRATCH_DIR not set in environment")
        sys.exit(2)

    if args.season is not None:
        run_season(scratch, args.season)
        return

    seasons = sorted({
        int(os.path.basename(p).split("_")[1][1:3])
        for p in glob(os.path.join(scratch, "output", "05_character_timestamps",
                                   "friends_s*_timestamps.json"))
    })
    log.info("found stage-05 output for seasons: %s", seasons)
    for s in seasons:
        run_season(scratch, s)


if __name__ == "__main__":
    main()
