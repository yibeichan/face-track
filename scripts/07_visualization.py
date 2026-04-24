#!/usr/bin/env python3
"""
QA visualization for the face-tracking and multimodal pipeline.

Generates per-episode and season-aggregate plots:
  1. timeline                  — temporal character presence (face + speaker)
  2. screentime                — horizontal bar chart of total screen time
  3. cooccurrence              — 6x6 co-occurrence heatmap
  4. confidence                — detection confidence distribution per character
  5. crossmodal                — stacked bar of seen+speaking / seen_only / speaking_only / absent
  6. speaking_only_timeline    — focused timeline of speaking_only events with QA flags
  7. modality_agreement        — per-second state heatmap across characters
  8. speaking_only_distribution — run-length histogram + per-character buckets
  9. multimodal_presence       — publication-quality three-panel figure

Plots 5-9 require stage 06 multimodal output.
"""

import os
import sys
import json
import argparse
import logging
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils
import constants

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MAIN_CHARACTERS_LIST = sorted(constants.MAIN_CHARACTERS)
COLORS = constants.CHARACTER_COLORS

ALL_PLOTS = [
    'timeline', 'screentime', 'cooccurrence', 'confidence', 'crossmodal',
    'speaking_only_timeline', 'modality_agreement',
    'speaking_only_distribution', 'multimodal_presence',
]

# Consistent state colors used across all multimodal plots
STATE_COLORS = {
    'seen_and_speaking': '#2ca02c',
    'seen_only':         '#7fc97f',
    'speaking_only':     '#fdae61',
    'absent':            '#d9d9d9',
}
STATE_ORDER = ['seen_and_speaking', 'seen_only', 'speaking_only', 'absent']
STATE_LABELS = ['Seen + Speaking', 'Seen only', 'Speaking only', 'Absent']


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_face_timestamps(path):
    data = read_json(path)
    ts = data.get('timestamps', {})
    total = data.get('metadata', {}).get('total_seconds', 0)
    result = {}
    for s, chars in ts.items():
        result[int(s)] = chars
    return result, total


def load_face_locations(path):
    data = read_json(path)
    return data.get('face_locations', {})


def load_multimodal(path):
    data = read_json(path)
    return data


def load_guest_candidates(path):
    """Load guest candidates JSON and aggregate per guest speaker.

    Returns a list of dicts sorted by first appearance:
        [{name, confidence, track_seconds: set, overlap_seconds: set}, ...]
    Each guest speaker gets one entry with the best (highest) confidence
    across all their candidate matches, and the union of all track/overlap
    seconds.
    """
    data = read_json(path)
    candidates = data.get('candidates', [])
    if not candidates:
        return []

    conf_rank = {'high': 3, 'medium': 2, 'low': 1}

    # Aggregate per guest speaker
    guests = {}
    for c in candidates:
        name = c['guest_speaker']
        if name not in guests:
            guests[name] = {
                'name': name,
                'confidence': c['confidence'],
                'track_seconds': set(),
                'overlap_seconds': set(),
            }
        # Keep the highest confidence across all matches
        if conf_rank.get(c['confidence'], 0) > conf_rank.get(guests[name]['confidence'], 0):
            guests[name]['confidence'] = c['confidence']

        # track_seconds is [min, max] range — expand to full set
        ts = c.get('track_seconds', [])
        if len(ts) == 2:
            guests[name]['track_seconds'].update(range(ts[0], ts[1] + 1))
        guests[name]['overlap_seconds'].update(c.get('overlap_seconds', []))

    # Sort by first overlap second
    result = sorted(guests.values(),
                    key=lambda g: min(g['overlap_seconds']) if g['overlap_seconds'] else 0)
    return result


# Confidence → alpha mapping for guest character gradient
CONFIDENCE_ALPHA = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
GUEST_COLOR = '#888888'  # neutral gray base for guest characters


# ---------------------------------------------------------------------------
# 1. Timeline plot
# ---------------------------------------------------------------------------

def plot_timeline(face_seconds, total_seconds, multimodal, output_path, episode_id,
                  guest_candidates=None):
    """Temporal presence: one row per character, x = seconds.

    If guest_candidates is provided, adds extra rows for matched guest
    characters with confidence-based color gradient.
    """
    n_main = len(MAIN_CHARACTERS_LIST)
    n_guests = len(guest_candidates) if guest_candidates else 0
    n_rows = n_main + n_guests
    fig_height = max(8, 1.2 * n_rows)

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, fig_height), sharex=True)
    if n_rows == 1:
        axes = [axes]

    has_mm = multimodal is not None
    per_second = multimodal.get('per_second', {}) if has_mm else {}

    # Main characters
    for idx, ch in enumerate(MAIN_CHARACTERS_LIST):
        ax = axes[idx]
        color = COLORS[ch]

        for sec in range(total_seconds):
            if has_mm:
                state = per_second.get(str(sec), {}).get('state', {}).get(ch)
                if state == 'seen_and_speaking':
                    ax.barh(0, 1, left=sec, height=0.8, color=color, alpha=1.0)
                elif state == 'seen_only':
                    ax.barh(0, 1, left=sec, height=0.8, color=color, alpha=0.4)
                elif state == 'speaking_only':
                    ax.barh(0, 1, left=sec, height=0.8, color=color, alpha=0.2,
                            hatch='///', edgecolor=color)
            else:
                if ch in face_seconds.get(sec, []):
                    ax.barh(0, 1, left=sec, height=0.8, color=color, alpha=0.8)

        ax.set_yticks([0])
        ax.set_yticklabels([ch.capitalize()])
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, total_seconds)

    # Guest characters
    if guest_candidates:
        for gi, guest in enumerate(guest_candidates):
            ax = axes[n_main + gi]
            conf = guest['confidence']
            alpha = CONFIDENCE_ALPHA.get(conf, 0.3)
            label = guest['name'].capitalize()

            # Track seconds (face on screen) — light background
            for sec in guest['track_seconds']:
                ax.barh(0, 1, left=sec, height=0.8,
                        color=GUEST_COLOR, alpha=0.15)

            # Overlap seconds (face + speaker) — confidence-graded color
            for sec in guest['overlap_seconds']:
                ax.barh(0, 1, left=sec, height=0.8,
                        color=STATE_COLORS['seen_and_speaking'], alpha=alpha)

            ax.set_yticks([0])
            conf_tag = f' ({conf[0].upper()})' if conf else ''
            ax.set_yticklabels([f'{label}{conf_tag}'], fontsize=9,
                               fontstyle='italic')
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlim(0, total_seconds)

    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle(f'Character Timeline — {episode_id}', fontsize=14)

    # Legend
    legend_patches = []
    if has_mm:
        legend_patches = [
            mpatches.Patch(color='gray', alpha=1.0, label='Seen + Speaking'),
            mpatches.Patch(color='gray', alpha=0.4, label='Seen only'),
            mpatches.Patch(facecolor='gray', alpha=0.2, hatch='///',
                          edgecolor='gray', label='Speaking only'),
        ]
    if guest_candidates:
        legend_patches.extend([
            mpatches.Patch(color=STATE_COLORS['seen_and_speaking'], alpha=1.0,
                          label='Guest: high conf'),
            mpatches.Patch(color=STATE_COLORS['seen_and_speaking'], alpha=0.6,
                          label='Guest: medium conf'),
            mpatches.Patch(color=STATE_COLORS['seen_and_speaking'], alpha=0.3,
                          label='Guest: low conf'),
            mpatches.Patch(color=GUEST_COLOR, alpha=0.15,
                          label='Guest: face track'),
        ])
    if legend_patches:
        fig.legend(handles=legend_patches, loc='upper right', fontsize=8,
                   ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  timeline: {output_path}")


# ---------------------------------------------------------------------------
# 2. Screen time bar chart
# ---------------------------------------------------------------------------

def compute_screentime(face_seconds, total_seconds):
    """Returns {char: seconds_visible}."""
    counts = {ch: 0 for ch in MAIN_CHARACTERS_LIST}
    for sec in range(total_seconds):
        for ch in face_seconds.get(sec, []):
            if ch in counts:
                counts[ch] += 1
    return counts


def plot_screentime(counts, total_seconds, output_path, episode_id):
    fig, ax = plt.subplots(figsize=(10, 5))
    chars = MAIN_CHARACTERS_LIST
    vals = [counts.get(ch, 0) for ch in chars]
    colors = [COLORS[ch] for ch in chars]

    bars = ax.barh([ch.capitalize() for ch in chars], vals, color=colors)
    for bar, val in zip(bars, vals):
        pct = val / total_seconds * 100 if total_seconds else 0
        ax.text(bar.get_width() + total_seconds * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val}s ({pct:.1f}%)', va='center', fontsize=9)

    ax.set_xlabel('Seconds on screen')
    ax.set_title(f'Screen Time — {episode_id}')
    ax.set_xlim(0, max(vals) * 1.2 if vals and max(vals) > 0 else 10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  screentime: {output_path}")


# ---------------------------------------------------------------------------
# 3. Co-occurrence heatmap
# ---------------------------------------------------------------------------

def compute_cooccurrence(face_seconds, total_seconds):
    """6x6 matrix of co-occurrence seconds."""
    n = len(MAIN_CHARACTERS_LIST)
    matrix = np.zeros((n, n), dtype=int)
    for sec in range(total_seconds):
        present = [ch for ch in face_seconds.get(sec, []) if ch in constants.MAIN_CHARACTERS]
        for i, ci in enumerate(MAIN_CHARACTERS_LIST):
            for j, cj in enumerate(MAIN_CHARACTERS_LIST):
                if ci in present and cj in present:
                    matrix[i][j] += 1
    return matrix


def plot_cooccurrence(matrix, output_path, episode_id):
    fig, ax = plt.subplots(figsize=(8, 7))
    labels = [ch.capitalize() for ch in MAIN_CHARACTERS_LIST]
    sns.heatmap(matrix, annot=True, fmt='d', xticklabels=labels, yticklabels=labels,
                cmap='YlOrRd', ax=ax)
    ax.set_title(f'Co-occurrence (seconds) — {episode_id}')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  cooccurrence: {output_path}")


# ---------------------------------------------------------------------------
# 4. Confidence distribution
# ---------------------------------------------------------------------------

def plot_confidence(face_locations, output_path, episode_id):
    """Violin/box plot of detection confidence per character."""
    char_confs = {ch: [] for ch in MAIN_CHARACTERS_LIST}
    for frame_idx, faces in face_locations.items():
        for f in faces:
            ch = f.get('char', '')
            if ch in char_confs:
                char_confs[ch].append(f.get('conf', 0))

    fig, ax = plt.subplots(figsize=(10, 5))
    data_for_plot = []
    labels_for_plot = []
    for ch in MAIN_CHARACTERS_LIST:
        confs = char_confs[ch]
        if confs:
            data_for_plot.append(confs)
            labels_for_plot.append(ch.capitalize())

    if data_for_plot:
        parts = ax.violinplot(data_for_plot, showmedians=True, showextrema=True)
        ax.set_xticks(range(1, len(labels_for_plot) + 1))
        ax.set_xticklabels(labels_for_plot)

        # Color the violin bodies
        for idx, body in enumerate(parts['bodies']):
            ch = MAIN_CHARACTERS_LIST[idx] if idx < len(MAIN_CHARACTERS_LIST) else None
            if ch and ch in COLORS:
                body.set_facecolor(COLORS[ch])
                body.set_alpha(0.7)

        # Flag low-confidence characters
        for idx, confs in enumerate(data_for_plot):
            median = np.median(confs)
            if median < 0.95:
                ax.annotate('LOW', xy=(idx + 1, median), fontsize=8,
                           color='red', ha='center', va='bottom')

    ax.set_ylabel('Detection confidence')
    ax.set_title(f'Face Detection Confidence — {episode_id}')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  confidence: {output_path}")


# ---------------------------------------------------------------------------
# 5. Cross-modal comparison
# ---------------------------------------------------------------------------

def plot_crossmodal(summary, output_path, episode_id):
    """Stacked bar: seen+speaking / seen_only / speaking_only / absent."""
    fig, ax = plt.subplots(figsize=(10, 5))

    states = ['seen_and_speaking', 'seen_only', 'speaking_only', 'absent']
    state_colors = ['#2ca02c', '#7fc97f', '#fdae61', '#d9d9d9']
    state_labels = ['Seen + Speaking', 'Seen only', 'Speaking only', 'Absent']

    chars = MAIN_CHARACTERS_LIST
    y_pos = np.arange(len(chars))

    left = np.zeros(len(chars))
    for state, color, label in zip(states, state_colors, state_labels):
        vals = [summary.get(ch, {}).get(state, 0) for ch in chars]
        ax.barh(y_pos, vals, left=left, color=color, label=label)
        left += np.array(vals)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([ch.capitalize() for ch in chars])
    ax.set_xlabel('Seconds')
    ax.set_title(f'Cross-modal Breakdown — {episode_id}')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  crossmodal: {output_path}")


# ---------------------------------------------------------------------------
# Helper utilities for multimodal plots
# ---------------------------------------------------------------------------

@contextmanager
def publication_style():
    """Context manager for publication-quality plot styling."""
    orig = matplotlib.rcParams.copy()
    matplotlib.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    try:
        yield
    finally:
        matplotlib.rcParams.update(orig)


def extract_speaking_only_runs(per_second, total_seconds):
    """Extract contiguous runs of speaking_only per character.

    Returns {char: [(start_sec, end_sec), ...]} where each tuple is an
    inclusive range of consecutive speaking_only seconds.
    """
    runs = {ch: [] for ch in MAIN_CHARACTERS_LIST}
    for ch in MAIN_CHARACTERS_LIST:
        run_start = None
        for sec in range(total_seconds):
            state = per_second.get(str(sec), {}).get('state', {}).get(ch)
            if state == 'speaking_only':
                if run_start is None:
                    run_start = sec
            else:
                if run_start is not None:
                    runs[ch].append((run_start, sec - 1))
                    run_start = None
        if run_start is not None:
            runs[ch].append((run_start, total_seconds - 1))
    return runs


def build_state_matrix(per_second, total_seconds):
    """Build a (n_chars, total_seconds) integer matrix encoding per-second state.

    Encoding: 0=absent, 1=seen_and_speaking, 2=seen_only, 3=speaking_only.
    """
    state_map = {'seen_and_speaking': 1, 'seen_only': 2, 'speaking_only': 3}
    matrix = np.zeros((len(MAIN_CHARACTERS_LIST), total_seconds), dtype=int)
    for sec in range(total_seconds):
        states = per_second.get(str(sec), {}).get('state', {})
        for i, ch in enumerate(MAIN_CHARACTERS_LIST):
            matrix[i, sec] = state_map.get(states.get(ch, 'absent'), 0)
    return matrix


# ---------------------------------------------------------------------------
# 6. Speaking-only timeline (QC)
# ---------------------------------------------------------------------------

def plot_speaking_only_timeline(multimodal, total_seconds, output_path, episode_id):
    """Focused timeline showing only speaking_only events with QA flag annotations."""
    per_second = multimodal.get('per_second', {})
    qa_flags = multimodal.get('qa_flags', [])
    runs = extract_speaking_only_runs(per_second, total_seconds)

    fig, axes = plt.subplots(len(MAIN_CHARACTERS_LIST), 1, figsize=(16, 7),
                             sharex=True)
    if len(MAIN_CHARACTERS_LIST) == 1:
        axes = [axes]

    long_threshold = 10  # seconds — matches long_speaking_not_seen QA flag

    for idx, ch in enumerate(MAIN_CHARACTERS_LIST):
        ax = axes[idx]
        color = COLORS[ch]
        total_so = 0

        for start, end in runs[ch]:
            duration = end - start + 1
            total_so += duration
            is_long = duration > long_threshold
            fc = '#e34a33' if is_long else color
            alpha = 0.9 if is_long else 0.5
            ec = '#b30000' if is_long else color
            ax.barh(0, duration, left=start, height=0.8,
                    color=fc, alpha=alpha, edgecolor=ec, linewidth=0.5)

        # Annotate QA flags for this character
        for flag in qa_flags:
            if flag.get('character') != ch:
                continue
            if flag['type'] == 'long_speaking_not_seen':
                fs = flag.get('start_second', 0)
                fe = flag.get('end_second', 0)
                ax.annotate(f"{fe - fs + 1}s", xy=((fs + fe) / 2, 0.45),
                            fontsize=7, color='#b30000', ha='center', va='bottom',
                            fontweight='bold')
            elif flag['type'] == 'speaker_never_seen':
                ax.text(total_seconds * 0.5, 0, 'NEVER SEEN',
                        fontsize=9, color='red', ha='center', va='center',
                        fontweight='bold')

        ax.set_yticks([0])
        ax.set_yticklabels([ch.capitalize()])
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, total_seconds)

        # Right margin: total speaking_only seconds
        ax.text(1.02, 0.5, f'{total_so}s',
                fontsize=9, va='center', color=COLORS[ch], fontweight='bold',
                transform=ax.transAxes, clip_on=False)

    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle(f'Speaking Only Events — {episode_id}', fontsize=14)

    # Legend
    legend_patches = [
        mpatches.Patch(color='gray', alpha=0.5, label='Short run (\u226410s)'),
        mpatches.Patch(color='#e34a33', alpha=0.9, label='Long run (>10s)'),
    ]
    fig.legend(handles=legend_patches, loc='upper right', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  speaking_only_timeline: {output_path}")


# ---------------------------------------------------------------------------
# 7. Modality agreement heatmap (QC)
# ---------------------------------------------------------------------------

def plot_modality_agreement(multimodal, total_seconds, output_path, episode_id,
                           guest_candidates=None):
    """Per-second heatmap of character state across the episode.

    Guest candidates are appended as additional rows with a simplified
    encoding: 0=absent, 1=overlap (seen+speaking match), 2=track only (face
    on screen but not speaking).
    """
    per_second = multimodal.get('per_second', {})
    matrix = build_state_matrix(per_second, total_seconds)

    # Append guest rows
    guest_labels = []
    if guest_candidates:
        for guest in guest_candidates:
            row = np.zeros(total_seconds, dtype=int)
            for sec in guest['track_seconds']:
                if 0 <= sec < total_seconds:
                    row[sec] = 2  # track only (seen_only encoding)
            for sec in guest['overlap_seconds']:
                if 0 <= sec < total_seconds:
                    row[sec] = 1  # overlap (seen_and_speaking encoding)
            matrix = np.vstack([matrix, row[np.newaxis, :]])
            conf_tag = guest['confidence'][0].upper()
            guest_labels.append(f"{guest['name'].capitalize()} ({conf_tag})")

    all_labels = [ch.capitalize() for ch in MAIN_CHARACTERS_LIST] + guest_labels
    n_rows = len(all_labels)

    # Bin into 5s windows if episode is long, using majority vote
    bin_size = 5 if total_seconds > 600 else 1
    if bin_size > 1:
        n_bins = (total_seconds + bin_size - 1) // bin_size
        binned = np.zeros((n_rows, n_bins), dtype=int)
        for b in range(n_bins):
            start = b * bin_size
            end = min(start + bin_size, total_seconds)
            chunk = matrix[:, start:end]
            for i in range(n_rows):
                vals, counts = np.unique(chunk[i], return_counts=True)
                binned[i, b] = vals[np.argmax(counts)]
        plot_matrix = binned
        x_label = f'Time ({bin_size}s bins)'
    else:
        plot_matrix = matrix
        x_label = 'Time (seconds)'

    cmap = mcolors.ListedColormap([
        STATE_COLORS['absent'],
        STATE_COLORS['seen_and_speaking'],
        STATE_COLORS['seen_only'],
        STATE_COLORS['speaking_only'],
    ])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig_height = max(4, 0.5 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    im = ax.imshow(plot_matrix, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(all_labels)
    # Italicize guest labels
    for i in range(len(MAIN_CHARACTERS_LIST), n_rows):
        ax.get_yticklabels()[i].set_fontstyle('italic')
        ax.get_yticklabels()[i].set_fontsize(9)

    # Draw separator line between main and guest characters
    if guest_labels:
        ax.axhline(y=len(MAIN_CHARACTERS_LIST) - 0.5, color='black',
                   linewidth=1.5, linestyle='--')

    ax.set_xlabel(x_label)
    ax.set_title(f'Modality Agreement — {episode_id}')

    # Colorbar with state labels
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8)
    cbar.ax.set_yticklabels(['Absent', 'Seen+Speaking', 'Seen only', 'Speaking only'],
                            fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  modality_agreement: {output_path}")


# ---------------------------------------------------------------------------
# 8. Speaking-only run distribution (QC)
# ---------------------------------------------------------------------------

def plot_speaking_only_distribution(multimodal, total_seconds, output_path, episode_id):
    """Histogram of speaking_only run lengths + per-character bucketed bar chart."""
    per_second = multimodal.get('per_second', {})
    runs = extract_speaking_only_runs(per_second, total_seconds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of all run lengths
    all_lengths = []
    for ch_runs in runs.values():
        all_lengths.extend(end - start + 1 for start, end in ch_runs)

    if all_lengths:
        max_len = max(all_lengths)
        bins = range(1, max_len + 2)
        ax1.hist(all_lengths, bins=bins, color='#fdae61', edgecolor='#e08214',
                 align='left')
        ax1.axvline(x=10, color='#e34a33', linestyle='--', linewidth=2,
                    label='QA threshold (10s)')
        ax1.set_yscale('log') if len(all_lengths) > 20 else None
        ax1.legend(fontsize=9)
    ax1.set_xlabel('Run length (seconds)')
    ax1.set_ylabel('Count')
    ax1.set_title('Speaking Only Run Lengths')

    # Right: per-character bucketed bar chart
    buckets = [('1s', 1, 1), ('2-5s', 2, 5), ('6-10s', 6, 10), ('>10s', 11, 9999)]
    x_pos = np.arange(len(buckets))
    bar_width = 0.12
    offsets = np.linspace(-bar_width * 2.5, bar_width * 2.5, len(MAIN_CHARACTERS_LIST))

    for ci, ch in enumerate(MAIN_CHARACTERS_LIST):
        lengths = [end - start + 1 for start, end in runs[ch]]
        bucket_counts = []
        for _, lo, hi in buckets:
            bucket_counts.append(sum(1 for l in lengths if lo <= l <= hi))
        ax2.bar(x_pos + offsets[ci], bucket_counts, width=bar_width,
                color=COLORS[ch], label=ch.capitalize())

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([b[0] for b in buckets])
    ax2.set_xlabel('Run duration')
    ax2.set_ylabel('Number of runs')
    ax2.set_title('Runs by Character & Duration')
    ax2.legend(fontsize=8, ncol=2)

    fig.suptitle(f'Speaking Only Distribution — {episode_id}', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  speaking_only_distribution: {output_path}")


def compute_speaking_only_run_data(multimodal, total_seconds):
    """Compute run-length data for season aggregation."""
    per_second = multimodal.get('per_second', {})
    runs = extract_speaking_only_runs(per_second, total_seconds)
    return {ch: [end - start + 1 for start, end in ch_runs]
            for ch, ch_runs in runs.items()}


# ---------------------------------------------------------------------------
# 9. Multimodal presence — publication quality
# ---------------------------------------------------------------------------

def plot_multimodal_presence(multimodal, total_seconds, output_path, episode_id,
                            guest_candidates=None):
    """Publication-quality three-panel figure: area chart, Gantt bars, summary.

    Guest candidates are shown as additional rows in the Gantt panel with
    confidence-graded alpha.
    """
    per_second = multimodal.get('per_second', {})
    summary = multimodal.get('summary', {})
    matrix = build_state_matrix(per_second, total_seconds)

    n_guests = len(guest_candidates) if guest_candidates else 0
    n_gantt_rows = len(MAIN_CHARACTERS_LIST) + n_guests
    mid_ratio = max(3, 0.5 * n_gantt_rows)

    with publication_style():
        fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
            3, 1, figsize=(12, 7 + 0.5 * n_guests),
            gridspec_kw={'height_ratios': [1.5, mid_ratio, 1.5]},
            sharex=False)

        # --- Top: stacked area of character count per state over time ---
        states_to_plot = ['seen_and_speaking', 'seen_only', 'speaking_only']
        seconds = np.arange(total_seconds)

        counts_per_state = {}
        for state_name in states_to_plot:
            state_val = {'seen_and_speaking': 1, 'seen_only': 2, 'speaking_only': 3}[state_name]
            counts_per_state[state_name] = np.sum(matrix == state_val, axis=0).astype(float)

        # Smooth with rolling window
        window = min(30, total_seconds // 4) if total_seconds > 30 else 1
        if window > 1:
            for state_name in states_to_plot:
                counts_per_state[state_name] = pd.Series(
                    counts_per_state[state_name]).rolling(window, center=True, min_periods=1).mean().values

        # Stacked area
        bottom = np.zeros(total_seconds)
        for state_name in states_to_plot:
            vals = counts_per_state[state_name]
            ax_top.fill_between(seconds, bottom, bottom + vals,
                                color=STATE_COLORS[state_name], alpha=0.8,
                                label=STATE_LABELS[STATE_ORDER.index(state_name)])
            bottom += vals

        ax_top.set_ylabel('Characters')
        ax_top.set_xlim(0, total_seconds)
        ax_top.set_ylim(0)
        ax_top.legend(loc='upper right', fontsize=9, ncol=3)
        ax_top.set_title(f'Multimodal Character Presence — {episode_id}')

        # --- Middle: Gantt-style per-character bars ---
        for ci, ch in enumerate(MAIN_CHARACTERS_LIST):
            prev_state = None
            seg_start = 0
            for sec in range(total_seconds + 1):
                cur = matrix[ci, sec] if sec < total_seconds else -1
                if cur != prev_state:
                    if prev_state is not None and prev_state > 0:
                        state_name = STATE_ORDER[prev_state - 1]
                        fc = STATE_COLORS[state_name]
                        hatch = '////' if state_name == 'speaking_only' else None
                        ax_mid.barh(ci, sec - seg_start, left=seg_start, height=0.7,
                                    color=fc, edgecolor='white', linewidth=0.3,
                                    hatch=hatch)
                    seg_start = sec
                    prev_state = cur

        # Guest rows in Gantt panel
        gantt_labels = [ch.capitalize() for ch in MAIN_CHARACTERS_LIST]
        if guest_candidates:
            ax_mid.axhline(y=len(MAIN_CHARACTERS_LIST) - 0.5, color='black',
                           linewidth=1, linestyle='--', alpha=0.5)
            for gi, guest in enumerate(guest_candidates):
                row_idx = len(MAIN_CHARACTERS_LIST) + gi
                alpha = CONFIDENCE_ALPHA.get(guest['confidence'], 0.3)
                conf_tag = guest['confidence'][0].upper()
                gantt_labels.append(f"{guest['name'].capitalize()} ({conf_tag})")

                # Track seconds — light background
                for sec in guest['track_seconds']:
                    if 0 <= sec < total_seconds:
                        ax_mid.barh(row_idx, 1, left=sec, height=0.7,
                                    color=GUEST_COLOR, alpha=0.15,
                                    edgecolor='none')
                # Overlap seconds — confidence-graded
                for sec in guest['overlap_seconds']:
                    if 0 <= sec < total_seconds:
                        ax_mid.barh(row_idx, 1, left=sec, height=0.7,
                                    color=STATE_COLORS['seen_and_speaking'],
                                    alpha=alpha, edgecolor='white', linewidth=0.3)

        ax_mid.set_yticks(range(n_gantt_rows))
        ax_mid.set_yticklabels(gantt_labels)
        # Italicize guest labels
        for i in range(len(MAIN_CHARACTERS_LIST), n_gantt_rows):
            ax_mid.get_yticklabels()[i].set_fontstyle('italic')
            ax_mid.get_yticklabels()[i].set_fontsize(9)
        ax_mid.set_xlim(0, total_seconds)
        ax_mid.set_ylim(-0.5, n_gantt_rows - 0.5)
        ax_mid.set_xlabel('Time (seconds)')
        ax_mid.invert_yaxis()

        # --- Bottom: grouped bar summary ---
        x_pos = np.arange(len(MAIN_CHARACTERS_LIST))
        bar_width = 0.2
        for si, (state_name, state_label) in enumerate(
                zip(STATE_ORDER[:3], STATE_LABELS[:3])):
            vals = [summary.get(ch, {}).get(state_name, 0) for ch in MAIN_CHARACTERS_LIST]
            ax_bot.bar(x_pos + (si - 1) * bar_width, vals, width=bar_width,
                       color=STATE_COLORS[state_name], label=state_label)

        ax_bot.set_xticks(x_pos)
        ax_bot.set_xticklabels([ch.capitalize() for ch in MAIN_CHARACTERS_LIST])
        ax_bot.set_ylabel('Seconds')
        ax_bot.legend(fontsize=9, ncol=3, loc='upper right')

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    logger.info(f"  multimodal_presence: {output_path}")


# ---------------------------------------------------------------------------
# Season aggregation
# ---------------------------------------------------------------------------

def plot_season_screentime(all_counts, total_seconds_all, output_path, season_label):
    """Aggregate screen time across all episodes in a season."""
    agg = {ch: 0 for ch in MAIN_CHARACTERS_LIST}
    total = 0
    for counts, ts in zip(all_counts, total_seconds_all):
        total += ts
        for ch in MAIN_CHARACTERS_LIST:
            agg[ch] += counts.get(ch, 0)
    plot_screentime(agg, total, output_path, season_label)


def plot_season_cooccurrence(all_matrices, output_path, season_label):
    """Sum co-occurrence matrices across episodes."""
    n = len(MAIN_CHARACTERS_LIST)
    total = np.zeros((n, n), dtype=int)
    for m in all_matrices:
        total += m
    plot_cooccurrence(total, output_path, season_label)


def plot_season_crossmodal(all_summaries, output_path, season_label):
    """Sum crossmodal summaries across episodes."""
    agg = {ch: defaultdict(int) for ch in MAIN_CHARACTERS_LIST}
    for s in all_summaries:
        for ch in MAIN_CHARACTERS_LIST:
            for state, val in s.get(ch, {}).items():
                agg[ch][state] += val
    agg = {ch: dict(v) for ch, v in agg.items()}
    plot_crossmodal(agg, output_path, season_label)


def plot_season_speaking_only_distribution(all_run_data, output_path, season_label):
    """Aggregate speaking_only run-length data across episodes."""
    agg_runs = {ch: [] for ch in MAIN_CHARACTERS_LIST}
    for rd in all_run_data:
        for ch in MAIN_CHARACTERS_LIST:
            agg_runs[ch].extend(rd.get(ch, []))

    # Build a fake multimodal-like structure to reuse the per-episode plotter
    # Instead, just do the plot directly since we have aggregated run lengths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    all_lengths = []
    for ch_lengths in agg_runs.values():
        all_lengths.extend(ch_lengths)

    if all_lengths:
        max_len = max(all_lengths)
        bins = range(1, min(max_len + 2, 60))
        ax1.hist(all_lengths, bins=bins, color='#fdae61', edgecolor='#e08214',
                 align='left')
        ax1.axvline(x=10, color='#e34a33', linestyle='--', linewidth=2,
                    label='QA threshold (10s)')
        if len(all_lengths) > 20:
            ax1.set_yscale('log')
        ax1.legend(fontsize=9)
    ax1.set_xlabel('Run length (seconds)')
    ax1.set_ylabel('Count')
    ax1.set_title('Speaking Only Run Lengths')

    buckets = [('1s', 1, 1), ('2-5s', 2, 5), ('6-10s', 6, 10), ('>10s', 11, 9999)]
    x_pos = np.arange(len(buckets))
    bar_width = 0.12
    offsets = np.linspace(-bar_width * 2.5, bar_width * 2.5, len(MAIN_CHARACTERS_LIST))

    for ci, ch in enumerate(MAIN_CHARACTERS_LIST):
        bucket_counts = []
        for _, lo, hi in buckets:
            bucket_counts.append(sum(1 for l in agg_runs[ch] if lo <= l <= hi))
        ax2.bar(x_pos + offsets[ci], bucket_counts, width=bar_width,
                color=COLORS[ch], label=ch.capitalize())

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([b[0] for b in buckets])
    ax2.set_xlabel('Run duration')
    ax2.set_ylabel('Number of runs')
    ax2.set_title('Runs by Character & Duration')
    ax2.legend(fontsize=8, ncol=2)

    fig.suptitle(f'Speaking Only Distribution — {season_label}', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  season speaking_only_distribution: {output_path}")


def plot_season_multimodal_summary(all_summaries, output_path, season_label):
    """Season-aggregate grouped bar chart of multimodal state totals."""
    agg = {ch: defaultdict(int) for ch in MAIN_CHARACTERS_LIST}
    for s in all_summaries:
        for ch in MAIN_CHARACTERS_LIST:
            for state, val in s.get(ch, {}).items():
                agg[ch][state] += val
    agg = {ch: dict(v) for ch, v in agg.items()}

    with publication_style():
        fig, ax = plt.subplots(figsize=(10, 5))
        x_pos = np.arange(len(MAIN_CHARACTERS_LIST))
        bar_width = 0.2
        for si, (state_name, state_label) in enumerate(
                zip(STATE_ORDER[:3], STATE_LABELS[:3])):
            vals = [agg.get(ch, {}).get(state_name, 0) for ch in MAIN_CHARACTERS_LIST]
            ax.bar(x_pos + (si - 1) * bar_width, vals, width=bar_width,
                   color=STATE_COLORS[state_name], label=state_label)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([ch.capitalize() for ch in MAIN_CHARACTERS_LIST])
        ax.set_ylabel('Seconds')
        ax.set_title(f'Multimodal Summary — {season_label}')
        ax.legend(fontsize=9, ncol=3)
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    logger.info(f"  season multimodal_summary: {output_path}")


# ---------------------------------------------------------------------------
# Episode processing
# ---------------------------------------------------------------------------

def process_episode(episode_id, scratch_dir, plots, output_base):
    """Generate plots for a single episode. Returns aggregation data."""
    logger.info(f"Visualizing: {episode_id}")

    ts_path = utils.get_output_path(
        scratch_dir, utils.OUTPUT_DIR_CHARACTER_TIMESTAMPS,
        f"{episode_id}_timestamps.json")
    fl_path = utils.get_output_path(
        scratch_dir, utils.OUTPUT_DIR_CHARACTER_TIMESTAMPS,
        f"{episode_id}_face_locations.json")
    mm_path = utils.get_output_path(
        scratch_dir, utils.OUTPUT_DIR_MULTIMODAL,
        f"{episode_id}_multimodal.json")

    if not os.path.exists(ts_path):
        logger.error(f"Missing timestamps: {ts_path}")
        return None

    face_seconds, total_seconds = load_face_timestamps(ts_path)

    # Optional multimodal + guest candidates
    multimodal = None
    if os.path.exists(mm_path):
        multimodal = load_multimodal(mm_path)

    gc_path = utils.get_output_path(
        scratch_dir, utils.OUTPUT_DIR_MULTIMODAL,
        f"{episode_id}_guest_candidates.json")
    guest_candidates = None
    if os.path.exists(gc_path):
        guest_candidates = load_guest_candidates(gc_path)
        if guest_candidates:
            logger.info(f"  Guest candidates loaded: {', '.join(g['name'] for g in guest_candidates)}")

    ep_dir = os.path.join(output_base, episode_id)
    os.makedirs(ep_dir, exist_ok=True)

    # Generate requested plots
    if 'timeline' in plots:
        plot_timeline(face_seconds, total_seconds, multimodal,
                      os.path.join(ep_dir, f"{episode_id}_timeline.png"), episode_id,
                      guest_candidates=guest_candidates)

    counts = compute_screentime(face_seconds, total_seconds)
    if 'screentime' in plots:
        plot_screentime(counts, total_seconds,
                        os.path.join(ep_dir, f"{episode_id}_screentime.png"), episode_id)

    matrix = compute_cooccurrence(face_seconds, total_seconds)
    if 'cooccurrence' in plots:
        plot_cooccurrence(matrix,
                          os.path.join(ep_dir, f"{episode_id}_cooccurrence.png"), episode_id)

    if 'confidence' in plots and os.path.exists(fl_path):
        face_locations = load_face_locations(fl_path)
        plot_confidence(face_locations,
                        os.path.join(ep_dir, f"{episode_id}_confidence.png"), episode_id)

    summary = None
    run_data = None
    if multimodal:
        summary = multimodal.get('summary', {})

        if 'crossmodal' in plots:
            plot_crossmodal(summary,
                            os.path.join(ep_dir, f"{episode_id}_crossmodal.png"), episode_id)

        if 'speaking_only_timeline' in plots:
            plot_speaking_only_timeline(multimodal, total_seconds,
                                        os.path.join(ep_dir, f"{episode_id}_speaking_only_timeline.png"),
                                        episode_id)

        if 'modality_agreement' in plots:
            plot_modality_agreement(multimodal, total_seconds,
                                    os.path.join(ep_dir, f"{episode_id}_modality_agreement.png"),
                                    episode_id, guest_candidates=guest_candidates)

        if 'speaking_only_distribution' in plots:
            plot_speaking_only_distribution(multimodal, total_seconds,
                                            os.path.join(ep_dir, f"{episode_id}_speaking_only_distribution.png"),
                                            episode_id)

        if 'multimodal_presence' in plots:
            plot_multimodal_presence(multimodal, total_seconds,
                                     os.path.join(ep_dir, f"{episode_id}_multimodal_presence.png"),
                                     episode_id, guest_candidates=guest_candidates)

        run_data = compute_speaking_only_run_data(multimodal, total_seconds)
    else:
        if any(p in plots for p in ['crossmodal', 'speaking_only_timeline',
               'modality_agreement', 'speaking_only_distribution', 'multimodal_presence']):
            logger.warning("Multimodal data not found — skipping multimodal plots")

    return {
        'counts': counts,
        'total_seconds': total_seconds,
        'matrix': matrix,
        'summary': summary,
        'run_data': run_data,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def list_episodes_for_season(scratch_dir, season):
    ts_dir = utils.get_output_path(scratch_dir, utils.OUTPUT_DIR_CHARACTER_TIMESTAMPS)
    if not os.path.isdir(ts_dir):
        return []
    prefix = f"friends_s{int(season):02d}"
    episodes = []
    for fname in sorted(os.listdir(ts_dir)):
        if fname.startswith(prefix) and fname.endswith('_timestamps.json'):
            episodes.append(fname.replace('_timestamps.json', ''))
    return episodes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QA visualization for char-tracker pipeline')
    parser.add_argument('episode_id', type=str, nargs='?',
                        help='Episode ID (e.g., friends_s01e01a)')
    parser.add_argument('--season', type=int, help='Process all episodes in a season')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Only produce season aggregate plots (skip per-episode)')
    parser.add_argument('--plots', type=str, default=None,
                        help=f'Comma-separated plot types: {",".join(ALL_PLOTS)}')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    if not args.episode_id and args.season is None:
        parser.error('Provide either episode_id or --season')

    logger.setLevel(getattr(logging, args.log_level))
    plots = args.plots.split(',') if args.plots else ALL_PLOTS

    from dotenv import load_dotenv
    load_dotenv()
    scratch_dir = os.getenv('SCRATCH_DIR')
    if not scratch_dir:
        logger.error('SCRATCH_DIR not set')
        sys.exit(1)

    output_base = utils.get_output_path(scratch_dir, utils.OUTPUT_DIR_VISUALIZATION)
    os.makedirs(output_base, exist_ok=True)

    if args.episode_id:
        result = process_episode(args.episode_id, scratch_dir, plots, output_base)
        if result is None:
            sys.exit(1)
    else:
        episodes = list_episodes_for_season(scratch_dir, args.season)
        if not episodes:
            logger.error(f'No episodes found for season {args.season}')
            sys.exit(1)

        logger.info(f"Processing {len(episodes)} episodes for season {args.season}")
        all_counts = []
        all_totals = []
        all_matrices = []
        all_summaries = []
        all_run_data = []

        for ep in episodes:
            try:
                if args.aggregate_only:
                    # Still need to load data for aggregation
                    ts_path = utils.get_output_path(
                        scratch_dir, utils.OUTPUT_DIR_CHARACTER_TIMESTAMPS,
                        f"{ep}_timestamps.json")
                    face_seconds, total_seconds = load_face_timestamps(ts_path)
                    all_counts.append(compute_screentime(face_seconds, total_seconds))
                    all_totals.append(total_seconds)
                    all_matrices.append(compute_cooccurrence(face_seconds, total_seconds))
                    mm_path = utils.get_output_path(
                        scratch_dir, utils.OUTPUT_DIR_MULTIMODAL, f"{ep}_multimodal.json")
                    if os.path.exists(mm_path):
                        mm = load_multimodal(mm_path)
                        all_summaries.append(mm.get('summary', {}))
                        all_run_data.append(
                            compute_speaking_only_run_data(mm, total_seconds))
                else:
                    result = process_episode(ep, scratch_dir, plots, output_base)
                    if result:
                        all_counts.append(result['counts'])
                        all_totals.append(result['total_seconds'])
                        all_matrices.append(result['matrix'])
                        if result['summary']:
                            all_summaries.append(result['summary'])
                        if result['run_data']:
                            all_run_data.append(result['run_data'])
            except Exception as e:
                logger.error(f"Failed on {ep}: {e}", exc_info=True)

        # Season aggregates
        season_label = f"season_{args.season:02d}"
        season_dir = os.path.join(output_base, season_label)
        os.makedirs(season_dir, exist_ok=True)

        if all_counts:
            plot_season_screentime(
                all_counts, all_totals,
                os.path.join(season_dir, f"{season_label}_screentime.png"),
                season_label)
        if all_matrices:
            plot_season_cooccurrence(
                all_matrices,
                os.path.join(season_dir, f"{season_label}_cooccurrence.png"),
                season_label)
        if all_summaries:
            plot_season_crossmodal(
                all_summaries,
                os.path.join(season_dir, f"{season_label}_crossmodal_summary.png"),
                season_label)
            plot_season_multimodal_summary(
                all_summaries,
                os.path.join(season_dir, f"{season_label}_multimodal_summary.png"),
                season_label)
        if all_run_data:
            plot_season_speaking_only_distribution(
                all_run_data,
                os.path.join(season_dir, f"{season_label}_speaking_only_distribution.png"),
                season_label)

        logger.info(f"Season aggregates saved to {season_dir}")
