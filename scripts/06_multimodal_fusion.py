#!/usr/bin/env python3
"""
Multimodal fusion: merge face presence (stage 05) with speaker annotations.

Part A: For main characters, produce per-second state (seen_and_speaking,
        seen_only, speaking_only, absent) plus QA flags.
Part B: Guest face enrichment — match unidentified face tracks to guest
        speakers via temporal overlap.

Inputs:
  - 05_character_timestamps/{ep}_timestamps.json (face presence)
  - 04a_face_clustering/{ep}_matched_faces_with_clusters_refined.json
  - 03_face_tracking/{ep}/{ep}_tracked_faces.json
  - $SPEAKER_DIR/s{N}/{ep}_sentence_speaker_table.tsv
"""

import os
import sys
import json
import argparse
import logging
from math import floor
from collections import defaultdict

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils
import constants

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MAIN_CHARACTERS = constants.MAIN_CHARACTERS
MAIN_CHARACTERS_LIST = sorted(MAIN_CHARACTERS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_face_timestamps(path):
    """Load stage-05 timestamps JSON -> {int_second: [char, ...]}."""
    data = read_json(path)
    timestamps = data.get('timestamps', {})
    total_seconds = data.get('metadata', {}).get('total_seconds', 0)
    result = {}
    for sec_str, chars in timestamps.items():
        result[int(sec_str)] = chars
    return result, total_seconds


def load_speaker_timestamps(tsv_path):
    """Load speaker TSV, expand utterances to per-second coverage.

    Returns:
        all_speakers: {int_second: set(speaker_names)} — all speakers
        guest_speakers: {int_second: set(guest_names)} — non-main only
        speaker_list: sorted list of all unique speakers found
    """
    df = pd.read_csv(tsv_path, sep='\t')
    all_speakers = defaultdict(set)
    guest_speakers = defaultdict(set)
    unique_speakers = set()

    for _, row in df.iterrows():
        speaker = str(row.get('speaker', '')).strip().lower()
        if not speaker or speaker == 'nan':
            continue
        start = row.get('start')
        end = row.get('end')
        if pd.isna(start) or pd.isna(end):
            continue

        unique_speakers.add(speaker)
        for sec in range(floor(float(start)), floor(float(end)) + 1):
            all_speakers[sec].add(speaker)
            if speaker not in MAIN_CHARACTERS:
                guest_speakers[sec].add(speaker)

    return dict(all_speakers), dict(guest_speakers), sorted(unique_speakers)


# ---------------------------------------------------------------------------
# Part A — main-character multimodal fusion
# ---------------------------------------------------------------------------

def merge_modalities(face_seconds, speaker_seconds, total_seconds):
    """Merge face and speaker presence into per-second state dict.

    Returns per_second dict keyed by str(second).
    """
    per_second = {}
    for sec in range(total_seconds):
        face_chars = set(face_seconds.get(sec, []))
        speak_chars = speaker_seconds.get(sec, set())
        # restrict to main characters
        speak_main = speak_chars & MAIN_CHARACTERS

        state = {}
        all_chars = face_chars | speak_main
        for ch in all_chars:
            seen = ch in face_chars
            speaking = ch in speak_main
            if seen and speaking:
                state[ch] = 'seen_and_speaking'
            elif seen:
                state[ch] = 'seen_only'
            elif speaking:
                state[ch] = 'speaking_only'

        per_second[str(sec)] = {
            'face': sorted(face_chars),
            'speaker': sorted(speak_main),
            'state': state,
        }
    return per_second


def compute_summary(per_second):
    """Per-character counts of each state."""
    summary = {ch: defaultdict(int) for ch in MAIN_CHARACTERS_LIST}
    total = len(per_second)
    for sec_data in per_second.values():
        present = set(sec_data['state'].keys())
        for ch in MAIN_CHARACTERS_LIST:
            if ch in present:
                summary[ch][sec_data['state'][ch]] += 1
            else:
                summary[ch]['absent'] += 1
    # convert defaultdicts
    return {ch: dict(counts) for ch, counts in summary.items()}


def generate_qa_flags(per_second, total_seconds):
    """Detect QA anomalies across the episode."""
    flags = []

    # Track consecutive speaking-not-seen stretches per character
    speaking_not_seen_run = {ch: 0 for ch in MAIN_CHARACTERS_LIST}
    for sec in range(total_seconds):
        data = per_second.get(str(sec), {})
        state = data.get('state', {})
        for ch in MAIN_CHARACTERS_LIST:
            if state.get(ch) == 'speaking_only':
                speaking_not_seen_run[ch] += 1
            else:
                if speaking_not_seen_run[ch] > 10:
                    flags.append({
                        'type': 'long_speaking_not_seen',
                        'character': ch,
                        'start_second': sec - speaking_not_seen_run[ch],
                        'end_second': sec - 1,
                        'duration': speaking_not_seen_run[ch],
                        'message': f"{ch} speaking without being seen for {speaking_not_seen_run[ch]}s",
                    })
                speaking_not_seen_run[ch] = 0

    # Check remaining runs at end
    for ch in MAIN_CHARACTERS_LIST:
        if speaking_not_seen_run[ch] > 10:
            flags.append({
                'type': 'long_speaking_not_seen',
                'character': ch,
                'start_second': total_seconds - speaking_not_seen_run[ch],
                'end_second': total_seconds - 1,
                'duration': speaking_not_seen_run[ch],
                'message': f"{ch} speaking without being seen for {speaking_not_seen_run[ch]}s",
            })

    # Per-character episode-level flags
    for ch in MAIN_CHARACTERS_LIST:
        face_secs = sum(
            1 for d in per_second.values()
            if ch in [c for c in d.get('face', [])]
        )
        speak_secs = sum(
            1 for d in per_second.values()
            if ch in [c for c in d.get('speaker', [])]
        )
        if speak_secs > 0 and face_secs == 0:
            flags.append({
                'type': 'speaker_never_seen',
                'character': ch,
                'speaking_seconds': speak_secs,
                'message': f"{ch} speaks ({speak_secs}s) but never seen on screen",
            })
        if face_secs > 0 and speak_secs == 0:
            flags.append({
                'type': 'face_never_speaks',
                'character': ch,
                'face_seconds': face_secs,
                'message': f"{ch} seen ({face_secs}s) but never speaks",
            })

    # Seen >60 sec but never speaks (for non-main too, but we only track main)
    for ch in MAIN_CHARACTERS_LIST:
        face_secs = sum(
            1 for d in per_second.values()
            if ch in d.get('face', [])
        )
        speak_secs = sum(
            1 for d in per_second.values()
            if ch in d.get('speaker', [])
        )
        if face_secs > 60 and speak_secs == 0:
            flags.append({
                'type': 'long_seen_not_speaking',
                'character': ch,
                'face_seconds': face_secs,
                'message': f"{ch} visible for {face_secs}s but never speaks in episode",
            })

    return flags


# ---------------------------------------------------------------------------
# Part B — guest face enrichment
# ---------------------------------------------------------------------------

def load_non_main_tracks(refined_path, tracked_path, fps):
    """Load non-main-character face tracks with their time ranges.

    Returns list of dicts: {unique_face_id, cluster_id, seconds: set(int)}
    """
    refined = read_json(refined_path)

    # Build set of non-main unique_face_ids and their cluster_ids
    cluster_info = refined.get('cluster_info', {})
    main_cluster_ids = set()
    for cid, info in cluster_info.items():
        label = info.get('label', '').lower()
        if label in MAIN_CHARACTERS:
            main_cluster_ids.add(cid)

    non_main_faces = {}  # unique_face_id -> cluster_id
    for key, value in refined.items():
        if key in ('metadata', 'cluster_info', 'data'):
            if key == 'data':
                scene_data = value
            else:
                continue
        else:
            scene_data = {key: value}

        if key in ('metadata', 'cluster_info'):
            continue

        items = value if key == 'data' else {key: value}
        for scene_id, faces in (items.items() if isinstance(items, dict) else []):
            if not isinstance(faces, list):
                continue
            for face in faces:
                cid = str(face.get('cluster_id', ''))
                uid = face.get('unique_face_id', '')
                if cid not in main_cluster_ids:
                    non_main_faces[uid] = cid

    # Load tracked faces and compute time ranges for non-main tracks
    tracked = read_json(tracked_path)
    tracks = []
    for scene_id, scene_tracks in tracked.items():
        for track_idx, observations in enumerate(scene_tracks):
            uid = f"{scene_id}_face_{track_idx}"
            if uid not in non_main_faces:
                continue
            frames = [obs['frame'] for obs in observations]
            if not frames:
                continue
            seconds = set()
            for f in frames:
                seconds.add(int(f / fps))
            tracks.append({
                'unique_face_id': uid,
                'cluster_id': non_main_faces[uid],
                'seconds': seconds,
                'min_second': min(seconds),
                'max_second': max(seconds),
            })
    return tracks


def find_guest_candidates(non_main_tracks, guest_speaker_seconds):
    """Match guest speakers to unidentified face tracks via temporal overlap.

    Args:
        non_main_tracks: list of track dicts from load_non_main_tracks
        guest_speaker_seconds: {second: set(guest_names)}

    Returns list of candidate dicts.
    """
    # Invert to per-guest: {guest: set(seconds)}
    guest_to_seconds = defaultdict(set)
    for sec, guests in guest_speaker_seconds.items():
        for g in guests:
            guest_to_seconds[g].add(sec)

    candidates = []
    for guest, g_seconds in sorted(guest_to_seconds.items()):
        for track in non_main_tracks:
            overlap = track['seconds'] & g_seconds
            if not overlap:
                continue

            # Determine confidence
            # Count how many non-main tracks overlap with this guest's speaking window
            competing_tracks = [
                t for t in non_main_tracks
                if t['unique_face_id'] != track['unique_face_id']
                and t['seconds'] & g_seconds
            ]

            overlap_sorted = sorted(overlap)
            track_sorted = sorted(track['seconds'])
            guest_sorted = sorted(g_seconds)

            if len(competing_tracks) == 0 and len(overlap) >= 2:
                confidence = 'high'
                reason = 'sole unidentified face during solo guest speaker turn'
            elif len(competing_tracks) == 0 and len(overlap) == 1:
                confidence = 'medium'
                reason = 'sole unidentified face but only 1s overlap'
            elif len(overlap) < 2:
                confidence = 'low'
                reason = f'short overlap ({len(overlap)}s) with {len(competing_tracks)+1} faces on screen'
            else:
                confidence = 'medium'
                reason = f'{len(competing_tracks)+1} unidentified faces during guest speaker turn'

            candidates.append({
                'guest_speaker': guest,
                'face_track': track['unique_face_id'],
                'current_cluster': track['cluster_id'],
                'overlap_seconds': overlap_sorted,
                'track_seconds': [track_sorted[0], track_sorted[-1]],
                'speaker_seconds': [guest_sorted[0], guest_sorted[-1]],
                'confidence': confidence,
                'reason': reason,
            })

    return candidates


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_multimodal(per_second, summary, qa_flags, metadata, output_dir, episode_id):
    """Save multimodal JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, f"{episode_id}_multimodal.json")
    out = {
        'metadata': metadata,
        'per_second': per_second,
        'summary': summary,
        'qa_flags': qa_flags,
    }
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved multimodal JSON: {json_path}")

    # CSV — flat matrix: second, face_<char>, speaker_<char>, state_<char>
    rows = []
    total = metadata.get('total_seconds', 0)
    for sec in range(total):
        data = per_second.get(str(sec), {})
        row = {'second': sec}
        for ch in MAIN_CHARACTERS_LIST:
            row[f'face_{ch}'] = 1 if ch in data.get('face', []) else 0
            row[f'speaker_{ch}'] = 1 if ch in data.get('speaker', []) else 0
            row[f'state_{ch}'] = data.get('state', {}).get(ch, 'absent')
        rows.append(row)

    csv_path = os.path.join(output_dir, f"{episode_id}_multimodal.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Saved multimodal CSV: {csv_path}")

    return json_path, csv_path


def save_guest_candidates(candidates, guest_speakers_found, non_main_tracks,
                          fps, output_dir, episode_id):
    """Save guest candidate matches JSON."""
    os.makedirs(output_dir, exist_ok=True)

    matched_tracks = {c['face_track'] for c in candidates}
    matched_guests = {c['guest_speaker'] for c in candidates}
    unmatched_guests = sorted(set(guest_speakers_found) - matched_guests)
    unmatched_faces = sorted(
        {t['unique_face_id'] for t in non_main_tracks} - matched_tracks
    )

    out = {
        'metadata': {'episode_id': episode_id, 'fps': fps},
        'candidates': candidates,
        'guest_speakers_found': sorted(guest_speakers_found),
        'unmatched_guests': unmatched_guests,
        'unmatched_faces': unmatched_faces,
    }

    path = os.path.join(output_dir, f"{episode_id}_guest_candidates.json")
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved guest candidates: {path}")
    return path


# ---------------------------------------------------------------------------
# Episode listing
# ---------------------------------------------------------------------------

def list_episodes_for_season(scratch_dir, season):
    """List episode IDs that have stage-05 timestamps for a season."""
    ts_dir = utils.get_output_path(scratch_dir, utils.OUTPUT_DIR_CHARACTER_TIMESTAMPS)
    if not os.path.isdir(ts_dir):
        return []
    prefix = f"friends_s{int(season):02d}"
    episodes = []
    for fname in sorted(os.listdir(ts_dir)):
        if fname.startswith(prefix) and fname.endswith('_timestamps.json'):
            ep_id = fname.replace('_timestamps.json', '')
            episodes.append(ep_id)
    return episodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_video_fps(video_path):
    """Get video FPS, default 30 if unavailable."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            return fps
    except Exception:
        pass
    return 30.0


def process_episode(episode_id, scratch_dir, speaker_dir):
    """Run multimodal fusion for a single episode."""
    logger.info(f"{'=' * 70}")
    logger.info(f"Multimodal Fusion: {episode_id}")
    logger.info(f"{'=' * 70}")

    # Resolve paths
    ts_path = utils.get_output_path(
        scratch_dir, utils.OUTPUT_DIR_CHARACTER_TIMESTAMPS,
        f"{episode_id}_timestamps.json"
    )
    refined_path = utils.get_output_path(
        scratch_dir, utils.OUTPUT_DIR_FACE_CLUSTERING,
        f"{episode_id}_matched_faces_with_clusters_refined.json"
    )
    tracked_path = utils.get_output_path(
        scratch_dir, utils.OUTPUT_DIR_FACE_TRACKING,
        episode_id, f"{episode_id}_tracked_faces.json"
    )
    # Map video episode ID to canonical episode ID for speaker data lookup.
    # Season 1 video files use a different episode ordering than the canonical
    # broadcast order used by speaker annotations (see constants.S1_VIDEO_TO_CANONICAL_EPISODE).
    canonical_id = utils.video_to_canonical_episode_id(episode_id)
    if canonical_id != episode_id:
        logger.info(f"Episode remapped: {episode_id} (video) -> {canonical_id} (canonical/speaker)")
    season = canonical_id.split('_s')[1][:2].lstrip('0')
    tsv_path = os.path.join(speaker_dir, f"s{season}", f"{canonical_id}_sentence_speaker_table.tsv")

    # Check required files
    for label, path in [('timestamps', ts_path), ('speaker TSV', tsv_path)]:
        if not os.path.exists(path):
            logger.error(f"Missing {label}: {path}")
            return False

    # Load data
    face_seconds, total_seconds = load_face_timestamps(ts_path)
    all_speaker_seconds, guest_speaker_seconds, all_speakers = load_speaker_timestamps(tsv_path)

    logger.info(f"Total seconds: {total_seconds}")
    logger.info(f"Speakers found: {', '.join(all_speakers)}")
    guest_list = [s for s in all_speakers if s not in MAIN_CHARACTERS]
    logger.info(f"Guest speakers: {', '.join(guest_list) if guest_list else '(none)'}")

    # Part A: main character fusion
    per_second = merge_modalities(face_seconds, all_speaker_seconds, total_seconds)
    summary = compute_summary(per_second)
    qa_flags = generate_qa_flags(per_second, total_seconds)

    metadata = {
        'episode_id': episode_id,
        'total_seconds': total_seconds,
        'speakers_found': all_speakers,
    }

    output_dir = utils.get_output_path(scratch_dir, utils.OUTPUT_DIR_MULTIMODAL)
    json_path, csv_path = save_multimodal(
        per_second, summary, qa_flags, metadata, output_dir, episode_id
    )

    # Part B: guest face enrichment
    if os.path.exists(refined_path) and os.path.exists(tracked_path):
        video_dir = os.getenv("VIDEO_DIR")
        video_path = utils.get_video_path(video_dir, episode_id)
        fps = get_video_fps(video_path)
        logger.info(f"Video FPS: {fps:.2f}")

        non_main_tracks = load_non_main_tracks(refined_path, tracked_path, fps)
        logger.info(f"Non-main face tracks: {len(non_main_tracks)}")

        candidates = find_guest_candidates(non_main_tracks, guest_speaker_seconds)

        # Add guest_candidate_match flags
        for c in candidates:
            qa_flags.append({
                'type': 'guest_candidate_match',
                'guest_speaker': c['guest_speaker'],
                'face_track': c['face_track'],
                'confidence': c['confidence'],
                'message': f"Guest '{c['guest_speaker']}' may match track {c['face_track']} ({c['confidence']} confidence)",
            })

        # Re-save with updated flags
        save_multimodal(per_second, summary, qa_flags, metadata, output_dir, episode_id)
        save_guest_candidates(
            candidates, guest_list, non_main_tracks, fps, output_dir, episode_id
        )

        logger.info(f"Guest candidates: {len(candidates)} "
                     f"(high: {sum(1 for c in candidates if c['confidence']=='high')}, "
                     f"medium: {sum(1 for c in candidates if c['confidence']=='medium')}, "
                     f"low: {sum(1 for c in candidates if c['confidence']=='low')})")
    else:
        logger.warning("Skipping guest enrichment (missing refined clustering or tracking data)")

    # Print summary
    logger.info(f"\n{'=' * 70}")
    logger.info("MULTIMODAL FUSION SUMMARY")
    logger.info(f"{'=' * 70}")
    for ch in MAIN_CHARACTERS_LIST:
        s = summary.get(ch, {})
        logger.info(
            f"  {ch.capitalize():10s}: "
            f"seen+speaking={s.get('seen_and_speaking', 0):4d}  "
            f"seen_only={s.get('seen_only', 0):4d}  "
            f"speaking_only={s.get('speaking_only', 0):4d}  "
            f"absent={s.get('absent', 0):4d}"
        )
    if qa_flags:
        logger.info(f"\nQA flags ({len(qa_flags)}):")
        for flag in qa_flags:
            logger.info(f"  [{flag['type']}] {flag['message']}")
    logger.info(f"{'=' * 70}")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multimodal fusion: merge face presence with speaker annotations'
    )
    parser.add_argument('episode_id', type=str, nargs='?',
                        help='Episode ID (e.g., friends_s01e01a)')
    parser.add_argument('--season', type=int,
                        help='Process all episodes in a season')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    if not args.episode_id and args.season is None:
        parser.error('Provide either episode_id or --season')

    logger.setLevel(getattr(logging, args.log_level))

    from dotenv import load_dotenv
    load_dotenv()
    scratch_dir = os.getenv('SCRATCH_DIR')
    speaker_dir = os.getenv('SPEAKER_DIR')

    if not scratch_dir:
        logger.error('SCRATCH_DIR not set')
        sys.exit(1)
    if not speaker_dir:
        logger.error('SPEAKER_DIR not set')
        sys.exit(1)

    if args.episode_id:
        ok = process_episode(args.episode_id, scratch_dir, speaker_dir)
        sys.exit(0 if ok else 1)
    else:
        episodes = list_episodes_for_season(scratch_dir, args.season)
        if not episodes:
            logger.error(f'No episodes found for season {args.season}')
            sys.exit(1)
        logger.info(f"Processing {len(episodes)} episodes for season {args.season}")
        failed = []
        for ep in episodes:
            try:
                if not process_episode(ep, scratch_dir, speaker_dir):
                    failed.append(ep)
            except Exception as e:
                logger.error(f"Failed on {ep}: {e}", exc_info=True)
                failed.append(ep)
        if failed:
            logger.error(f"Failed episodes: {', '.join(failed)}")
            sys.exit(1)
        logger.info(f"All {len(episodes)} episodes processed successfully")
