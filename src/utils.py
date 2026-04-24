"""
Shared utilities and constants for the char-tracker pipeline.
"""

import os
import re

import constants

# Output directory names (numbered by pipeline stage)
OUTPUT_DIR_SCENE_DETECTION = "01_scene_detection"
OUTPUT_DIR_FACE_DETECTION = "02_face_detection"
OUTPUT_DIR_FACE_TRACKING = "03_face_tracking"
OUTPUT_DIR_FACE_CLUSTERING = "04a_face_clustering"
OUTPUT_DIR_FACE_TRACKING_BY_CLUSTER = "04b_face_tracking_by_cluster"
OUTPUT_DIR_FACE_TRACKING_REFINED = "04c_face_tracking_by_cluster_refined"
OUTPUT_DIR_CHARACTER_TIMESTAMPS = "05_character_timestamps"
OUTPUT_DIR_MULTIMODAL = "06_multimodal"
OUTPUT_DIR_VISUALIZATION = "07_visualization"


def get_video_path(video_dir, episode_id):
    """Resolve episode_id to video file path.

    Maps episode_id (e.g. 'friends_s01e03b') to the correct season subdir
    and .mkv file: video_dir/s1/friends_s01e03b.mkv

    Args:
        video_dir: Root directory containing season subdirs (s1/, s2/, ...).
        episode_id: Episode identifier (e.g. 'friends_s01e03b').

    Returns:
        Full path to the .mkv video file.
    """
    season = episode_id.split('_s')[1][:2].lstrip('0')
    return os.path.join(video_dir, f"s{season}", f"{episode_id}.mkv")


def video_to_canonical_episode_id(episode_id):
    """Map a video-derived episode ID to the canonical (broadcast) episode ID.

    Season 1 video files use a different episode ordering than the canonical
    broadcast order used by speaker annotations. This function translates
    video episode IDs (used by face data, stages 01-05) to canonical episode
    IDs (used by speaker data).

    Only season 1 episodes 1-6 are remapped; all others pass through unchanged.

    Args:
        episode_id: Video-derived episode ID (e.g. 'friends_s01e02a').

    Returns:
        Canonical episode ID (e.g. 'friends_s01e03a').
    """
    m = re.match(r'^(friends_s)(\d{2})e(\d{2})(.*)$', episode_id)
    if not m:
        return episode_id

    prefix, season_str, ep_str, suffix = m.groups()
    season = int(season_str)
    ep_num = int(ep_str)

    if season == 1 and ep_num in constants.S1_VIDEO_TO_CANONICAL_EPISODE:
        canonical_ep = constants.S1_VIDEO_TO_CANONICAL_EPISODE[ep_num]
        return f"{prefix}{season_str}e{canonical_ep:02d}{suffix}"

    return episode_id


def get_output_path(scratch_dir, *parts):
    """Construct an output path under scratch_dir/output/.

    Args:
        scratch_dir: Base scratch directory from environment.
        *parts: Path components to join under output/.

    Returns:
        Full path to the output location.
    """
    return os.path.join(scratch_dir, "output", *parts)
