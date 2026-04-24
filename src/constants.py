"""
Shared constants for the char-tracker pipeline.

This module contains constants that are used across multiple scripts
to avoid duplication and ensure consistency.
"""

# Main characters in Friends TV show (set for O(1) lookups)
MAIN_CHARACTERS = {'rachel', 'monica', 'chandler', 'joey', 'phoebe', 'ross'}

# Labels to skip during processing (non-human or unclear)
SKIP_LABELS = [
    'dk', 'not_human', 'background', 'unclear', 'junk',
    'not face', 'not clear', 'guest', 'guy on the wheelchair',
    'kid in the hospital', 'random kid'
]

# DK label prefix for unknown/uncertain faces
DK_LABEL_PREFIX = 'dk'

# Quality modifiers for faces (can be combined with other labels)
# These faces are down-weighted during refinement due to poor quality
QUALITY_MODIFIERS = {'@poor', '@profile', '@back', '@dark', '@blurry'}

# Video-to-canonical episode number mapping for season 1.
#
# The video files (and all face-derived data from stages 01-05) use the
# episode numbering from the original video source, which differs from the
# canonical broadcast order used by speaker annotations (te-charnet).
#
# This mapping translates: video episode number → canonical episode number.
# Only season 1 is affected; episodes 7+ and all other seasons are identity.
#
# Example: video file friends_s01e02a contains the content of canonical
#          episode 3, so face data from s01e02 should be fused with speaker
#          data from s01e03.
S1_VIDEO_TO_CANONICAL_EPISODE = {
    1: 1,   # no change
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 2,
}

# Consistent character colors for visualization
CHARACTER_COLORS = {
    'chandler': '#1f77b4',
    'joey':     '#ff7f0e',
    'monica':   '#2ca02c',
    'phoebe':   '#d62728',
    'rachel':   '#9467bd',
    'ross':     '#8c564b',
}