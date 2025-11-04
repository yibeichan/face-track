import sys
from pathlib import Path

import numpy as np

# Ensure the source module is importable when running pytest from repo root.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from face_tracker import FaceTracker  # noqa: E402


def test_track_faces_basic_tracking():
    """Test that faces in the same position are tracked together."""
    tracker = FaceTracker(iou_threshold=0.5, max_gap=2, box_expansion=0.1)

    face_sequence = [
        (0, [0.0, 0.0, 10.0, 10.0], 0.9),
        (1, [1.0, 0.0, 11.0, 10.0], 0.85),  # Small movement, should link
    ]

    tracks = tracker.track_faces(face_sequence, min_faces_per_cluster=0)
    assert len(tracks) == 1
    assert len(tracks[0]) == 2


def test_track_faces_best_match_assignment():
    """Test that best-match assignment picks the highest IoU track."""
    tracker = FaceTracker(iou_threshold=0.3, max_gap=2, box_expansion=0.1)

    # Create two tracks, then a face that overlaps both
    face_sequence = [
        (0, [0.0, 0.0, 10.0, 10.0], 0.9),      # Track 1
        (0, [20.0, 0.0, 30.0, 10.0], 0.9),     # Track 2
        (1, [5.0, 0.0, 15.0, 10.0], 0.85),     # Overlaps both, closer to Track 1
    ]

    tracks = tracker.track_faces(face_sequence, min_faces_per_cluster=0)

    # Should assign to track 1 (better overlap)
    assert len(tracks) == 2
    assert len(tracks[0]) == 2  # Track 1 got the new detection
    assert len(tracks[1]) == 1  # Track 2 didn't


def test_track_faces_respects_max_gap():
    """Test that tracks are not linked if gap exceeds max_gap."""
    tracker = FaceTracker(
        iou_threshold=0.5,
        max_gap=1,  # Only allow 1 missing frame
        box_expansion=0.1
    )

    face_sequence = [
        (0, [0.0, 0.0, 10.0, 10.0], 0.9),
        (3, [0.0, 0.0, 10.0, 10.0], 0.85),  # Gap of 3 frames -> new track
    ]

    tracks = tracker.track_faces(face_sequence, min_faces_per_cluster=0)
    assert len(tracks) == 2
    assert all(len(track) == 1 for track in tracks)


def test_track_faces_median_box_reference():
    """Test that median box is used when enabled with enough observations."""
    tracker = FaceTracker(iou_threshold=0.5, max_gap=2, use_median_box=True)

    # Create a track with jittery detections
    face_sequence = [
        (0, [0.0, 0.0, 10.0, 10.0], 0.9),
        (1, [1.0, 0.0, 11.0, 10.0], 0.9),
        (2, [0.5, 0.0, 10.5, 10.0], 0.9),
        (3, [2.0, 0.0, 12.0, 10.0], 0.9),  # Outlier
        (4, [0.8, 0.0, 10.8, 10.0], 0.9),
    ]

    tracks = tracker.track_faces(face_sequence, min_faces_per_cluster=0)

    # All should be in one track (median handles jitter)
    assert len(tracks) == 1
    assert len(tracks[0]) == 5


def test_track_faces_min_cluster_length_filter():
    """Test that short tracks are filtered out."""
    tracker = FaceTracker(iou_threshold=0.5, max_gap=2)

    face_sequence = [
        (0, [0.0, 0.0, 10.0, 10.0], 0.9),
        (0, [100.0, 100.0, 110.0, 110.0], 0.8),  # Different face (no overlap)
        (1, [1.0, 0.0, 11.0, 10.0], 0.88),
    ]

    tracks = tracker.track_faces(face_sequence, min_faces_per_cluster=1)

    # Only the 2-observation track should remain
    assert len(tracks) == 1
    assert len(tracks[0]) == 2


def test_track_faces_output_format():
    """Test that output format is correct (no smoothed_face)."""
    tracker = FaceTracker(iou_threshold=0.5, max_gap=2)

    face_sequence = [
        (0, [0.0, 0.0, 10.0, 10.0], 0.9),
        (1, [1.0, 0.0, 11.0, 10.0], 0.85),
    ]

    tracks = tracker.track_faces(face_sequence, min_faces_per_cluster=0)

    # Check output structure
    assert len(tracks) == 1
    obs = tracks[0][0]

    # Should have these fields
    assert "frame" in obs
    assert "face" in obs
    assert "conf" in obs

    # Should NOT have smoothed_face
    assert "smoothed_face" not in obs


def test_track_faces_different_positions():
    """Test that faces in different positions create separate tracks."""
    tracker = FaceTracker(iou_threshold=0.5, max_gap=2)

    face_sequence = [
        (0, [0.0, 0.0, 10.0, 10.0], 0.9),
        (0, [50.0, 50.0, 60.0, 60.0], 0.85),  # Different position
        (1, [1.0, 0.0, 11.0, 10.0], 0.88),
        (1, [51.0, 50.0, 61.0, 60.0], 0.82),
    ]

    tracks = tracker.track_faces(face_sequence, min_faces_per_cluster=0)

    # Should create 2 tracks
    assert len(tracks) == 2
    assert all(len(track) == 2 for track in tracks)


def test_track_faces_no_median_box():
    """Test that disabling median box uses most recent detection."""
    tracker = FaceTracker(iou_threshold=0.5, max_gap=2, use_median_box=False)

    face_sequence = [
        (0, [0.0, 0.0, 10.0, 10.0], 0.9),
        (1, [1.0, 0.0, 11.0, 10.0], 0.9),
        (2, [2.0, 0.0, 12.0, 10.0], 0.9),
    ]

    tracks = tracker.track_faces(face_sequence, min_faces_per_cluster=0)

    # Should still track correctly
    assert len(tracks) == 1
    assert len(tracks[0]) == 3
