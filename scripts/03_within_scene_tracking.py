import os
import sys
import argparse
from dotenv import load_dotenv
import pandas as pd
import json

# Ensure the custom module is in the Python path
sys.path.append("/om2/user/yibei/face-track/src")
from face_tracker import FaceTracker, FrameSelector


def save_selected_frames(selected_frames, output_file):
    """Save the selected frames to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(selected_frames, f, indent=4)

def main(video_name, output_dir):

    scene_file = os.path(scratch_dir, "output", "scene_detection", f"{video_name}.txt")
    face_detection_file = os.path(scratch_dir, "output", "face_detection", f"{video_name}.json")
    video_file = os.path.join(scratch_dir, "data", "mkv2mp4", f"{video_name}.mp4")
    
    scene_data = pd.read_csv(scene_file, sep=",")
    with open(face_detection_file, "r") as f:
        face_data = json.load(f)

    # Initialize the tracker and selector
    face_tracker = FaceTracker(iou_threshold=0.5)
    frame_selector = FrameSelector(video_file=video_file, top_n=3)

    # Track faces across scenes
    tracked_faces = face_tracker.track_faces_across_scenes(scene_data, face_data)

    # Select top frames per face
    selected_frames = frame_selector.select_top_frames_per_face(tracked_data=tracked_faces)

    output_file = os.path.join(output_dir, f"{video_name}_selected_frames_per_face.json")
    save_selected_frames(selected_frames, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Detection in Video')
    parser.add_argument('video_name', type=str, help='Name of the input video file without extension.')  # Clarified the argument description
    
    args = parser.parse_args()
    video_name = args.video_name

    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    if scratch_dir is None: 
        print("Error: SCRATCH_DIR environment variable is not set.")
        sys.exit(1)

    output_dir = os.path.join(scratch_dir, "output", "face_tracking")
    os.makedirs(output_dir, exist_ok=True) 

    main(video_name, output_dir)