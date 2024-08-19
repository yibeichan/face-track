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

def main(scene_file, face_detection_file, video_file):
    
    scene_data = pd.read_csv(scene_file, sep=",")
    with open(face_detection_file, "r") as f:
        face_data = json.load(f)

    # Initialize the tracker and selector
    face_tracker = FaceTracker(iou_threshold=0.5)
    frame_selector = FrameSelector(video_file="your_video.mp4", top_n=3)

    # Track faces across scenes
    tracked_faces = face_tracker.track_faces_across_scenes(scene_data, face_data)

    # Select top frames per face
    selected_frames = frame_selector.select_top_frames_per_face(tracked_data=tracked_faces)

    output_file = "/om2/user/yibei/face-track/output/friends_s01e01b_selected_frames_per_face.json"
    save_selected_frames(selected_frames, output_file)

if __name__ == "__main__":
    # 
    scene_file = "/om2/user/yibei/face-track/output/scene_detection/friends_s01e01b_scenes.txt"
    face_detection_file = "/om2/user/yibei/face-track/output/face_detection/friends_s01e01b_face_detections.json"
    video_file = "/om2/user/yibei/face-track/data/friends_s01e01b.mp4"

    main(scene_file, face_detection_file, video_file)