import os
import sys
import argparse
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import json

sys.path.append("/om2/user/yibei/face-track/src")
from face_clusterer import FaceEmbedder, FaceClusterer

def match_clusters_with_unique_faces(clustered_faces, unique_faces_per_scene):
    matched_results = {}

    for scene_id, unique_faces in unique_faces_per_scene.items():
        matched_results[scene_id] = []

        for unique_face in unique_faces:
            unique_face_id = unique_face['unique_face_id']
            matched_data = {
                "unique_face_id": unique_face_id,
                "global_face_id": unique_face.get('global_face_id', None),  # Global face ID from unique_face
                "cluster_id": None,
                "embeddings": [],  # List to store embeddings
                "image_paths": []  # List to store image paths
            }

            # Find the corresponding cluster
            for cluster_id, faces_in_cluster in clustered_faces.items():
                for face_data in faces_in_cluster:
                    # Match based on unique_face_id directly
                    if face_data['unique_face_id'] == unique_face_id:
                        # Add embedding and image path to the matched data
                        matched_data["embeddings"].append(face_data['embedding'].tolist())
                        matched_data["image_paths"].append(face_data['image_path'])
                        matched_data["cluster_id"] = cluster_id

                if matched_data["cluster_id"] is not None:
                    matched_results[scene_id].append(matched_data)
                    break

    return matched_results

def read_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, output_file):
    """Save the selected frames to a JSON file."""
    def convert_np(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4, default=convert_np)
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")
        sys.exit(1)


def main(video_name, face_selection_file, output_dir):
    face_embedder = FaceEmbedder()
    face_clusterer = FaceClusterer(similarity_threshold=0.6, max_iterations=100)

    selected_faces = read_json(face_selection_file)
    image_dir = os.path.dirname(face_selection_file)

    face_embeddings = face_embedder.get_face_embeddings(selected_faces, image_dir)

    consolidated_clusters = face_clusterer.cluster_faces(face_embeddings)

    matched_faces = match_clusters_with_unique_faces(consolidated_clusters, selected_faces)

    output_file = os.path.join(output_dir, f'{video_name}_matched_faces_with_clusters.json')
    save_json(matched_faces, output_file)

    print("Processing complete!")

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

    face_selection_file = os.path.join(scratch_dir, "output", "face_tracking", f"{video_name}", f"{video_name}_selected_frames_per_face.json")
    output_dir = os.path.join(scratch_dir, "output", "face_clustering")
    os.makedirs(output_dir, exist_ok=True)

    main(video_name, face_selection_file, output_dir)
    