import os
import sys
import argparse
from dotenv import load_dotenv
import pandas as pd
import json

sys.path.append("/om2/user/yibei/face-track/src")
from face_clusterer import FaceCropper, FaceEmbedder, FaceClusterer

def match_clusters_with_unique_faces(clustered_faces, unique_faces_per_scene):
    matched_results = {}

    for scene_id, unique_faces in unique_faces_per_scene.items():
        matched_results[scene_id] = []

        for unique_face in unique_faces:
            unique_face_id = unique_face['unique_face_id']
            scene_face_key = (scene_id, unique_face_id)  # Composite key for matching
            
            # Find the corresponding cluster
            for cluster_id, faces_in_cluster in clustered_faces.items():
                for face_data in faces_in_cluster:
                    # Match based on composite key
                    if (face_data['scene_id'], face_data['unique_face_id']) == scene_face_key:
                        # Found a match, add to results
                        matched_results[scene_id].append({
                            "unique_face_id": unique_face_id,
                            "global_face_id": face_data['global_face_id'],
                            "cluster_id": cluster_id,
                            "embeddings": face_data['embeddings'] 
                        })
                        break

    return matched_results

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save2json(data, output_file):
    """Save the selected frames to a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")
        sys.exit(1)

def main(video_file, face_selection_file, output_dir):
    face_cropper = FaceCropper(video_file=video_file)
    face_embedder = FaceEmbedder()
    face_clusterer = FaceClusterer(similarity_threshold=0.6, max_iterations=100)

    unique_faces_per_scene = read_json_file(face_selection_file)
    cropped_faces = face_cropper.crop_faces(unique_faces_per_scene)

    face_embeddings = face_embedder.get_face_embeddings(cropped_faces)

    consolidated_clusters = face_clusterer.cluster_faces(face_embeddings)

    matched_faces = match_clusters_with_unique_faces(consolidated_clusters, unique_faces_per_scene)

    with open(os.path.join(output_dir, 'matched_faces_with_clusters.json'), 'w') as f:
        json.dump(matched_faces, f, indent=4)

    print("Processing complete!")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Face Detection in Video')
    # parser.add_argument('video_name', type=str, help='Name of the input video file without extension.')  # Clarified the argument description
    
    # args = parser.parse_args()
    # video_name = args.video_name

    load_dotenv()
    base_dir = os.getenv("BASE_DIR")
    if base_dir is None: 
        print("Error: SCRATCH_DIR environment variable is not set.")
        sys.exit(1)

    video_file = os.path.join(base_dir, "data", "friends_s01e01b.mp4")
    face_selection_file = os.path.join(base_dir, "output", "friends_s01e01b_selected_frames_per_face.json")
    output_dir = os.path.join(base_dir, "output", "face_clustering")

    os.makedirs(output_dir, exist_ok=True)
    main(video_file, face_selection_file, output_dir)
    