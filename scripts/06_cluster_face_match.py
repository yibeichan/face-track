import os
import glob
import json
import numpy as np
from dotenv import load_dotenv
import argparse
import re
from collections import Counter

def extract_season_episode(file_name):
    # Regular expression to capture 'sXX' and 'eXX' patterns
    match = re.search(r's(\d{2})e(\d{2})', file_name)
    
    if match:
        s_id = f"s{match.group(1)}" 
        e_id = int(match.group(2))
        return s_id, e_id
    else:
        raise ValueError(f"Filename '{file_name}' does not contain a valid season and episode format")

def get_episode_ranges(episode, total_episodes):
    if episode == 1:
        # Special case for the first episode, we can't look back
        return f"e01-e03"
    elif episode == total_episodes:
        # Special case for the last episode, we can't look forward
        return f"e{total_episodes-2:02d}-e{total_episodes:02d}"
    else:
        # General case for most episodes
        return f"e{episode-1:02d}-e{episode+1:02d}"

def reorganize_by_cluster(clustered_faces):
    clusters = {}
    for scene_id, faces in clustered_faces.items():
        for face_data in faces:
            cluster_id = face_data['cluster_id']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(face_data)
    return clusters

def load_reference_embeddings(season, episode_range, ref_emb_dir):
    embeddings_path = os.path.join(ref_emb_dir, f"{season}_{episode_range}_char_*_embeddings.npy")
    embeddings_files = glob.glob(embeddings_path)
    embeddings = {}
    for file in embeddings_files:
        char_id = os.path.basename(file).split('_char_')[1].split('_')[0]
        embeddings[char_id] = np.load(file)
    return embeddings

def identify_character(cluster_embeddings, character_dict, threshold):
    character_scores = {char_id: 0 for char_id in character_dict.keys()}
    assigned_characters = []

    for cluster_embedding in cluster_embeddings:
        cluster_embedding = np.array(cluster_embedding).flatten()
        best_similarity = -1
        best_char_id = "unknown"

        # Calculate similarity with each character's embeddings
        for char_id, char_embeddings in character_dict.items():
            for ref_embedding in char_embeddings:
                ref_embedding = np.array(ref_embedding).flatten()
                
                # Calculate cosine similarity
                cosine_sim = np.dot(cluster_embedding, ref_embedding) / (np.linalg.norm(cluster_embedding) * np.linalg.norm(ref_embedding))
                
                if cosine_sim > threshold and cosine_sim > best_similarity:
                    best_similarity = cosine_sim
                    best_char_id = char_id

        assigned_characters.append(best_char_id)
        if best_char_id != "unknown":
            character_scores[best_char_id] += 1

    # Determine the most frequently assigned character
    character_count = Counter(assigned_characters)
    most_common_char, count = character_count.most_common(1)[0]

    # Check if the most assigned character appears more than half the time
    if most_common_char != "unknown" and count > len(cluster_embeddings) / 2:
        return most_common_char
    else:
        return "unknown"
    
def main(episode_id, clustered_faces, ref_emb_dir, output_dir):
    clusters = reorganize_by_cluster(clustered_faces)
    
    threshold = 0.6

    season, episode = extract_season_episode(episode_id)
    episode_range = get_episode_ranges(episode, total_episodes=24)
    reference_embeddings = load_reference_embeddings(season, episode_range, ref_emb_dir)

    cluster_char_assignments = {}
    for cluster_id, faces in clusters.items():

        cluster_embeddings = [embedding for face in faces for embedding in face['embeddings']]
        best_character = identify_character(cluster_embeddings, reference_embeddings, threshold)
        print(f"Cluster {cluster_id}: Best Character = {best_character}, with {len(faces)} faces")
        cluster_char_assignments[cluster_id] = best_character

    with open(os.path.join(output_dir, f"{episode_id}_cluster-face_matching.json"), 'w') as f:
        json.dump(cluster_char_assignments, f)

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Match characters in an episode.")
    parser.add_argument("episode_id", help="Episode ID for processing")
    args = parser.parse_args()
    episode_id = args.episode_id

    scratch_dir = os.getenv("SCRATCH_DIR")
    cluster_file = os.path.join(scratch_dir, "output", "face_clustering", f"{episode_id}_matched_faces_with_clusters.json")
    ref_emb_dir = os.path.join(scratch_dir, "output", "char_ref_embs")
    output_dir = os.path.join(scratch_dir, "output", "cluster_face_matching")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(cluster_file, 'r') as f:
        clustered_data = json.load(f)

    main(episode_id, clustered_data, ref_emb_dir, output_dir)
    