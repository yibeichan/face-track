import os
import glob
import json
import numpy as np
from dotenv import load_dotenv
import argparse
import re

def extract_season_episode(file_name):
    # Regular expression to capture 'sXX' and 'eXX' patterns
    match = re.search(r's(\d{2})e(\d{2})', file_name)
    
    if match:
        s_id = f"s{match.group(1)}"  # Extract the season ID
        e_id = int(match.group(2))   # Extract the episode ID as an integer
        return s_id, e_id
    else:
        raise ValueError(f"Filename '{file_name}' does not contain a valid season and episode format")

def get_episode_ranges(episode, total_episodes):
    if episode <= 1:
        # Special case for the first episode, we can't look back
        return [f"e01-e03"]
    elif episode == 2:
        # Special case for the second episode, we can only look back one episode
        return [f"e01-e03", f"e02-e04"]
    elif episode >= total_episodes:
        # Special case for the last episode, we can't look forward
        return [f"e{total_episodes-2:02d}-e{total_episodes:02d}"]
    elif episode == total_episodes - 1:
        # Special case for the second-to-last episode, we can only look forward one episode
        return [f"e{episode-2:02d}-e{episode:02d}", f"e{episode-1:02d}-e{episode+1:02d}"]
    else:
        # General case for most episodes
        return [
            f"e{episode-2:02d}-e{episode:02d}",
            f"e{episode-1:02d}-e{episode+1:02d}",
            f"e{episode:02d}-e{episode+2:02d}"
        ]

def reorganize_by_cluster(clustered_faces):
    clusters = {}
    for scene_id, faces in clustered_faces.items():
        for face_data in faces:
            cluster_id = face_data['cluster_id']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(face_data)
    return clusters

def select_top_clusters(clusters, top_n=7):
    cluster_sizes = [(cluster_id, len(faces)) for cluster_id, faces in clusters.items()]
    sorted_clusters = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)
    top_clusters = {cluster_id: clusters[cluster_id] for cluster_id, _ in sorted_clusters[:top_n]}
    return top_clusters

def load_reference_embeddings(season, episode_range, ref_emb_dir):
    embeddings_path = os.path.join(ref_emb_dir, f"{season}_{episode_range}_char_*_embeddings.npy")
    embeddings_files = glob.glob(embeddings_path)
    embeddings = {}
    for file in embeddings_files:
        char_id = os.path.basename(file).split('_char_')[1].split('_')[0]
        embeddings[char_id] = np.load(file)
    return embeddings

def compare_with_reference(cluster_embeddings, reference_embeddings):
    similarities = []
    for cluster_embedding in cluster_embeddings:
        cluster_embedding = np.array(cluster_embedding)
        # Compute similarity for each of the 3 embeddings
        for emb in cluster_embedding:
            emb = np.array(emb).flatten()
            cosine_sim = np.dot(reference_embeddings, emb) / (np.linalg.norm(reference_embeddings, axis=1) * np.linalg.norm(emb))
            similarities.append(np.max(cosine_sim))
    
    return np.mean(similarities)

def compare_clusters_with_reference(clusters, season, episode, total_episodes, output_dir):
    episode_ranges = get_episode_ranges(episode, total_episodes)
    results = {}

    for episode_range in episode_ranges:
        reference_embeddings = load_reference_embeddings(season, episode_range, output_dir)
        
        for cluster_id, cluster_data in clusters.items():
            cluster_embeddings = np.array(cluster_data['embeddings'])
            similarities = {char_id: compare_with_reference(cluster_embeddings, ref_embs) for char_id, ref_embs in reference_embeddings.items()}
    
            # Assign the character with the highest similarity
            top_char = max(similarities, key=similarities.get)
            results[cluster_id] = top_char
    return results

def identify_characters(top_clusters, reference_files):
    character_assignments = {}
    for cluster_id, faces in top_clusters.items():
        best_character = None
        best_similarity = -1
        
        cluster_embeddings = [embedding for face in faces for embedding in face['embeddings']]
        
        for ref_file in reference_files:
            reference_embeddings = np.load(ref_file)
            similarity = compare_with_reference(cluster_embeddings, reference_embeddings)
            if similarity > best_similarity:
                best_similarity = similarity
                best_character = ref_file  # Or extract character ID from file name
        
        character_assignments[cluster_id] = (best_character, best_similarity)
    return character_assignments

def main(episode_id, clustered_faces, ref_emb_dir, output_dir):
    # Step 1: Reorganize by cluster
    clusters = reorganize_by_cluster(clustered_faces)
    
    similarity_threshold = 0.6

    # Step 2: Select top clusters
    top_clusters = select_top_clusters(clusters, top_n=7)

    # Step 3: Load relevant reference embeddings
    season, episode = extract_season_episode(episode_id)
    episode_ranges = get_episode_ranges(episode, total_episodes=24)
    reference_embeddings = {}
    for episode_range in episode_ranges:
        ref_embs = load_reference_embeddings(season, episode_range, ref_emb_dir)
        reference_embeddings.update(ref_embs)

    # Step 4: Compare clusters with reference embeddings
    cluster_char_assignments = {}
    for cluster_id, cluster_faces in top_clusters.items():
        cluster_embeddings = [face_data['embeddings'] for face_data in cluster_faces]
        cluster_char_assignments[cluster_id] = {}
        for char_id, ref_embs in reference_embeddings.items():
            similarity_score = compare_with_reference(cluster_embeddings, ref_embs)
            cluster_char_assignments[cluster_id][char_id] = similarity_score

    # Step 5: Assign characters to clusters
    final_assignments = {}
    for cluster_id, similarity_scores in cluster_char_assignments.items():
        best_char_id = max(similarity_scores, key=similarity_scores.get)
        if similarity_scores[best_char_id] > similarity_threshold:
            final_assignments[cluster_id] = best_char_id
        else:
            final_assignments[cluster_id] = "unknown"

    # Step 6: Save results
    with open(os.path.join(output_dir, f"{episode_id}_cluster-face_matching.json"), 'w') as f:
        json.dump(final_assignments, f)

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
    