import os
import glob
import sys
import argparse
import re
import json
from dotenv import load_dotenv
from tqdm import tqdm

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("/om2/user/yibei/face-track/src")
from face_clusterer import FaceEmbedder

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

def find_best_match(face_embeddings, character_embeddings, similarity_threshold=0.5, mean_diff_threshold=0.1):
    """
    Identifies the best matching character for a face's 3 embeddings using similarity distributions.
    
    Parameters:
        face_embeddings (list): List of 3 embeddings for the face.
        character_embeddings (dict): Dictionary of character embeddings.
        similarity_threshold (float): Threshold for cosine similarity to assign a match.
        mean_diff_threshold (float): Threshold for the mean difference between two distributions to skip comparison.
    
    Returns:
        str: Best matching character or 'unknown'.
    """
    character_similarities = {char: [] for char in character_embeddings.keys()}
    
    for character, char_emb in character_embeddings.items():
        best_mean_sim = -np.inf  # To track the best mean similarity for this character
        best_similarities = None  # To store the best set of similarities
        
        # Iterate through the 3 face embeddings
        for face_emb in face_embeddings:
            similarities = cosine_similarity(face_emb, char_emb)  # Shape: (1, N), where N = number of char embeddings
            mean_sim = np.mean(similarities.flatten())
            
            # If this embedding has a higher mean similarity, update the best similarity
            if mean_sim > best_mean_sim:
                best_mean_sim = mean_sim
                best_similarities = similarities.flatten()
        
        # Store the best similarity distribution for the current character
        if best_similarities is not None:
            character_similarities[character] = best_similarities
            print(f"Character: {character}, Best Mean Similarity: {best_mean_sim}")
    
    passing_characters = {char: sims for char, sims in character_similarities.items() 
                          if np.nanmean(sims) > similarity_threshold}
    
    if not passing_characters:
        print("No passing characters found.")
        return '99' 
    
    # Sort characters by the mean of their similarity distributions
    sorted_characters = sorted(passing_characters.items(), key=lambda x: np.mean(x[1]), reverse=True)
    
    # Sequentially compare the top character with the next best using Mann-Whitney U test
    best_character, best_sims = sorted_characters[0]  # Character with the highest mean
    best_mean = np.mean(best_sims)
    
    for next_character, next_sims in sorted_characters[1:]:
        next_mean = np.mean(next_sims)
        
        # If the mean difference is large enough, pick the best character directly
        if best_mean - next_mean > mean_diff_threshold:
            print(f"Mean difference between {best_character} and {next_character} is large enough.")
            return best_character
        
        # Perform Mann-Whitney U test comparing best_sims with the next best
        stat, p_value = mannwhitneyu(best_sims, next_sims, alternative='greater')
        
        # If the best character is statistically "greater", break and assign
        if p_value < 0.05:
            print(f"Statistically significant difference between {best_character} and {next_character}.")
            return best_character
        
        # Otherwise, update the best character and continue comparison
        best_character, best_sims = next_character, next_sims
        best_mean = next_mean
    
    return best_character if best_character else '99'

def main(episode_id, face_selection_file, ref_emb_dir, output_dir):
    # this script skipped the clustering part and directly used the selected faces for face matching
    char_mapping = {
            '0': 'chandler', '1': 'joey', '2': 'monica',
            '3': 'phoebe', '4': 'rachel', '5': 'ross', '99': 'unknown'
        }
    
    face_embedder = FaceEmbedder()

    selected_faces = read_json(face_selection_file)
    image_dir = os.path.dirname(face_selection_file)

    face_info = face_embedder.get_face_embeddings(selected_faces, image_dir)

    season, episode = extract_season_episode(episode_id)
    episode_range = get_episode_ranges(episode, total_episodes=24)
    reference_embeddings = load_reference_embeddings(season, episode_range, ref_emb_dir)

    new_face_info = []
    with tqdm(total=len(face_info), desc="Matching Faces with Characters", unit="face") as pbar:
        for face in face_info:
            print(face['unique_face_id'])
            emb_info = face['embeddings']
            face_embeddings = [emb['embedding'] for emb in emb_info]
            best_character = find_best_match(face_embeddings, reference_embeddings)
            new_face_info.append({
                    "scene_id": face['scene_id'],
                    "unique_face_id": face['unique_face_id'], 
                    "global_face_id": face['global_face_id'],
                    "embeddings": emb_info,
                    "character": char_mapping[best_character]
                })
            pbar.update(1)
    

    output_file = os.path.join(output_dir, f'{episode_id}_matched_faces_with_characters.json')
    save_json(new_face_info, output_file)

    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Detection in Video')
    parser.add_argument("episode_id", help="Episode ID for processing")
    args = parser.parse_args()
    episode_id = args.episode_id

    load_dotenv()
    nese_dir = os.getenv("NESE_DIR")
    if nese_dir is None: 
        print("Error: SCRATCH_DIR environment variable is not set.")
        sys.exit(1)

    face_selection_file = os.path.join(nese_dir, "output", "face_tracking", f"{episode_id}", f"{episode_id}_selected_frames_per_face.json")
    ref_emb_dir = os.path.join(nese_dir, "output", "char_ref_embs")
    output_dir = os.path.join(nese_dir, "output", "face_matching")
    os.makedirs(output_dir, exist_ok=True)

    main(episode_id, face_selection_file, ref_emb_dir, output_dir)