import glob
import os
import sys
from dotenv import load_dotenv
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import argparse

def extract_episode_number(file_path, season_id):
    try:
        # Assuming the episode number directly follows "e" in the filename
        episode_part = file_path.split(f'{season_id}e')[1][:2]
        return int(episode_part)
    except (IndexError, ValueError) as e:
        print(f"Error extracting episode number from {file_path}: {e}")
        return None

def main(input_dir, season_id, save_dir, start_episode=None, end_episode=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    if start_episode is None or end_episode is None:
        all_episodes = sorted({extract_episode_number(f, season_id) for f in glob.glob(os.path.join(input_dir, f'friends_{season_id}e*.jpg')) if extract_episode_number(f, season_id) is not None})
        print("Found episodes:", all_episodes)  # Debugging output

        if not all_episodes:
            print(f"No episodes found in directory {input_dir} with pattern friends_{season_id}e*.jpg")
            sys.exit(1)
        
        start_episode = min(all_episodes)
        end_episode = max(all_episodes) + 1  # Adding 1 to include the last episode
    
    for episode in range(start_episode, end_episode):
        char_images = {char_id: [] for char_id in range(6)}
        for e in range(episode, episode + 3):
            for char_id in range(6):
                pattern = os.path.join(input_dir, f'friends_{season_id}e{e:02d}*char_{char_id}.jpg')
                files = glob.glob(pattern)
                char_images[char_id].extend(files)
        
        image_counts = [len(files) for files in char_images.values()]
        min_images = min(image_counts)
        print(f'Minimum images across 3 episodes for any character: {min_images}')
        
        if min_images == 0:
            print("Skipping processing for this episode window due to 0 images for one or more characters.")
            continue
        
        for char_id, files in char_images.items():
            if len(files) == 0:
                continue
            selected_images = np.random.choice(files, min_images, replace=False)
            embeddings = []

            for img in tqdm(selected_images, desc=f"Processing char_{char_id} embeddings"):
                embedding = get_face_embedding(img, model, device)
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
            filename = os.path.join(save_dir, f'{season_id}_e{episode:02d}-e{episode+2:02d}_char_{char_id}_embeddings.npy')
            np.save(filename, embeddings)
            print(f"Saved {filename}")

def get_face_embedding(image_path, model, device):
    try:
        face_tensor = load_image(image_path).to(device)
        with torch.no_grad():
            embedding = model(face_tensor).cpu().numpy().flatten()
        return embedding
    except Exception as e:
        print(f"Error processing image: {image_path}")
        return np.full((512,), np.nan)

def load_image(image_path):
    """Load an image from disk and convert it to a tensor."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    return preprocess_face(image)

def preprocess_face(face_image):
    """Preprocess the face image for embedding extraction (resize, normalize, etc.)."""
    face_image = cv2.resize(face_image, (160, 160))  # Resize to 160x160 pixels if required
    face_tensor = torch.tensor(face_image).permute(2, 0, 1).float().unsqueeze(0)
    face_tensor = (face_tensor - 127.5) / 128.0  # Normalize
    return face_tensor
        
if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Extract embeddings for all characters in a season.")
    parser.add_argument("season_id", help="Season ID for processing")
    args = parser.parse_args()

    season_id = args.season_id


    required_env_vars = ["BASE_DIR", "SCRATCH_DIR"]

    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")
    output_dir = os.path.join(scratch_dir, "output")
    input_dir = os.path.join(output_dir, "char_face")
    save_dir = os.path.join(output_dir, "char_ref_embs")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    main(input_dir, season_id, save_dir)
