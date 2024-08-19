import os
import cv2
import numpy as np
from collections import defaultdict
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
from tqdm import tqdm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

def _iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
    
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y2_p + 1)
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def _select_best_frame(frames, all_faces, iou_threshold=0.75, consistency_threshold=0.9):
    """Select the best frames based on consistency and confidence of face detections."""
    
    # Flatten the list of faces
    flattened_faces = [face for faces in all_faces for face in faces]
    face_counts = len(flattened_faces)

    if face_counts == 0:
        return frames[len(frames) // 2], []

    # Create clusters of faces based on IoU
    face_clusters = defaultdict(list)
    cluster_id = 0

    for face in flattened_faces:
        matched_cluster = None
        for cid, cluster in face_clusters.items():
            if _iou(face["xyxy"], cluster[0]["xyxy"]) > iou_threshold:
                matched_cluster = cid
                break
        
        if matched_cluster is not None:
            face_clusters[matched_cluster].append(face)
        else:
            face_clusters[cluster_id].append(face)
            cluster_id += 1

    aggregated_faces = []
    best_frame_idx = 0
    best_consistency_score = 0

    # Process each cluster independently
    for cluster_id, cluster in face_clusters.items():
        best_cluster_frame_idx = 0
        best_cluster_consistency_score = 0

        # Evaluate consistency of the current cluster across frames
        for i, frame_faces in enumerate(all_faces):
            consistency_score = 0
            for face in frame_faces:
                if _iou(face["xyxy"], cluster[0]["xyxy"]) > iou_threshold:
                    consistency_score += 1
            
            if consistency_score > best_cluster_consistency_score:
                best_cluster_consistency_score = consistency_score
                best_cluster_frame_idx = i

        # Find the best face within the cluster
        best_face = max(cluster, key=lambda x: x["conf"])
        if len(cluster) / face_counts >= consistency_threshold:
            aggregated_faces.append(best_face)

        # Update the overall best frame if this cluster is more consistent
        if best_cluster_consistency_score > best_consistency_score:
            best_consistency_score = best_cluster_consistency_score
            best_frame_idx = best_cluster_frame_idx

    return frames[best_frame_idx], aggregated_faces

def extract_face_embeddings(scene_file, face_detection_file, video_file, output_dir):
    with open(scene_file, 'r') as f:
        scenes = [line.strip().split(',') for line in f.readlines()[1:]]  # Skips the first header line

    with open(face_detection_file, 'r') as f:
        face_detections = eval(f.read())

    # Open the video file once
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    scene_results = []

    # Add a progress bar for processing scenes
    for scene_idx, scene in tqdm(enumerate(scenes), total=len(scenes), desc="Processing Scenes"):
        start_frame, end_frame = int(scene[0]), int(scene[1])
        scene_start_time, scene_end_time = float(scene[2]), float(scene[3])
        scene_frames = list(range(start_frame + 1, end_frame + 2))  # Adjusting frame IDs
        chunks = [scene_frames[i:i + int(frame_rate)] for i in range(0, len(scene_frames), int(frame_rate))]

        scene_data = {
            "scene_index": scene_idx,
            "scene_start_time": scene_start_time,
            "scene_end_time": scene_end_time,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "detections": []
        }

        for chunk in chunks:
            # Read and store frames in memory
            frames = []
            for frame_idx in chunk:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)  # Frame index in OpenCV starts from 0
                ret, frame = cap.read()
                if not ret:
                    print(f"Frame {frame_idx} could not be read. Skipping...")
                    continue
                frames.append((frame_idx, frame))
            
            # Process the stored frames
            chunk_faces = [face_detections.get(f"frame_{i}", []) for i in chunk]
            best_frame_idx, best_faces = _select_best_frame([f[0] for f in frames], chunk_faces)
            best_frame = next(f[1] for f in frames if f[0] == best_frame_idx)

            frame_detections = []
            for face in best_faces:
                # Crop the face from the best frame
                cropped_face = crop_face(best_frame, face["xyxy"])
                preprocessed_face = preprocess_image(cropped_face)
                embedding = model(preprocessed_face).detach().cpu().numpy().flatten()
                face["embedding"] = embedding

                frame_detections.append({
                    "conf": face["conf"],
                    "xyxy": face["xyxy"],
                    "embedding": embedding.tolist()  # Convert numpy array to list for JSON serialization
                })

            scene_data["detections"].append({
                "frame_idx": best_frame_idx,
                "faces": frame_detections
            })

        scene_results.append(scene_data)
    
    cap.release()

    # Save the scene results to a JSON file
    output_file = os.path.join(output_dir, "scene_detection_with_embeddings.json")
    with open(output_file, 'w') as out_f:
        import json
        json.dump(scene_results, out_f, indent=4)

def crop_face(frame, coords):
    """
    Crop the face region from the frame based on bounding box coordinates.
    Returns the cropped face as an image.
    """
    x1, y1, x2, y2 = map(int, coords)
    return frame[y1:y2, x1:x2]

scene_file = "/om2/user/yibei/face-track/output/scene_detection/friends_s01e01b_scenes.txt"
face_detection_file = "/om2/user/yibei/face-track/output/face_detection/friends_s01e01b_detections.json"
video_file = "/om2/user/yibei/face-track/data/friends_s01e01b.mp4"
output_dir = "/om2/user/yibei/face-track/output/"

extract_face_embeddings(scene_file, face_detection_file, video_file, output_dir)
