import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceTracker:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        box1 = torch.tensor(box1, device=device, dtype=torch.float32)
        box2 = torch.tensor(box2, device=device, dtype=torch.float32)

        x1_inter = torch.max(box1[0], box2[0])
        y1_inter = torch.max(box1[1], box2[1])
        x2_inter = torch.min(box1[2], box2[2])
        y2_inter = torch.min(box1[3], box2[3])

        inter_area = torch.max(torch.tensor(0.0, device=device), x2_inter - x1_inter) * \
                     torch.max(torch.tensor(0.0, device=device), y2_inter - y1_inter)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou.item()

    def track_faces(self, face_data, min_faces_per_cluster):
        """Track faces within a scene based on IoU threshold."""
        clusters = []

        for frame_number, face, conf in face_data:
            face_added = False

            for cluster in clusters:
                last_face_in_cluster = cluster[-1]["face"]
                iou = self.calculate_iou(last_face_in_cluster, face)

                if iou > self.iou_threshold:
                    cluster.append({"frame": frame_number, "face": face, "conf": conf})
                    face_added = True
                    break

            if not face_added:
                clusters.append([{"frame": frame_number, "face": face, "conf": conf}])

        return [cluster for cluster in clusters if len(cluster) > min_faces_per_cluster]

    def track_faces_across_scenes(self, scene_data, face_data):
        """Track faces across all scenes in a video."""
        all_tracked_faces = {}

        # Iterate over each scene
        for index, row in tqdm(scene_data.iterrows(), total=scene_data.shape[0], desc="Tracking Faces Across Scenes"):
            frame_start, frame_end = int(row["Start Frame"]), int(row["End Frame"])
            scene_name = f"scene_{index + 1}"

            n_frames = frame_end - frame_start + 1
            min_faces_per_cluster = max(n_frames // 2, 30)  # 30 is FPS

            face_data_for_scene = []
            
            for i in range(frame_start, frame_end):
                faces = face_data[i]["detections"]
                if len(faces) != 0:
                    for f in faces:
                        face_data_for_scene.append((i, f["box"], f["confidence"]))

            tracked_faces = self.track_faces(face_data_for_scene, min_faces_per_cluster)
            all_tracked_faces[scene_name] = tracked_faces

        return all_tracked_faces

class FrameSelector:
    def __init__(self, video_file, top_n=3):
        self.video_file = video_file
        self.top_n = top_n

    @staticmethod
    def calculate_brightness(image):
        """Calculate the brightness of an image using GPU if available."""
        image_tensor = torch.tensor(image, device=device, dtype=torch.float32)
        return torch.mean(image_tensor).item()

    @staticmethod
    def calculate_blurriness(image):
        """Calculate the blurriness of an image using GPU if available."""
        image_tensor = torch.tensor(image, device=device, dtype=torch.float32)
        laplacian = torch.tensor(cv2.Laplacian(image, cv2.CV_32F), device=device)
        return torch.var(laplacian).item()

    def select_top_frames_per_face(self, tracked_data):
        """Select top frames per face based on confidence, size, brightness, and blurriness."""
        cap = cv2.VideoCapture(self.video_file)
        selected_frames = {}
        global_face_id = 0

        total_faces = sum(len(faces) for faces in tracked_data.values())

        with tqdm(total=total_faces, desc="Processing Faces") as pbar:
            for scene_name, faces in tracked_data.items():
                selected_frames[scene_name] = []
                
                for face_id, face_group in enumerate(faces):
                    frame_scores = []

                    for entry in face_group:
                        frame_idx = entry['frame']
                        face_coords = entry['face']
                        confidence = entry['conf']

                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            print(f"Warning: Could not read frame {frame_idx}. Skipping.")
                            continue

                        height, width, _ = frame.shape
                        x1, y1, x2, y2 = map(int, face_coords)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)

                        face_image = frame[y1:y2, x1:x2]
                        if face_image.size == 0:
                            print(f"Warning: Face image is empty for frame {frame_idx}. Skipping.")
                            continue

                        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                        face_size = (x2 - x1) * (y2 - y1)
                        brightness = self.calculate_brightness(gray_face)
                        blurriness = self.calculate_blurriness(gray_face)

                        # Normalize the components
                        normalized_face_size = face_size / (width * height)
                        normalized_brightness = brightness / 255.0
                        normalized_blurriness = blurriness / (blurriness + 1e-6)  # normalize blurriness itself

                        # Combine features into a score
                        score = confidence + 0.5 * normalized_face_size + 0.3 * normalized_brightness - 0.2 * normalized_blurriness

                        frame_scores.append({
                            "frame_idx": frame_idx,
                            "total_score": score,
                            "face_coord": face_coords
                        })

                    if frame_scores:
                        top_frames = sorted(frame_scores, key=lambda x: x["total_score"], reverse=True)[:self.top_n]

                        unique_face_id = f"{scene_name}_face_{face_id}"
                        global_unique_face_id = f"global_face_{global_face_id}"
                        selected_frames[scene_name].append({
                            "unique_face_id": unique_face_id,
                            "global_face_id": global_unique_face_id,
                            "top_frames": top_frames
                        })

                    global_face_id += 1
                    pbar.update(1)

        cap.release()
        return selected_frames
