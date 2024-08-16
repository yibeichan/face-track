import cv2
import argparse
import os
from dotenv import load_dotenv
from facenet_pytorch import MTCNN
import torch
import json
import numpy as np
from tqdm import tqdm
from imutils.video import FileVideoStream
from collections import defaultdict
import threading

class FaceDetector:
    def __init__(self, video_path, scene_boundaries, output_dir, device='cuda' if torch.cuda.is_available() else 'cpu', min_confidence=0.8):
        self.video_path = video_path
        self.scene_boundaries = scene_boundaries
        self.output_dir = output_dir
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=self.device, factor=0.6)
        self.frame_buffer = []
        self.buffer_lock = threading.Lock()
        self.stop_signal = False
        self.min_confidence = min_confidence
    
    def start_video_stream(self):
        self.vs = FileVideoStream(self.video_path).start()
    
    def stop_video_stream(self):
        self.vs.stop()

    def read_frames(self):
        while not self.stop_signal:
            if len(self.frame_buffer) < 100:
                frame = self.vs.read()
                if frame is None:
                    break
                with self.buffer_lock:
                    self.frame_buffer.append(frame)
            else:
                cv2.waitKey(10)

    def get_frame_from_buffer(self):
        with self.buffer_lock:
            if self.frame_buffer:
                return self.frame_buffer.pop(0)
        return None

    def detect_faces_in_scene(self):
        self.start_video_stream()
        threading.Thread(target=self.read_frames, daemon=True).start()
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scene_face_detections = []
        annotated_video_path = os.path.join(self.output_dir, f"{os.path.basename(self.video_path).split('.')[0]}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(annotated_video_path, fourcc, int(fps), (int(cap.get(3)), int(cap.get(4))))

        for scene_idx, (start_frame, end_frame) in enumerate(self.scene_boundaries):
            chunk_size = int(fps)  # Process in chunks of 1 second (fps frames)
            for chunk_start in tqdm(range(start_frame, end_frame, chunk_size), desc=f"Processing Scene {scene_idx + 1}/{len(self.scene_boundaries)}"):
                chunk_end = min(chunk_start + chunk_size, end_frame)
                middle_frames, frame_idxs = self._get_middle_frames(chunk_start, chunk_end, fps)

                if middle_frames:
                    faces, _ = self._detect_faces_in_frames(middle_frames)
                    
                    if faces:
                        best_frame, aggregated_faces = self._select_best_frame(middle_frames, faces)
                        representative_frame = self._annotate_frame(best_frame, aggregated_faces)
                    else:
                        representative_frame = middle_frames[len(middle_frames) // 2]
                        representative_frame = self._annotate_frame(representative_frame, [])

                    # Write the annotated representative frame for each frame in the current chunk
                    for _ in range(chunk_end - chunk_start):
                        out.write(representative_frame)

                    scene_face_detections.append({
                        "scene_index": scene_idx,
                        "scene_start_time": chunk_start / fps,
                        "scene_end_time": chunk_end / fps,
                        "start_frame": chunk_start,
                        "end_frame": chunk_end,
                        "detections": aggregated_faces if faces else []
                    })

        self.stop_signal = True
        cap.release()
        out.release()
        self.stop_video_stream()
        return scene_face_detections

    def _get_middle_frames(self, start_frame, end_frame, fps, num_samples=6):
        """Get middle frames within a chunk."""
        middle_frames = []
        frame_idxs = []

        chunk_size = end_frame - start_frame
        step = chunk_size // num_samples if num_samples > 0 else 1
        middle_frame_start = start_frame + (chunk_size - num_samples * step) // 2

        for i in range(num_samples):
            frame_idx = middle_frame_start + i * step
            frame = self.get_frame_from_buffer()
            if frame is None:
                break
            middle_frames.append(frame)
            frame_idxs.append(frame_idx)
        
        return middle_frames, frame_idxs

    def _select_best_frame(self, frames, all_faces, iou_threshold=0.75, consistency_threshold=0.9):
        """Select the best frame based on consistency and confidence of face detections."""
        flattened_faces = [face for faces in all_faces for face in faces]
        face_counts = len(flattened_faces)

        if face_counts == 0:
            return frames[len(frames) // 2], []

        face_clusters = defaultdict(list)
        cluster_id = 0

        for i, face in enumerate(flattened_faces):
            matched_cluster = None

            for cid, cluster in face_clusters.items():
                if self._iou(face["box"], cluster[0]["box"]) > iou_threshold:
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

        for i, frame_faces in enumerate(all_faces):
            consistency_score = 0
            for face in frame_faces:
                for cluster in face_clusters.values():
                    if self._iou(face["box"], cluster[0]["box"]) > iou_threshold:
                        consistency_score += 1
            
            if consistency_score > best_consistency_score:
                best_consistency_score = consistency_score
                best_frame_idx = i

        # Aggregate the faces from the best frame's clusters
        for cid, cluster in face_clusters.items():
            if len(cluster) / face_counts >= consistency_threshold:
                best_face = max(cluster, key=lambda x: x["confidence"])
                aggregated_faces.append(best_face)

        return frames[best_frame_idx], aggregated_faces

    def _detect_faces_in_frames(self, frames):
        all_faces = []
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

        for frame_rgb in frames_rgb:
            boxes, probs, landmarks = self.mtcnn.detect(frame_rgb, landmarks=True)
            faces = []
            if boxes is not None:
                for box, prob, landmark in zip(boxes, probs, landmarks):
                    if prob >= self.min_confidence and self._is_valid_box(box):
                        faces.append({
                            "box": box.tolist(), 
                            "confidence": prob, 
                            "landmarks": landmark.tolist()
                        })
            all_faces.append(faces)
        
        return all_faces, frames
    
    def _annotate_frame(self, frame, faces, rect_color=(0, 255, 0), circle_color=(0, 0, 255), text_color=(255, 255, 255)):
        for face in faces:
            box = face['box']
            landmarks = face['landmarks']
            confidence = face['confidence']
            
            # Draw the bounding box
            cv2.rectangle(frame, 
                          (int(box[0]), int(box[1])), 
                          (int(box[2]), int(box[3])), 
                          rect_color, 2)
            
            # Draw the landmarks
            for point in landmarks:
                cv2.circle(frame, 
                           (int(point[0]), int(point[1])), 
                           2, 
                           circle_color, -1)
            
            # Display the confidence level
            cv2.putText(frame, 
                        f'{confidence:.2f}', 
                        (int(box[0]), int(box[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        text_color, 2)
        return frame

    def _is_valid_box(self, box, min_size=20, max_aspect_ratio=1.5):
        width = box[2] - box[0]
        height = box[3] - box[1]
        aspect_ratio = max(width / height, height / width)
        return width > min_size and height > min_size and aspect_ratio <= max_aspect_ratio

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxB[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def save_results(self, output_file, face_detections):
        with open(output_file, 'w') as f:
            json.dump(face_detections, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Detection in Video Scenes')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    
    args = parser.parse_args()

    video_path = args.video_path

    load_dotenv()
    base_dir = os.getenv("BASE_DIR")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    input_filename = os.path.basename(video_path).split('.')[0]
    scene_file = os.path.join(output_dir, "scene_detection", f"{input_filename}_scenes.txt")
    
    scene_boundaries = []
    with open(scene_file, 'r') as f:
        for line in f.readlines()[1:]:  # Skip the header
            start_frame, end_frame = map(int, line.strip().split(',')[:2])
            scene_boundaries.append((start_frame, end_frame))
            

    output_subdir = os.path.join(output_dir, "face_detection")
    os.makedirs(output_subdir, exist_ok=True)

    # Initialize FaceDetector and detect faces
    face_detector = FaceDetector(video_path, scene_boundaries, output_subdir)
    face_detections = face_detector.detect_faces_in_scene()

    # Save the face detection results
    output_file = os.path.join(output_subdir, f"{input_filename}_face_detections.json")
    face_detector.save_results(output_file, face_detections)

    print(f"Face detection completed. Results saved to {output_file}")
