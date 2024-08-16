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
import threading

class FaceDetector:
    def __init__(self, video_path, scene_boundaries, output_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.video_path = video_path
        self.scene_boundaries = scene_boundaries
        self.output_dir = output_dir
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=self.device, factor=0.6)
        self.frame_buffer = []
        self.buffer_lock = threading.Lock()
        self.stop_signal = False
    
    def start_video_stream(self):
        self.vs = FileVideoStream(self.video_path).start()
    
    def stop_video_stream(self):
        self.vs.stop()

    def read_frames(self):
        while not self.stop_signal:
            if len(self.frame_buffer) < 100:  # Keep a buffer of 100 frames
                frame = self.vs.read()
                if frame is None:
                    break
                with self.buffer_lock:
                    self.frame_buffer.append(frame)
            else:
                cv2.waitKey(10)  # Pause briefly to avoid CPU overuse

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
        out = cv2.VideoWriter(annotated_video_path, fourcc, 1, (int(cap.get(3)), int(cap.get(4))))  # Save only 1 frame per second

        for scene_idx, (start_frame, end_frame) in enumerate(self.scene_boundaries):
            start_time = start_frame / fps
            end_time = end_frame / fps

            scene_results = []

            for second in tqdm(range(int(start_time), int(end_time) + 1), desc=f"Processing Scene {scene_idx + 1}/{len(self.scene_boundaries)}"):
                middle_frames = self._get_middle_frames(second, fps)
                faces, annotated_frames = self._detect_faces_in_frames(middle_frames)

                if faces:
                    aggregated_faces = self._aggregate_faces(faces)
                    if aggregated_faces:
                        scene_results.append({
                            "timepoint": second,
                            "relative_index": second - int(start_time),
                            "faces": aggregated_faces
                        })
                        out.write(annotated_frames[0])  # Save the frame related to the final decision
                    else:
                        out.write(annotated_frames[len(annotated_frames)//2])  # Save the middle frame if no consistent face detection
                else:
                    out.write(middle_frames[len(middle_frames)//2])  # Save the middle frame if no faces are detected

            scene_face_detections.append({
                "scene_index": scene_idx,
                "scene_start_time": start_time,
                "scene_end_time": end_time,
                "detections": scene_results
            })

        self.stop_signal = True
        cap.release()
        out.release()
        self.stop_video_stream()
        return scene_face_detections

    def _get_middle_frames(self, second, fps, num_samples=12):
        start_frame = int(second * fps)
        middle_frames = []

        for i in range(0, num_samples, 2):
            frame_idx = start_frame + int(fps / 2) + i - (num_samples // 2)
            frame = self.get_frame_from_buffer()
            if frame is None:
                break
            middle_frames.append(frame)
        
        return middle_frames

    def _detect_faces_in_frames(self, frames):
        all_faces = []
        annotated_frames = []

        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs, landmarks = self.mtcnn.detect(frame_rgb, landmarks=True)

            if boxes is not None:
                faces = [{"box": box.tolist(), "confidence": prob, "landmarks": landmark.tolist()} 
                         for box, prob, landmark in zip(boxes, probs, landmarks)]
                all_faces.append(faces)

                # Annotate frame with bounding boxes and landmarks
                annotated_frame = self._annotate_frame(frame, boxes, landmarks)
                annotated_frames.append(annotated_frame)
        
        return all_faces, annotated_frames
    
    def _annotate_frame(self, frame, boxes, landmarks):
        for box, landmark in zip(boxes, landmarks):
            # Draw bounding box
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # Draw landmarks
            for point in landmark:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
        return frame

    def _aggregate_faces(self, all_faces, iou_threshold=0.8, consistency_threshold=0.9):
        if not all_faces:
            return []
        
        # Flatten the list of face detections across frames
        flattened_faces = [face for faces in all_faces for face in faces]
        face_counts = len(flattened_faces)

        if face_counts == 0:
            return []

        # Calculate IoU or a similar measure for consistency checking
        selected_face = None

        for face in flattened_faces:
            similar_faces = [f for f in flattened_faces if self._iou(face["box"], f["box"]) > iou_threshold]

            if len(similar_faces) / face_counts >= consistency_threshold:
                if not selected_face or face["confidence"] > selected_face["confidence"]:
                    selected_face = face
        
        return [selected_face] if selected_face else []

    def _iou(self, boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def save_results(self, output_file, face_detections):
        with open(output_file, 'w') as f:
            json.dump(face_detections, f, indent=4)

        print(f"Results saved to {output_file}.")

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