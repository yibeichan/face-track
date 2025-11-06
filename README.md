# face-track

A comprehensive video face tracking and clustering pipeline for analyzing character appearances in video content.

## Pipeline Overview

The pipeline consists of 5 sequential steps:

1. **Scene Detection** (`01_scene_detection.py`) - Detects scene boundaries in videos
2. **Face Detection** (`02_face_detection.py`) - Detects faces in each frame
3. **Within-Scene Tracking** (`03_within_scene_tracking.py`) - Tracks faces across frames within scenes
4. **Face Clustering** (`04_face_clustering.py`) - Clusters similar faces across the entire video
5. **Reorganize by Cluster** (`04b_reorganize_by_cluster.py`) - Organizes face images by character cluster

## Running the Pipeline

### Single Video Processing

To process a single video through the entire pipeline (steps 01-04b):

```bash
cd scripts
./run_pipeline_01_to_04b.sh <video_name> [--mode copy|move|symlink]
```

**Arguments:**
- `<video_name>`: Name of the video file (without .mp4 extension)
- `--mode`: (Optional) How to organize files in step 04b
  - `copy` (default): Copy files to cluster directories
  - `move`: Move files to cluster directories
  - `symlink`: Create symbolic links to original files

**Example:**
```bash
./run_pipeline_01_to_04b.sh friends_s01e01
./run_pipeline_01_to_04b.sh friends_s01e02 --mode symlink
```

### Batch Processing with SLURM

For processing multiple videos in parallel using SLURM array jobs:

```bash
# Individual steps
sbatch 01_scene_detection.sh
sbatch 02_face_detection.sh
sbatch 03_within_scene_tracking.sh
sbatch 04_face_clustering.sh
```

## Prerequisites

1. **Environment Setup:**
   - Conda environment named `face-track`
   - Required packages specified in `env.yaml`

2. **Environment Variables:**
   - Set `SCRATCH_DIR` in `.env` file pointing to your data directory

3. **Data Structure:**
   ```
   ${SCRATCH_DIR}/
   ├── data/
   │   └── mkv2mp4/          # Input videos (.mp4)
   └── output/
       ├── scene_detection/   # Scene boundary files (.txt)
       ├── face_detection/    # Face detection results (.json)
       ├── face_tracking/     # Tracked faces per video
       ├── face_clustering/   # Clustering results
       └── face_tracking_by_cluster/  # Reorganized by character
   ```

## Pipeline Features

### Error Handling
- Validates input video exists before starting
- Checks output files after each step
- Exits on error to prevent cascading failures
- Colored logging for easy debugging

### Transition Validation
The pipeline script automatically validates:
- Input video file exists
- Scene detection output created
- Face detection output created
- Tracking directory and files created
- Face images saved during tracking
- Clustering output created
- Cluster directories created

### Logging
- **INFO** (Blue): General information
- **SUCCESS** (Green): Successful operations
- **WARNING** (Yellow): Non-critical issues
- **ERROR** (Red): Critical failures

## Individual Step Details

### Step 01: Scene Detection
```bash
python 01_scene_detection.py <video_name> [--detector content|adaptive|hash]
```
Detects scene boundaries using PySceneDetect.

### Step 02: Face Detection
```bash
python 02_face_detection.py <video_name>
```
Detects faces in video frames using a face detection model.

### Step 03: Within-Scene Tracking
```bash
python 03_within_scene_tracking.py <video_name> [options]
```
Options:
- `--iou-threshold`: Minimum IoU for tracking (default: 0.5)
- `--max-gap`: Max missing frames before track ends (default: 2)
- `--box-expansion`: Box expansion ratio (default: 0.1)
- `--use-median-box/--no-median-box`: Use median box for stability

### Step 04: Face Clustering
```bash
python 04_face_clustering.py <video_name>
```
Clusters faces using embeddings and similarity thresholds.

### Step 04b: Reorganize by Cluster
```bash
python 04b_reorganize_by_cluster.py <video_name> [--mode copy|move|symlink]
```
Organizes face images into directories by character cluster.

## Output Files

Each video generates:
- `{video_name}.txt` - Scene boundaries
- `{video_name}.json` - Face detection results
- `{video_name}_tracked_faces.json` - Tracking results
- `{video_name}_selected_frames_per_face.json` - Selected representative frames
- `{video_name}_matched_faces_with_clusters.json` - Clustering results
- `{video_name}/` - Directory with cluster subdirectories

## Troubleshooting

**Pipeline fails at step 03:**
- Check if scenes were detected in step 01
- Verify face detection found faces in step 02

**No cluster directories created:**
- May indicate no faces were successfully tracked
- Check intermediate outputs from steps 02 and 03

**SCRATCH_DIR not set:**
- Create `.env` file with `SCRATCH_DIR=/path/to/your/data`
