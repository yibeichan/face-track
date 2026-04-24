# char-tracker

A comprehensive video face tracking and clustering pipeline for analyzing character appearances in TV episodes (Friends dataset). The system uses an annotation-driven workflow where manual corrections via ClusterMark feed into automated refinement.

## Pipeline Overview

The pipeline consists of sequential stages where each stage depends on outputs from previous stages:

| Stage | Script | Description |
|-------|--------|-------------|
| 01 | `01_scene_detection.py` | Segments video into scenes using PySceneDetect |
| 02 | `02_face_detection.py` | Detects faces in all frames using MTCNN |
| 03 | `03_within_scene_tracking.py` | Tracks faces across frames within scenes, selects representative frames |
| 04a | `04a_face_clustering.py` | Clusters similar faces across scenes using embeddings |
| 04b | `04b_reorganize_by_cluster.py` | Reorganizes face images into cluster directories for annotation |
|  —  | **ClusterMark (manual)** | Human annotator labels each cluster with a character name; exports JSON |
| 04c | `04c_refine_with_annotations.py` | Propagates manual labels, merges/splits clusters (post-annotation) |
| 05 | `05_generate_character_timestamps.py` | Generates per-second character presence timestamps |

Downstream multimodal fusion (face + speaker) and visualization live in the
te-charnet repository, which consumes stage-05 outputs.

## Running the Pipeline

### Initial Pipeline Run (Stages 01-04b)

Process a single video through the initial clustering pipeline:

```bash
cd scripts
./run_pipeline_01_to_04b.sh <episode_id> [--mode copy|move|symlink]
```

**Arguments:**
- `<episode_id>`: Episode identifier in format `friends_s<season>e<episode>` (e.g., `friends_s01e01b`)
- `--mode`: (Optional) How to organize files in step 04b
  - `copy` (default): Copy files to cluster directories
  - `move`: Move files to cluster directories
  - `symlink`: Create symbolic links to original files

**Additional options:**
- `--no-sequential`: Disable optimized sequential frame reading (slower legacy mode)
- `--no-batch`: Disable batch embedding processing (slower sequential mode)
- `--similarity-threshold N`: Set clustering similarity threshold (default: 0.5)

**Example:**
```bash
./run_pipeline_01_to_04b.sh friends_s01e03b --mode symlink
```

### Annotation Workflow (Stages 04c + 05)

After manually annotating clusters in ClusterMark:

```bash
# 1. Re-run tracking with more frames for refinement (optional but recommended)
python 03_within_scene_tracking.py <episode_id> --top-n 3

# 2. Apply annotation-based refinement
python 04c_refine_with_annotations.py <episode_id> <annotation_file.json>

# 3. Generate final character timestamps
python 05_generate_character_timestamps.py <episode_id>
```

### Batch Processing with SLURM

Three SLURM entry points cover the full dataset workflow:

#### 1. Initial clustering batch — `submit_all_episodes_batched.sh`

Throttled submitter for stages 01–04b across all 292 s1–s6 episodes. Submits
10 jobs at a time and waits 2h between batches; safe to interrupt/resume
since submission state is logged. Designed to run inside tmux.

```bash
tmux new -s batch-submit
./scripts/submit_all_episodes_batched.sh
```

#### 2. Annotation pilot — `run_pipeline_annotation_episodes.sh`

Wrapper around `run_pipeline_01_to_04b.sh` for the annotation-selected
episodes listed in `data/annotation_episodes.txt`. Historically used to
stage a representative subset for ClusterMark; still useful for re-running
a specific annotation target.

```bash
# Default array (see script header for the current episode list)
sbatch scripts/run_pipeline_annotation_episodes.sh

# Single episode by its line number in data/episode_id.txt (e.g. s01e03b = 6)
sbatch --array=6 scripts/run_pipeline_annotation_episodes.sh
```

Resources: `ou_bcs_normal,pi_satra`, 5 min, 4 GB, 1×A100.

#### 3. Post-annotation batch — `run_pipeline_04c_to_05.sh`

Per-season array task: task N iterates all episodes of season N sequentially
on a single GPU allocation. Covers s1–s6 (all reviewed annotations).

```bash
# All six seasons (default --array=1-6)
sbatch scripts/run_pipeline_04c_to_05.sh

# A single season
sbatch --array=3 scripts/run_pipeline_04c_to_05.sh

# Local test for one episode
./scripts/run_pipeline_04c_to_05.sh friends_s01e01a
```

Resources: `mit_preemptable`, 12 h (empirically ~100 s/season), 4 GB, 1×A100.

## Prerequisites

### Environment Setup

```bash
# Create virtual environment and install dependencies
uv sync
```

### Environment Variables

Create a `.env` file in the repository root:
```
SCRATCH_DIR=/path/to/intermediate/and/final/outputs
VIDEO_DIR=/path/to/input/videos
```

- `SCRATCH_DIR` — where all stage outputs are written (`output/` subtree below).
- `VIDEO_DIR` — root directory containing input `.mkv` files organized by season
  (`s1/friends_s01e01a.mkv`, ...). Resolved by `utils.get_video_path()`.

### Data Structure

Input videos (read from `$VIDEO_DIR`):
```
${VIDEO_DIR}/
├── s1/
│   ├── friends_s01e01a.mkv
│   └── ...
├── s2/
└── ...
```

Pipeline outputs (written under `$SCRATCH_DIR`):
```
${SCRATCH_DIR}/output/
├── 01_scene_detection/            # Scene boundary files (.txt)
├── 02_face_detection/             # Face detection results (.json)
├── 03_face_tracking/              # Tracked faces with selected frames
├── 04a_face_clustering/           # Clustering results + stage-04c refined JSON
├── 04b_face_tracking_by_cluster/  # Reorganized by cluster (for annotation)
├── 04c_face_tracking_by_cluster_refined/  # Only populated when 04c is run with --reorganize
└── 05_character_timestamps/       # Per-second character presence + face locations
```

## Individual Step Details

### Step 01: Scene Detection

```bash
python 01_scene_detection.py <episode_id> [--detector adaptive|content|hash]
```

Detects scene boundaries using PySceneDetect. Default detector is `adaptive`.

### Step 02: Face Detection

```bash
python 02_face_detection.py <episode_id>
```

Detects faces in all video frames using MTCNN via facenet-pytorch.

### Step 03: Within-Scene Tracking

```bash
python 03_within_scene_tracking.py <episode_id> [options]
```

**Options:**
- `--iou-threshold`: Minimum IoU for tracking (default: 0.5)
- `--max-gap`: Max missing frames before track ends (default: -1 for auto, 0.5×fps)
- `--box-expansion`: Box expansion ratio (default: 0.1)
- `--use-median-box/--no-median-box`: Use median box for stability (default: True)
- `--top-n N`: Number of frames to select per track (default: 3)
  - Use `1` for initial clustering (consistent embeddings)
  - Use `3+` for refinement with annotations
- `--no-diverse-frames`: Disable temporal diversity in frame selection

**Frame Selection Strategy:**
- **Initial clustering** (`--top-n 1 --no-diverse-frames`): Selects single best frame per track. Ensures consistent embeddings within each track.
- **Refinement** (`--top-n 3` with diverse frames): Selects frames across temporal segments for better pose/lighting coverage.

### Step 04a: Face Clustering

```bash
python 04a_face_clustering.py <episode_id> [--similarity-threshold N] [--no-batch]
```

Clusters faces using embeddings and similarity thresholding.

**Options:**
- `--similarity-threshold`: Clustering similarity threshold (default: 0.5)
- `--no-batch`: Disable batch embedding processing

### Step 04b: Reorganize by Cluster

```bash
python 04b_reorganize_by_cluster.py <episode_id> [--mode copy|move|symlink] [--create-zip]
```

Organizes face images into cluster directories for manual annotation in ClusterMark.

### Manual Annotation with ClusterMark

Between stages 04b and 04c the pipeline pauses for a human-in-the-loop step.
ClusterMark is used to inspect the 04b output and assign a character name to
each cluster (plus optional quality modifiers — see below). The annotator
uploads `<episode_id>.zip` produced by 04b, labels clusters, and exports the
annotations as JSON. That JSON is the input to stage 04c.

### Step 04c: Refine with Annotations

```bash
python 04c_refine_with_annotations.py <episode_id> <annotation_file.json>
```

Propagates manual labels from ClusterMark, merges/splits clusters, and produces refined character assignments.

### Step 05: Generate Character Timestamps

```bash
python 05_generate_character_timestamps.py <episode_id>
```

Generates per-second character presence timestamps from refined clustering.
Downstream multimodal fusion and visualization live in the te-charnet repo.

## Output Files

Each episode generates:

| Stage | Output File |
|-------|-------------|
| 01 | `<episode_id>.txt` - Scene boundaries |
| 02 | `<episode_id>.json` - Face detection results |
| 03 | `<episode_id>_tracked_faces.json` - Tracking results |
| 03 | `<episode_id>_selected_frames_per_face.json` - Selected frames |
| 04a | `<episode_id>_matched_faces_with_clusters.json` - Clustering results |
| 04b | `<episode_id>/` - Directory with cluster subdirectories |
| 04b | `<episode_id>.zip` - ZIP file for ClusterMark upload |
| 04c | `<episode_id>_matched_faces_with_clusters_refined.json` - Refined clusters (written to 04a dir) |
| 05 | `<episode_id>_timestamps.{json,csv}` - Character timestamps |
| 05 | `<episode_id>_face_locations.json` - Per-frame bounding boxes |

## Embedding Models

Two embedding models are supported:

| Model | Description | Dimensions | Input Size |
|-------|-------------|------------|------------|
| `vggface2` | InceptionResnetV1 on VGGFace2 | 512-dim | 160×160 |
| `buffalo_l` | InsightFace buffalo_l (ArcFace) | 512-dim | 112×112 |

**IMPORTANT:** Embedding models are incompatible. Switching models requires re-running stages 04a, 04c, and 05 for all episodes.

## Annotation Workflow with ClusterMark

1. **Run stages 01-04b** to generate cluster folders and ZIP file
2. **Upload to ClusterMark** and annotate clusters (assign character labels)
3. **Export annotations** as JSON from ClusterMark
4. **Run stage 04c** to propagate labels and refine clustering
5. **Run stage 05** to generate final character timestamps

### Quality Modifiers

You can add quality modifiers to labels (e.g., `rachel @poor`, `monica @blurry`) to down-weight faces during refinement:

- `@poor` - Low quality face (blurry, dark, or extreme angle)
- `@blurry` - Motion blur or out of focus
- `@dark` - Poorly lit face
- `@profile` - Side view or extreme angle
- `@back` - Back of head or not visible

## Testing

```bash
pytest                    # All tests
pytest -k test_name       # Specific test
```

## Project Structure

```
char-tracker/
├── scripts/               # Pipeline stage scripts
│   ├── 01_scene_detection.py
│   ├── 02_face_detection.py
│   ├── 03_within_scene_tracking.py
│   ├── 04a_face_clustering.py
│   ├── 04b_reorganize_by_cluster.py
│   ├── 04c_refine_with_annotations.py
│   ├── 05_generate_character_timestamps.py
│   ├── run_pipeline_01_to_04b.sh            # local driver for stages 01-04b
│   ├── run_pipeline_annotation_episodes.sh  # SLURM wrapper for annotation pilot
│   ├── run_pipeline_04c_to_05.sh            # SLURM post-annotation batch (per season)
│   └── submit_all_episodes_batched.sh       # throttled 292-episode 01-04b submitter
├── src/                   # Core modules
│   ├── scene_detector.py
│   ├── face_detector.py
│   ├── face_tracker.py
│   ├── face_clusterer.py
│   ├── cluster_refiner.py
│   ├── constants.py
│   └── utils.py
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── data/                  # Episode lists and reference data
```

## License

MIT
