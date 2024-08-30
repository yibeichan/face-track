#!/bin/bash

#SBATCH --job-name=cluster_face_match
#SBATCH --output=/om2/user/yibei/face-track/logs/%x_%j.out 
#SBATCH --error=/om2/user/yibei/face-track/logs/%x_%j.err 
#SBATCH --partition=normal
#SBATCH --exclude=node[030-070]
#SBATCH --time=00:10:00 
#SBATCH --array=1-292
#SBATCH --ntasks=1 
#SBATCH --mem=1G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/miniconda3/etc/profile.d/conda.sh
# Activate your Conda environment
conda activate face-track

TASK_FILE="/om2/user/yibei/face-track/data/episode_id.txt"

TASK_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $TASK_FILE)

echo "Processing: $TASK_ID"

cd /om2/user/yibei/face-track/scripts
python 06_cluster_face_match.py "${TASK_ID}"